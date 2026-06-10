//! Disk-resident extended cosets: hold a `2^extended_k` coset on disk and read
//! only a tile-sized window into RAM.
//!
//! The out-of-core quotient (`evaluate_h_ooc`) tiles the extended domain and reads
//! each coset through a window `[s - halo, s + len + halo)` (mod `size`) — see
//! [`crate::tiled`]. That version still materialises the full cosets in RAM first.
//! This module sources the same windows straight from disk, so the resident set is
//! the window, not the domain: a `DiskCoset` is the storage, and [`DiskCoset::window`]
//! reads exactly the range [`crate::tiled::TileSpec::window`] would build, with at
//! most two positioned reads to resolve the wrap. The record format is the 32-byte
//! raw `Fr` of [`crate::fft::coeff_to_extended_ooc`], so that function's on-disk
//! output can be opened directly.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;

use halo2curves::bn256::Fr;
use halo2curves::serde::SerdeObject;

const FR_SZ: usize = 32;

/// An extended coset of `size` `Fr` records on disk (raw 32-byte records). Windows
/// are read on demand; the struct itself holds only the file handle and the length.
pub struct DiskCoset {
    f: File,
    size: usize,
}

impl DiskCoset {
    /// Write `vals` to `path` as raw records and keep it open for reading.
    pub fn create(path: &Path, vals: &[Fr]) -> io::Result<Self> {
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let mut buf = vec![0u8; vals.len() * FR_SZ];
        for (e, c) in vals.iter().zip(buf.chunks_exact_mut(FR_SZ)) {
            c.copy_from_slice(&e.to_raw_bytes());
        }
        f.write_all_at(&buf, 0)?;
        Ok(Self {
            f,
            size: vals.len(),
        })
    }

    /// Open an existing coset file of `size` records (e.g. a `coeff_to_extended_ooc`
    /// output).
    pub fn open(path: &Path, size: usize) -> io::Result<Self> {
        let f = OpenOptions::new().read(true).open(path)?;
        Ok(Self { f, size })
    }

    /// Number of records.
    pub fn size(&self) -> usize {
        self.size
    }

    fn read_contig(&self, off: usize, count: usize) -> io::Result<Vec<Fr>> {
        let mut buf = vec![0u8; count * FR_SZ];
        self.f.read_exact_at(&mut buf, (off * FR_SZ) as u64)?;
        Ok(buf
            .chunks_exact(FR_SZ)
            .map(Fr::from_raw_bytes_unchecked)
            .collect())
    }

    /// The window `[start - halo, start - halo + wl)` taken mod `size` — exactly the
    /// range [`crate::tiled::TileSpec::window`] builds for a tile at `start` with
    /// `wl = len + 2·halo`. Reads contiguous runs, wrapping at the domain boundary,
    /// so the resident set is `wl` records. For a real out-of-core tile `wl ≤ size`,
    /// so it is one or two positioned reads; it stays correct for `wl > size` (the
    /// degenerate full-domain tile) by wrapping more than once.
    pub fn window(&self, start: usize, halo: usize, wl: usize) -> io::Result<Vec<Fr>> {
        let mut out = Vec::with_capacity(wl);
        let mut cur = (start as i64 - halo as i64).rem_euclid(self.size as i64) as usize;
        let mut remaining = wl;
        while remaining > 0 {
            let run = remaining.min(self.size - cur);
            out.extend(self.read_contig(cur, run)?);
            remaining -= run;
            cur += run;
            if cur == self.size {
                cur = 0;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiled::TileSpec;

    fn sample(n: usize) -> Vec<Fr> {
        let mut x = Fr::from(7);
        (0..n)
            .map(|_| {
                x = x * Fr::from(6364136223846793005u64) + Fr::from(1442695040888963407u64);
                x
            })
            .collect()
    }

    // A window read from disk must equal the in-RAM window TileSpec builds, for
    // every tile origin including those whose window wraps the domain boundary.
    #[test]
    fn disk_window_matches_tilespec() {
        let dir = std::env::var("TMPDIR").unwrap_or_else(|_| "/var/tmp".into());
        for &(size, rot_scale, max_rot) in &[(64usize, 4i32, 3usize), (96, 2, 3), (256, 8, 5)] {
            let spec = TileSpec::new(size, rot_scale, max_rot);
            let coset = sample(size);
            let path = Path::new(&dir).join(format!("hs_disk_test_{size}.bin"));
            let disk = DiskCoset::create(&path, &coset).unwrap();
            assert_eq!(disk.size(), size);

            for &tile in &[1usize, 7, 16, 31, size] {
                let mut s = 0;
                while s < size {
                    let len = tile.min(size - s);
                    let wl = spec.window_len(len);
                    let want = spec.window(&coset, s, len);
                    let got = disk.window(s, spec.halo, wl).unwrap();
                    assert_eq!(want, got, "size={size} tile={tile} s={s}");
                    s += len;
                }
            }
            std::fs::remove_file(&path).ok();
        }
    }
}
