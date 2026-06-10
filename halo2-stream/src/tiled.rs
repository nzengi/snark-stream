//! Tiled, windowed access over the extended domain — the index arithmetic the
//! out-of-core quotient evaluator needs.
//!
//! halo2's `evaluate_h` reads each extended coset at `get_rotation_idx(idx, rot,
//! rot_scale, isize) = (idx + rot·rot_scale).rem_euclid(size)` — rotations *wrap*
//! around the domain. To evaluate the quotient in tiles without the full cosets
//! resident, a tile `[s, s+len)` needs only the coset values in the window
//! `[s - halo, s + len + halo)` (mod `size`), where `halo = max_rotation ·
//! rot_scale` is a small constant (the circuit's largest rotation times the
//! extension factor). This module is that windowing, proven equivalent to direct
//! full-array access — including wrap at the domain boundary and tiles that don't
//! divide the domain. The [`crate::disk`] module feeds it cosets straight from
//! disk; the arithmetic here is identical either way.

use halo2curves::bn256::Fr;

/// The extended domain's tiling geometry.
#[derive(Clone, Copy, Debug)]
pub struct TileSpec {
    /// Extended-domain size (`2^extended_k`).
    pub size: usize,
    /// Extension factor `2^(extended_k - k)` — one original-domain step is this
    /// many extended-domain rows.
    pub rot_scale: i32,
    /// Window margin each side of a tile: `max_abs_rotation · rot_scale`.
    pub halo: usize,
}

impl TileSpec {
    /// `max_abs_rotation` is the largest `|rotation|` any gate/permutation/lookup
    /// term queries (in original-domain rows).
    pub fn new(size: usize, rot_scale: i32, max_abs_rotation: usize) -> Self {
        Self {
            size,
            rot_scale,
            halo: max_abs_rotation * rot_scale as usize,
        }
    }

    /// halo2's rotated index into the full extended coset.
    #[inline]
    pub fn rotation_idx(&self, idx: usize, rot: i32) -> usize {
        (idx as i32 + rot * self.rot_scale).rem_euclid(self.size as i32) as usize
    }

    /// Number of records a tile of `len` rows needs resident: `len + 2·halo`.
    #[inline]
    pub fn window_len(&self, len: usize) -> usize {
        len + 2 * self.halo
    }

    /// Build the window `[start - halo, start + len + halo)` (mod `size`) of a
    /// full in-RAM coset. ([`crate::disk::DiskCoset`] reads the same range straight
    /// from disk.)
    pub fn window(&self, coset: &[Fr], start: usize, len: usize) -> Vec<Fr> {
        let lo = start as i64 - self.halo as i64;
        (0..self.window_len(len))
            .map(|j| coset[(lo + j as i64).rem_euclid(self.size as i64) as usize])
            .collect()
    }

    /// Window-local position of row `i` of the tile under rotation `rot`. The
    /// window is laid out contiguously over `[start - halo, ...)`, so the access
    /// is wrap-free (the wrap was resolved when the window was built) as long as
    /// `|rot · rot_scale| <= halo`.
    #[inline]
    pub fn local(&self, i: usize, rot: i32) -> usize {
        (self.halo as i32 + i as i32 + rot * self.rot_scale) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::group::ff::Field;

    fn sample(n: usize) -> Vec<Fr> {
        let mut x = Fr::from(99);
        (0..n)
            .map(|_| {
                x = x * Fr::from(1103515245u64) + Fr::from(12345u64);
                x
            })
            .collect()
    }

    // A rotation-using "constraint" evaluated directly over the full coset must
    // equal the same thing evaluated tile-by-tile through the windowed access,
    // including wrap at the boundary and tiles that don't divide the domain.
    #[test]
    fn tiled_window_matches_direct() {
        let rotations: [i32; 6] = [0, 1, -1, 2, -3, 3];
        // max_rot must cover the largest |rotation| used above (3).
        for &(size, rot_scale, max_rot) in &[(64usize, 4i32, 3usize), (96, 2, 3), (256, 8, 3)] {
            let spec = TileSpec::new(size, rot_scale, max_rot);
            let coset = sample(size);
            let coef = sample(rotations.len());

            // Direct: out[idx] = sum_r coef[r] * coset[rotation_idx(idx, rotations[r])].
            let direct: Vec<Fr> = (0..size)
                .map(|idx| {
                    rotations
                        .iter()
                        .zip(&coef)
                        .fold(Fr::ZERO, |acc, (&rot, &c)| {
                            acc + c * coset[spec.rotation_idx(idx, rot)]
                        })
                })
                .collect();

            // Tiled: process [s, s+len) through a window, read window-local.
            for &tile in &[1usize, 7, 16, 31, size] {
                let mut tiled = vec![Fr::ZERO; size];
                let mut s = 0;
                while s < size {
                    let len = tile.min(size - s);
                    let win = spec.window(&coset, s, len);
                    for i in 0..len {
                        tiled[s + i] = rotations.iter().zip(&coef).fold(Fr::ZERO, |acc, (&rot, &c)| {
                            acc + c * win[spec.local(i, rot)]
                        });
                    }
                    s += len;
                }
                assert_eq!(direct, tiled, "size={size} rot_scale={rot_scale} tile={tile}");
            }
        }
    }
}
