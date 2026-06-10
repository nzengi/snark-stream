//! Out-of-core variable-base MSM over a halo2curves curve.
//!
//! Same streaming Pippenger as the arkworks side, retyped for the curves halo2
//! uses and made generic over `C: CurveAffine + SerdeObject` so a patched
//! `ParamsKZG<E>` (generic over the pairing engine) can call it for any engine —
//! `bn256` today, `bls12381` for free. One difference in shape from the arkworks
//! version: here the **scalars stay in RAM** (halo2 hands the prover the
//! polynomial coefficients as `&[C::Scalar]`) and only the **bases** — the SRS,
//! the `O(n)` part that is gigabytes — stream from disk. Buckets for every window
//! are held resident (`W * (2^c - 1)` points, independent of `n`), each base
//! routed into all windows as it passes, so the base file is read once.
//!
//! The result is the same group element as halo2curves' own [`msm_best`] on the
//! same inputs (group addition is abelian, so streaming order is irrelevant).

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use halo2curves::group::ff::PrimeField;
use halo2curves::group::prime::PrimeCurveAffine;
use halo2curves::group::Group;
use halo2curves::msm::msm_best;
use halo2curves::serde::SerdeObject;
use halo2curves::CurveAffine;
use rayon::prelude::*;

/// Window width: the classic `c ~ ln(n)` Pippenger choice, floored for tiny
/// inputs and capped to keep the bucket array bounded.
fn window_bits(n: usize) -> usize {
    if n < 4 {
        2
    } else {
        ((n as f64).ln().ceil() as usize).clamp(4, 16)
    }
}

/// Threads for the bucket scatter — capped well below a high core count because
/// the scatter is memory-bandwidth bound (it peaks around a handful of threads;
/// see the arkworks-side measurement). `min` with the core count stays sane on
/// small machines.
fn fill_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(1, |p| p.get())
        .clamp(1, 8)
}

/// The `c`-bit digit of `repr` (a little-endian scalar repr) at window `w`.
fn digit(repr: &[u8], w: usize, c: usize) -> usize {
    let start = w * c;
    let mut d = 0usize;
    for i in 0..c {
        let bit = start + i;
        let byte = bit / 8;
        if byte < repr.len() && (repr[byte] >> (bit % 8)) & 1 == 1 {
            d |= 1 << i;
        }
    }
    d
}

/// Reduce one window's buckets (`buckets[d-1]` = sum of bases with digit `d`) to
/// `Sum_{d>=1} d * buckets[d-1]` via a running sum from the top.
fn reduce_window<C: CurveAffine>(buckets: &[C::Curve]) -> C::Curve {
    let mut running = C::Curve::identity();
    let mut sum = C::Curve::identity();
    for b in buckets.iter().rev() {
        running += b;
        sum += running;
    }
    sum
}

/// In-core reference: halo2curves' own MSM. The oracle every streamed path is
/// checked against, point for point.
pub fn msm_incore<C: CurveAffine>(bases: &[C], scalars: &[C::Scalar]) -> C::Curve {
    msm_best(scalars, bases)
}

/// In-core bucketed Pippenger (the streaming structure with bases in RAM), kept
/// so the algorithm can be checked against [`msm_incore`] before any disk is
/// involved.
pub fn msm_pippenger<C: CurveAffine>(bases: &[C], scalars: &[C::Scalar]) -> C::Curve {
    assert_eq!(bases.len(), scalars.len());
    let n = bases.len();
    if n == 0 {
        return C::Curve::identity();
    }
    let c = window_bits(n);
    let nbits = C::Scalar::NUM_BITS as usize;
    let num_windows = nbits.div_ceil(c);
    let nbuckets = (1usize << c) - 1;
    let reprs: Vec<_> = scalars.iter().map(|s| s.to_repr()).collect();

    let mut acc = C::Curve::identity();
    for w in (0..num_windows).rev() {
        if w != num_windows - 1 {
            for _ in 0..c {
                acc = acc.double();
            }
        }
        let mut buckets = vec![C::Curve::identity(); nbuckets];
        for i in 0..n {
            let d = digit(reprs[i].as_ref(), w, c);
            if d != 0 {
                buckets[d - 1] += bases[i];
            }
        }
        acc += reduce_window::<C>(&buckets);
    }
    acc
}

/// Serialized record width for a base point of curve `C` (halo2curves raw
/// uncompressed layout).
fn base_size<C: SerdeObject + PrimeCurveAffine>() -> usize {
    C::generator().to_raw_bytes().len()
}

/// Write `bases` to `path` in halo2curves raw uncompressed layout — the
/// fixed-width records [`msm_ooc_bases`] streams back.
pub fn write_bases<C: SerdeObject>(path: &Path, bases: &[C]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for b in bases {
        b.write_raw(&mut w)?;
    }
    w.flush()
}

/// Read `n` bases back from a [`write_bases`] file into RAM. Used to build the
/// *stock* (in-RAM) params for the A/B prover comparison; the streaming prover
/// never calls this.
pub fn read_bases<C>(path: &Path, n: usize) -> io::Result<Vec<C>>
where
    C: SerdeObject + PrimeCurveAffine,
{
    let base_sz = base_size::<C>();
    let mut r = BufReader::new(File::open(path)?);
    let mut buf = vec![0u8; base_sz];
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        r.read_exact(&mut buf)?;
        out.push(C::from_raw_bytes_unchecked(&buf));
    }
    Ok(out)
}

/// Out-of-core variable-base MSM with the bases streamed from `bases_file` and
/// the scalars resident. Reads exactly the first `scalars.len()` base records,
/// sequentially, in `block`-sized chunks; resident set is `O(W * 2^c + block)`,
/// independent of the SRS size on disk. Equal to [`msm_incore`].
pub fn msm_ooc_bases<C>(
    bases_file: &Path,
    scalars: &[C::Scalar],
    block: usize,
) -> io::Result<C::Curve>
where
    C: CurveAffine + SerdeObject,
{
    let n = scalars.len();
    if n == 0 {
        return Ok(C::Curve::identity());
    }
    let c = window_bits(n);
    let nbits = C::Scalar::NUM_BITS as usize;
    let num_windows = nbits.div_ceil(c);
    let nbuckets = (1usize << c) - 1;
    let base_sz = base_size::<C>();
    let mut buckets = vec![C::Curve::identity(); num_windows * nbuckets];
    let reprs: Vec<_> = scalars.iter().map(|s| s.to_repr()).collect();

    let mut bf = BufReader::new(File::open(bases_file)?);
    let mut bbuf = vec![0u8; block * base_sz];

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(fill_threads())
        .build()
        .map_err(io::Error::other)?;

    let mut done = 0;
    while done < n {
        let this = block.min(n - done);
        bf.read_exact(&mut bbuf[..this * base_sz])?;

        // Decode the block's bases in parallel (compute-bound), then scatter into
        // the buckets in parallel across windows: window `w` owns the disjoint
        // slice `buckets[w*nbuckets..]`, so no two threads touch the same bucket
        // and the per-bucket order stays the file order (still bit-identical).
        let bases: Vec<C> = (0..this)
            .into_par_iter()
            .map(|k| C::from_raw_bytes_unchecked(&bbuf[k * base_sz..(k + 1) * base_sz]))
            .collect();
        let off = done;
        pool.install(|| {
            buckets
                .par_chunks_mut(nbuckets)
                .enumerate()
                .for_each(|(w, wbuckets)| {
                    for (k, base) in bases.iter().enumerate() {
                        let d = digit(reprs[off + k].as_ref(), w, c);
                        if d != 0 {
                            wbuckets[d - 1] += *base;
                        }
                    }
                });
        });
        done += this;
    }

    let mut acc = C::Curve::identity();
    for w in (0..num_windows).rev() {
        if w != num_windows - 1 {
            for _ in 0..c {
                acc = acc.double();
            }
        }
        acc += reduce_window::<C>(&buckets[w * nbuckets..(w + 1) * nbuckets]);
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::bn256::{Fr, G1Affine, G1};
    use halo2curves::group::Curve;

    fn sample(n: usize) -> (Vec<G1Affine>, Vec<Fr>) {
        // Deterministic-but-varied: bases = i*G, scalars = a small LCG. No rng dep.
        let g = G1::generator();
        let mut acc = g;
        let mut bases = Vec::with_capacity(n);
        let mut scalars = Vec::with_capacity(n);
        let mut x = Fr::from(7);
        for _ in 0..n {
            bases.push(acc.to_affine());
            acc += g;
            x = x * Fr::from(1103515245u64) + Fr::from(12345u64);
            scalars.push(x);
        }
        (bases, scalars)
    }

    #[test]
    fn pippenger_matches_msm_best() {
        for &n in &[0usize, 1, 2, 3, 4, 8, 31, 100, 257] {
            let (bases, scalars) = sample(n);
            assert_eq!(
                msm_pippenger::<G1Affine>(&bases, &scalars),
                msm_incore::<G1Affine>(&bases, &scalars),
                "mismatch at n={n}"
            );
        }
    }

    #[test]
    fn ooc_matches_msm_best() {
        let dir = std::env::temp_dir();
        for &n in &[1usize, 2, 8, 100, 257] {
            let (bases, scalars) = sample(n);
            let bp = dir.join(format!("hs_msm_b_{n}"));
            write_bases(&bp, &bases).unwrap();
            // block=7 forces the chunk loop to wrap.
            let got = msm_ooc_bases::<G1Affine>(&bp, &scalars, 7).unwrap();
            assert_eq!(got, msm_incore::<G1Affine>(&bases, &scalars), "mismatch at n={n}");
            std::fs::remove_file(&bp).ok();
        }
    }
}
