//! Variable-base multi-scalar multiplication, in core and out of core.
//!
//! The MSM is the universal heavy primitive under every SNARK (KZG commit,
//! Groth16, PLONK/Halo2, folding), and the bases are the prover's biggest memory
//! consumer: the SRS / proving key is `O(n)` curve points, gigabytes at scale.
//! So the out-of-core lever is to keep the bases on disk and stream them once.
//!
//! ## Why Pippenger is naturally out-of-core
//!
//! Pippenger's bucket method splits each scalar into `W = ceil(bits / c)` windows
//! of `c` bits. For window `j`, base `P_i` lands in bucket `digit(s_i, j)`; the
//! window sum is `Sum_d d * bucket[d]` (a running-sum reduction, no scalar muls),
//! and the windows recombine with `c` doublings between them.
//!
//! The key observation for streaming: a base's digits for *all* windows are known
//! from its scalar alone. So if we hold every window's buckets resident at once,
//! we touch each base exactly once and route it into all `W` buckets in a single
//! sequential pass. The bucket memory is `W * 2^c` points, which is independent of
//! `n` (tens of MB for `c ~ 16`), while the bases, the `O(n)` part, never need to
//! be resident beyond the current block. One sequential read of a multi-GB base
//! file, fixed RAM. (Group addition is abelian, so streaming order does not change
//! the result: the out-of-core point is bit-identical to the in-core one.)

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{AdditiveGroup, AffineRepr, VariableBaseMSM};
use ark_ff::{BigInteger, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress};
use ark_std::Zero;
use rayon::prelude::*;

/// In-core reference MSM (delegates to arkworks). The oracle every other path is
/// checked against, point for point.
pub fn msm_incore(bases: &[G1Affine], scalars: &[Fr]) -> G1Projective {
    G1Projective::msm(bases, scalars).expect("bases and scalars must be the same length")
}

/// Window width heuristic: `c ~ ln(n)` bits, the classic Pippenger choice, with a
/// floor for tiny inputs and a ceiling that keeps the bucket array small.
fn window_bits(n: usize) -> usize {
    if n < 4 {
        2
    } else {
        ((n as f64).ln().ceil() as usize).clamp(4, 16)
    }
}

/// Threads to use for the per-block bucket fill. The fill is a scatter into the
/// shared bucket array, so it is bound by memory bandwidth, not cores: measured
/// at `n = 2^22` it speeds up to ~3-8 threads (~2.1 s, ~2x over one thread) and
/// then *regresses* as more threads contend for bandwidth (16 threads is slower
/// than 4). So we cap well below a high core count; the `min` with the core count
/// keeps it sane on small machines.
fn fill_threads() -> usize {
    let cores = std::thread::available_parallelism().map_or(1, |p| p.get());
    cores.clamp(1, 8)
}

/// The `c`-bit digit of `scalar` (given as little-endian bits) at window `w`.
fn digit(bits: &[bool], w: usize, c: usize) -> usize {
    let mut d = 0usize;
    for k in 0..c {
        let idx = w * c + k;
        if idx < bits.len() && bits[idx] {
            d |= 1 << k;
        }
    }
    d
}

/// In-core bucketed Pippenger (M1). Same structure as the out-of-core path will
/// use, but with the bases in RAM, so it can be checked against [`msm_incore`]
/// before any streaming is involved. Result is the same group element as the
/// oracle.
pub fn msm_pippenger(bases: &[G1Affine], scalars: &[Fr]) -> G1Projective {
    assert_eq!(bases.len(), scalars.len());
    let n = bases.len();
    if n == 0 {
        return G1Projective::zero();
    }
    let c = window_bits(n);
    let nbits = Fr::MODULUS_BIT_SIZE as usize;
    let num_windows = nbits.div_ceil(c);
    let scalar_bits: Vec<Vec<bool>> = scalars
        .iter()
        .map(|s| s.into_bigint().to_bits_le())
        .collect();

    let mut acc = G1Projective::zero();
    for w in (0..num_windows).rev() {
        if w != num_windows - 1 {
            for _ in 0..c {
                acc.double_in_place();
            }
        }
        // buckets[d-1] holds the sum of bases whose window-w digit is d, d in 1..2^c
        let mut buckets = vec![G1Projective::zero(); (1 << c) - 1];
        for i in 0..n {
            let d = digit(&scalar_bits[i], w, c);
            if d != 0 {
                buckets[d - 1] += bases[i];
            }
        }
        // window sum = Sum_{d>=1} d * buckets[d-1], via a running sum from the top
        let mut running = G1Projective::zero();
        let mut window_sum = G1Projective::zero();
        for b in buckets.iter().rev() {
            running += b;
            window_sum += &running;
        }
        acc += window_sum;
    }
    acc
}

fn ser_err(e: ark_serialize::SerializationError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}

/// Write `points` to `path` as uncompressed `CanonicalSerialize` records — the
/// fixed-width layout `msm_ooc` streams back.
pub fn write_points(path: &Path, points: &[G1Affine]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for p in points {
        p.serialize_uncompressed(&mut w).map_err(ser_err)?;
    }
    w.flush()
}

/// Write `scalars` to `path` as uncompressed records.
pub fn write_scalars(path: &Path, scalars: &[Fr]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for s in scalars {
        s.serialize_uncompressed(&mut w).map_err(ser_err)?;
    }
    w.flush()
}

/// Out-of-core variable-base MSM.
///
/// Streams the `n` bases and scalars from disk in `block`-sized chunks while
/// holding every window's buckets resident (`W * (2^c - 1)` points, independent
/// of `n`). Each base is routed into all `W` windows' buckets as it passes, so
/// the files are read exactly once, sequentially; resident set is
/// `O(W * 2^c + block)`. Equal to [`msm_incore`] on the same inputs (group
/// addition is order-independent, so the streaming order does not change the
/// point).
///
/// Each block is decoded once (the scalar bits cached per base) and its buckets
/// are filled in parallel across windows: window `w` owns the disjoint bucket
/// slice `buckets[w * nbuckets ..]`, so the threads never touch the same bucket
/// and no locking is needed. The per-bucket addition order is still the file
/// order, so the result is byte-for-byte the single-threaded one.
pub fn msm_ooc(
    bases_file: &Path,
    scalars_file: &Path,
    n: usize,
    block: usize,
) -> io::Result<G1Projective> {
    if n == 0 {
        return Ok(G1Projective::zero());
    }
    let c = window_bits(n);
    let nbits = Fr::MODULUS_BIT_SIZE as usize;
    let num_windows = nbits.div_ceil(c);
    let nbuckets = (1usize << c) - 1;
    let mut buckets = vec![G1Projective::zero(); num_windows * nbuckets];

    let base_sz = G1Affine::generator().serialized_size(Compress::No);
    let scal_sz = Fr::zero().serialized_size(Compress::No);
    let mut bf = BufReader::new(File::open(bases_file)?);
    let mut sf = BufReader::new(File::open(scalars_file)?);
    let mut bbuf = vec![0u8; block * base_sz];
    let mut sbuf = vec![0u8; block * scal_sz];

    // A pool sized to the bandwidth sweet spot (see `fill_threads`), not the whole
    // machine, so the scatter does not oversubscribe memory bandwidth.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(fill_threads())
        .build()
        .map_err(io::Error::other)?;

    let mut done = 0;
    while done < n {
        let this = block.min(n - done);
        bf.read_exact(&mut bbuf[..this * base_sz])?;
        sf.read_exact(&mut sbuf[..this * scal_sz])?;

        pool.install(|| -> io::Result<()> {
            // Decode the block once, in parallel: each base becomes its point and
            // the little-endian bits of its scalar (cached so the per-window
            // routing below does not re-expand them).
            let items: Vec<(G1Affine, Vec<bool>)> = (0..this)
                .into_par_iter()
                .map(|k| {
                    let p = G1Affine::deserialize_uncompressed_unchecked(
                        &bbuf[k * base_sz..(k + 1) * base_sz],
                    )
                    .map_err(ser_err)?;
                    let s = Fr::deserialize_uncompressed_unchecked(
                        &sbuf[k * scal_sz..(k + 1) * scal_sz],
                    )
                    .map_err(ser_err)?;
                    Ok((p, s.into_bigint().to_bits_le()))
                })
                .collect::<io::Result<_>>()?;

            // Fill the buckets in parallel across windows: window `w` only ever
            // writes into its own `nbuckets`-long slice, so the chunks are disjoint
            // and the routing needs no synchronization.
            buckets
                .par_chunks_mut(nbuckets)
                .enumerate()
                .for_each(|(w, wbuckets)| {
                    for (p, bits) in &items {
                        let d = digit(bits, w, c);
                        if d != 0 {
                            wbuckets[d - 1] += p;
                        }
                    }
                });
            Ok(())
        })?;
        done += this;
    }

    let mut acc = G1Projective::zero();
    for w in (0..num_windows).rev() {
        if w != num_windows - 1 {
            for _ in 0..c {
                acc.double_in_place();
            }
        }
        let base = w * nbuckets;
        let mut running = G1Projective::zero();
        let mut window_sum = G1Projective::zero();
        for d in (0..nbuckets).rev() {
            running += &buckets[base + d];
            window_sum += &running;
        }
        acc += window_sum;
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::UniformRand;

    // M1 correctness: the bucketed Pippenger must equal the arkworks oracle.
    // (Internal correctness check only — the outward proof that it works is the
    // cgroup demo, not this test.)
    #[test]
    fn pippenger_matches_arkworks() {
        let mut rng = ark_std::test_rng();
        for &n in &[0usize, 1, 2, 3, 4, 8, 31, 100, 257] {
            let bases: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
            let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            assert_eq!(
                msm_pippenger(&bases, &scalars),
                msm_incore(&bases, &scalars),
                "mismatch at n={n}"
            );
        }
    }

    // M2 correctness: streaming msm_ooc must equal the arkworks oracle. Small
    // `block` forces the chunk loop to wrap. (Internal check; the outward proof
    // is the cgroup demo, not this.)
    #[test]
    fn ooc_matches_arkworks() {
        let mut rng = ark_std::test_rng();
        let dir = std::env::temp_dir();
        for &n in &[1usize, 2, 8, 100, 257] {
            let bases: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();
            let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let bp = dir.join(format!("ss_msm_b_{n}"));
            let sp = dir.join(format!("ss_msm_s_{n}"));
            write_points(&bp, &bases).unwrap();
            write_scalars(&sp, &scalars).unwrap();
            let got = msm_ooc(&bp, &sp, n, 7).unwrap();
            assert_eq!(got, msm_incore(&bases, &scalars), "mismatch at n={n}");
            std::fs::remove_file(&bp).ok();
            std::fs::remove_file(&sp).ok();
        }
    }
}
