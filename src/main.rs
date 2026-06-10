//! Demo: out-of-core MSM (and the KZG commitment built on it) against a base set
//! bigger than the memory budget.
//!
//!   gen      <log_n> <dir>           write 2^log_n bases + scalars to <dir>
//!   incore   <log_n> <dir>           load them all, arkworks MSM (OOMs under a tight cgroup)
//!   ooc      <log_n> <dir> [block]   streaming msm_ooc (survives; prints RSS)
//!   kzgincore <log_n> <dir>          in-RAM KZG commit over the same files (OOMs)
//!   kzgooc    <log_n> <dir> [block]  out-of-core KZG commit (survives; prints RSS)
//!
//! The point is to run the in-core and out-of-core forms under the same memory
//! cgroup and watch the in-core one get OOM-killed while the streaming one
//! finishes. The `kzg*` modes treat the base file as a powers-of-tau SRS and the
//! scalar file as polynomial coefficients: the commitment `Sum_i coeff_i * SRS_i`
//! is exactly what arkworks' deployed `KZG10::commit` computes (verified
//! bit-identical in `kzg::tests`), here run with the SRS streamed off disk.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read};
use std::path::Path;
use std::time::Instant;

use ark_bn254::{Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress};

use snark_stream::kzg::{commit_incore, commit_ooc};
use snark_stream::msm::{msm_incore, msm_ooc};

fn rss_mb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
        })
        .map_or(0, |kb| kb / 1024)
}

fn fingerprint(q: &G1Projective) -> String {
    fingerprint_affine(&q.into_affine())
}

fn fingerprint_affine(q: &G1Affine) -> String {
    let mut bytes = Vec::new();
    q.serialize_compressed(&mut bytes).unwrap();
    bytes.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

fn base_path(dir: &str) -> std::path::PathBuf {
    Path::new(dir).join("ss_bases.bin")
}
fn scalar_path(dir: &str) -> std::path::PathBuf {
    Path::new(dir).join("ss_scalars.bin")
}

// Stream-write the files so generation itself stays O(1) in memory: bases are
// g, 2g, 3g, ... (distinct valid points, one addition each); scalars are 1, 2, 3, ...
fn gen(log_n: usize, dir: &str) {
    let n = 1usize << log_n;
    let g = G1Affine::generator();
    let mut acc = G1Projective::from(g);
    let mut bw = BufWriter::new(File::create(base_path(dir)).unwrap());
    let mut sw = BufWriter::new(File::create(scalar_path(dir)).unwrap());
    for i in 0..n {
        acc.into_affine().serialize_uncompressed(&mut bw).unwrap();
        Fr::from(i as u64 + 1)
            .serialize_uncompressed(&mut sw)
            .unwrap();
        acc += g;
    }
    println!("gen: wrote {n} bases + scalars to {dir}");
}

fn read_points(path: &Path, n: usize) -> Vec<G1Affine> {
    let mut r = BufReader::new(File::open(path).unwrap());
    let sz = G1Affine::generator().serialized_size(Compress::No);
    let mut out = Vec::with_capacity(n);
    let mut buf = vec![0u8; sz];
    for _ in 0..n {
        r.read_exact(&mut buf).unwrap();
        out.push(G1Affine::deserialize_uncompressed_unchecked(&buf[..]).unwrap());
    }
    out
}

fn read_scalars(path: &Path, n: usize) -> Vec<Fr> {
    let mut r = BufReader::new(File::open(path).unwrap());
    let sz = Fr::from(0u64).serialized_size(Compress::No);
    let mut out = Vec::with_capacity(n);
    let mut buf = vec![0u8; sz];
    for _ in 0..n {
        r.read_exact(&mut buf).unwrap();
        out.push(Fr::deserialize_uncompressed_unchecked(&buf[..]).unwrap());
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map_or("", String::as_str);
    let log_n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let dir = args.get(3).map_or("/var/tmp", String::as_str);
    let n = 1usize << log_n;

    match mode {
        "gen" => gen(log_n, dir),
        "incore" => {
            let t = Instant::now();
            let bases = read_points(&base_path(dir), n);
            let scalars = read_scalars(&scalar_path(dir), n);
            let q = msm_incore(&bases, &scalars);
            println!(
                "incore  n=2^{log_n}  time={:.2}s  peakRSS={} MB  result={}",
                t.elapsed().as_secs_f64(),
                rss_mb(),
                fingerprint(&q)
            );
        }
        "ooc" => {
            let block: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1 << 14);
            let t = Instant::now();
            let q = msm_ooc(&base_path(dir), &scalar_path(dir), n, block).expect("msm_ooc");
            println!(
                "ooc     n=2^{log_n}  time={:.2}s  peakRSS={} MB  result={}",
                t.elapsed().as_secs_f64(),
                rss_mb(),
                fingerprint(&q)
            );
        }
        "kzgincore" => {
            let t = Instant::now();
            let srs = read_points(&base_path(dir), n);
            let coeffs = read_scalars(&scalar_path(dir), n);
            let c = commit_incore(&srs, &coeffs);
            println!(
                "kzgincore  deg=2^{log_n}-1  time={:.2}s  peakRSS={} MB  commit={}",
                t.elapsed().as_secs_f64(),
                rss_mb(),
                fingerprint_affine(&c)
            );
        }
        "kzgooc" => {
            let block: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1 << 14);
            let t = Instant::now();
            let c = commit_ooc(&base_path(dir), &scalar_path(dir), n, block).expect("commit_ooc");
            println!(
                "kzgooc     deg=2^{log_n}-1  time={:.2}s  peakRSS={} MB  commit={}",
                t.elapsed().as_secs_f64(),
                rss_mb(),
                fingerprint_affine(&c)
            );
        }
        _ => eprintln!("usage: gen|incore|ooc|kzgincore|kzgooc <log_n> <dir> [block]"),
    }
}
