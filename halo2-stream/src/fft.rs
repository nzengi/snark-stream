//! Out-of-core FFT over the scalar field (halo2curves `bn256::Fr`).
//!
//! The extended-domain FFT (`coeff_to_extended`) is a deployed Halo2 prover's
//! largest transient — the quotient is evaluated over the blown-up domain. The
//! four-step (Bailey) decomposition is what lets it stream: it factors a
//! size-`N = n1·n2` transform into small transforms over the rows and columns
//! of an `n1 × n2` matrix, with a twiddle multiply between. Each small transform
//! touches only one row or column, so the working set is `O(sqrt N)`, not `O(N)`.
//!
//! The small transforms are computed with halo2curves' own [`best_fft`], and the
//! sub-roots are derived as powers of the caller's `omega` (`omega^n1`, `omega^n2`),
//! so the four-step output is *bit-for-bit* `best_fft(_, omega, log_n)` — the exact
//! thing halo2's `EvaluationDomain` produces. Verified in the tests.

use std::fs::OpenOptions;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;

use halo2curves::bn256::Fr;
use halo2curves::fft::best_fft;
use halo2curves::group::ff::Field;
use halo2curves::serde::SerdeObject;
use rayon::prelude::*;

/// Serialized width of one `Fr` record on disk (halo2curves raw layout).
const FR_SZ: usize = 32;

/// Split `n = 2^k` into `n1·n2 = 2^k1 · 2^(k-k1)` with the exponents as balanced
/// as possible (the `O(sqrt N)` working set).
fn factor(log_n: usize) -> (usize, usize) {
    let k1 = log_n / 2;
    (1usize << k1, 1usize << (log_n - k1))
}

/// Small in-core forward FFT via halo2curves `best_fft` (so the sub-transforms
/// match halo2 exactly). `a[i] <- Sum_j a[j] omega^{ij}`, natural order.
fn small_fft(a: &mut [Fr], omega: Fr, log_len: u32) {
    best_fft(a, omega, log_len);
}

/// In-core four-step forward FFT. Produces exactly `best_fft(x, omega, log_n)`,
/// where `omega` is a primitive `2^log_n`-th root of unity — the same convention
/// halo2's `EvaluationDomain` uses. The point is the access pattern (rows then
/// columns), which the out-of-core version exploits.
///
/// Layout (matching the STARK-side convention, verified against the direct FFT):
/// - input  `x[k1 + n1·k2]` — an `n1 × n2` matrix, column index `k2`;
/// - step 1: size-`n2` FFT of each row `k1` (root `omega^n1`);
/// - step 2: multiply entry `(k1, j2)` by `omega^{k1·j2}`;
/// - step 3: size-`n1` FFT of each column `j2` (root `omega^n2`);
/// - output `y[j2 + n2·j1]`.
pub fn four_step(x: &[Fr], omega: Fr, log_n: usize) -> Vec<Fr> {
    let n = x.len();
    assert_eq!(n, 1 << log_n);
    let (n1, n2) = factor(log_n);
    let log_n1 = n1.trailing_zeros();
    let log_n2 = n2.trailing_zeros();
    let omega_row = omega.pow_vartime([n1 as u64]); // order n2
    let omega_col = omega.pow_vartime([n2 as u64]); // order n1

    // Steps 1 & 2: row transforms, then twiddle. `a[k1·n2 + j2]`.
    let mut a = vec![Fr::ZERO; n];
    for k1 in 0..n1 {
        let mut row: Vec<Fr> = (0..n2).map(|k2| x[k1 + n1 * k2]).collect();
        small_fft(&mut row, omega_row, log_n2);
        // twiddle factor omega^{k1·j2}, built incrementally across j2.
        let step = omega.pow_vartime([k1 as u64]);
        let mut tw = Fr::ONE;
        for (j2, val) in row.into_iter().enumerate() {
            a[k1 * n2 + j2] = val * tw;
            tw *= step;
        }
    }

    // Step 3: column transforms, written to the transposed output index.
    let mut y = vec![Fr::ZERO; n];
    for j2 in 0..n2 {
        let mut col: Vec<Fr> = (0..n1).map(|k1| a[k1 * n2 + j2]).collect();
        small_fft(&mut col, omega_col, log_n1);
        for (j1, val) in col.into_iter().enumerate() {
            y[j2 + n2 * j1] = val;
        }
    }
    y
}

// ===========================================================================
// Out-of-core four-step FFT
// ===========================================================================
//
// The N-element matrix lives in files of 32-byte `Fr` records. Every step reads
// and writes contiguous rows; the strided accesses of the in-core version become
// explicit out-of-core transposes. Resident working set is one tile / one row
// batch, independent of N. Bit-for-bit identical to `best_fft` (it is the in-core
// four-step with the same sub-transforms, only the memory schedule differs).

fn scratch(path: &Path, len: usize) -> io::Result<std::fs::File> {
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    f.set_len((len * FR_SZ) as u64)?;
    Ok(f)
}

/// Read `len` `Fr` records back from a file (e.g. the [`coeff_to_extended_ooc`]
/// output) into RAM — for verification against the in-core oracle.
pub fn read_to_vec(path: &Path, len: usize) -> io::Result<Vec<Fr>> {
    let f = OpenOptions::new().read(true).open(path)?;
    read_elems_at(&f, 0, len)
}

fn read_elems_at(f: &std::fs::File, elem_off: usize, count: usize) -> io::Result<Vec<Fr>> {
    let mut buf = vec![0u8; count * FR_SZ];
    f.read_exact_at(&mut buf, (elem_off * FR_SZ) as u64)?;
    Ok(buf
        .chunks_exact(FR_SZ)
        .map(Fr::from_raw_bytes_unchecked)
        .collect())
}

fn write_elems_at(f: &std::fs::File, elem_off: usize, elems: &[Fr]) -> io::Result<()> {
    let mut buf = vec![0u8; elems.len() * FR_SZ];
    for (e, c) in elems.iter().zip(buf.chunks_exact_mut(FR_SZ)) {
        c.copy_from_slice(&e.to_raw_bytes());
    }
    f.write_all_at(&buf, (elem_off * FR_SZ) as u64)
}

/// Out-of-core transpose of a `rows × cols` matrix (row-major `Fr` records) into
/// a `cols × rows` matrix in `dst`, moving `block × block` tiles so the resident
/// set is `O(block²)`.
fn transpose(
    src: &std::fs::File,
    dst: &std::fs::File,
    rows: usize,
    cols: usize,
    block: usize,
) -> io::Result<()> {
    let mut i0 = 0;
    while i0 < rows {
        let bi = block.min(rows - i0);
        let mut j0 = 0;
        while j0 < cols {
            let bj = block.min(cols - j0);
            // Read a bi × bj tile from src.
            let mut tile = vec![Fr::ZERO; bi * bj];
            for i in 0..bi {
                let row = read_elems_at(src, (i0 + i) * cols + j0, bj)?;
                tile[i * bj..(i + 1) * bj].copy_from_slice(&row);
            }
            // Write the transposed tile (bj × bi) to dst (cols × rows).
            let mut colbuf = vec![Fr::ZERO; bi];
            for j in 0..bj {
                for i in 0..bi {
                    colbuf[i] = tile[i * bj + j];
                }
                write_elems_at(dst, (j0 + j) * rows + i0, &colbuf)?;
            }
            j0 += bj;
        }
        i0 += bi;
    }
    Ok(())
}

/// For each contiguous row (length `row_len`) of a `num_rows × row_len` matrix,
/// FFT it with `sub_omega` and, if `twiddle` is set, multiply entry `(row, col)`
/// by `twiddle^{row·col}`. Rows are independent, so a bounded batch is read and
/// transformed across cores; the resident set is the batch, not `num_rows`.
fn fft_rows(
    f: &std::fs::File,
    num_rows: usize,
    row_len: usize,
    log_row_len: u32,
    sub_omega: Fr,
    twiddle: Option<Fr>,
) -> io::Result<()> {
    let batch = ((1usize << 18) / row_len).clamp(1, num_rows);
    let mut r0 = 0;
    while r0 < num_rows {
        let rb = batch.min(num_rows - r0);
        let mut rows = read_elems_at(f, r0 * row_len, rb * row_len)?;
        rows.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(i, row)| {
                best_fft(row, sub_omega, log_row_len);
                if let Some(base) = twiddle {
                    let r = r0 + i;
                    let step = base.pow_vartime([r as u64]);
                    let mut tw = Fr::ONE;
                    for e in row.iter_mut() {
                        *e *= tw;
                        tw *= step;
                    }
                }
            });
        write_elems_at(f, r0 * row_len, &rows)?;
        r0 += rb;
    }
    Ok(())
}

/// The five out-of-core steps, on files whose `a` already holds the input in
/// `P[k2][k1]` (n2 × n1) row-major layout. Leaves the result in `a`, in final
/// `y[j2 + n2·j1]` (n1 × n2) row-major = natural order.
fn four_step_files(
    a: &std::fs::File,
    b: &std::fs::File,
    c: &std::fs::File,
    omega: Fr,
    log_n: usize,
    block: usize,
) -> io::Result<()> {
    let (n1, n2) = factor(log_n);
    let log_n1 = n1.trailing_zeros();
    let log_n2 = n2.trailing_zeros();
    let omega_row = omega.pow_vartime([n1 as u64]); // order n2
    let omega_col = omega.pow_vartime([n2 as u64]); // order n1
    transpose(a, b, n2, n1, block)?; // P(n2×n1) -> Q(n1×n2): row k1 contiguous
    fft_rows(b, n1, n2, log_n2, omega_row, Some(omega))?; // steps 1&2
    transpose(b, c, n1, n2, block)?; // Q -> R(n2×n1): column j2 contiguous
    fft_rows(c, n2, n1, log_n1, omega_col, None)?; // step 3
    transpose(c, a, n2, n1, block) // S(n2×n1) -> final(n1×n2) = y[j2 + n2·j1]
}

/// Out-of-core forward FFT, holding roughly `block²` + one row batch resident.
/// `dir` is where the scratch matrix files live — point it at fast storage.
/// Bit-for-bit identical to `best_fft(input, omega, log_n)`.
pub fn four_step_ooc(input: &[Fr], omega: Fr, log_n: usize, dir: &Path, block: usize) -> io::Result<Vec<Fr>> {
    let n = input.len();
    assert_eq!(n, 1 << log_n);
    let pid = std::process::id();
    let p = |s: &str| dir.join(format!("hs_fft_{pid}_{s}.bin"));
    let (pa, pb, pc) = (p("a"), p("b"), p("c"));
    let a = scratch(&pa, n)?;
    let b = scratch(&pb, n)?;
    let c = scratch(&pc, n)?;

    // input x[k1 + n1·k2] is already P[k2][k1] (n2 × n1) row-major — write as-is.
    write_elems_at(&a, 0, input)?;
    four_step_files(&a, &b, &c, omega, log_n, block)?;
    let out = read_elems_at(&a, 0, n)?;
    for path in [pa, pb, pc] {
        std::fs::remove_file(path).ok();
    }
    Ok(out)
}

/// Out-of-core `coeff_to_extended`: the extended-domain transform halo2 runs on
/// every advice / instance / permutation / lookup / shuffle polynomial inside
/// `evaluate_h`, with the **extended-size result left on disk** (`out_path`) — it
/// is one of the ~8 extended buffers the stock prover keeps in RAM at once, and
/// the one the tiled evaluator reads back a window at a time.
///
/// Mirrors `EvaluationDomain::coeff_to_extended`: multiply coefficient `i` by
/// `zeta^i` (the period-3 ZETA coset shift, `zeta` = `Fr::ZETA`), zero-pad to
/// `2^extended_k`, then forward-FFT with `extended_omega`. Only the `2^k`
/// coefficients are ever resident (plus the four-step working set); the padding
/// is free (a freshly sized file reads back as zero). Bit-for-bit identical to
/// `domain.coeff_to_extended(poly).values`.
// The parameters mirror `EvaluationDomain::coeff_to_extended` plus the scratch
// location, so they read naturally despite the count.
#[allow(clippy::too_many_arguments)]
pub fn coeff_to_extended_ooc(
    coeffs: &[Fr],
    zeta: Fr,
    extended_omega: Fr,
    k: usize,
    extended_k: usize,
    out_path: &Path,
    dir: &Path,
    block: usize,
) -> io::Result<()> {
    assert_eq!(coeffs.len(), 1 << k);
    // distribute_powers_zeta: coset_powers = [zeta, zeta^2], a[i] *= coset_powers[i%3 - 1].
    let coset = [zeta, zeta.square()];
    let mut c: Vec<Fr> = coeffs.to_vec();
    for (i, v) in c.iter_mut().enumerate() {
        let j = i % 3;
        if j != 0 {
            *v *= coset[j - 1];
        }
    }

    let ext_len = 1usize << extended_k;
    let pid = std::process::id();
    let p = |s: &str| dir.join(format!("hs_c2e_{pid}_{s}.bin"));
    let (pb, pc) = (p("b"), p("c"));
    // `a` is the output file (and scratch step-0 input); set_len zero-fills the pad.
    let a = scratch(out_path, ext_len)?;
    let b = scratch(&pb, ext_len)?;
    let c_file = scratch(&pc, ext_len)?;
    write_elems_at(&a, 0, &c)?; // first 2^k records; rest stay zero (the pad)

    four_step_files(&a, &b, &c_file, extended_omega, extended_k, block)?;

    std::fs::remove_file(pb).ok();
    std::fs::remove_file(pc).ok();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::group::ff::PrimeField;

    /// A primitive `2^log_n`-th root of unity in `Fr` (halo2's convention).
    fn omega(log_n: u32) -> Fr {
        Fr::ROOT_OF_UNITY.pow_vartime([1u64 << (Fr::S - log_n)])
    }

    fn sample(n: usize) -> Vec<Fr> {
        // Deterministic varied input via a small LCG; no rng dependency.
        let mut x = Fr::from(12345);
        (0..n)
            .map(|_| {
                x = x * Fr::from(1103515245u64) + Fr::from(12345u64);
                x
            })
            .collect()
    }

    #[test]
    fn four_step_matches_best_fft() {
        for log_n in [1u32, 2, 3, 4, 5, 6, 7, 8, 10, 12] {
            let n = 1usize << log_n;
            let w = omega(log_n);
            let x = sample(n);
            let mut direct = x.clone();
            best_fft(&mut direct, w, log_n);
            let got = four_step(&x, w, log_n as usize);
            assert_eq!(got, direct, "four_step != best_fft at log_n={log_n}");
        }
    }

    #[test]
    fn four_step_ooc_matches_best_fft() {
        let dir = std::env::temp_dir();
        // Include odd log_n (n1 != n2) and a small block to force tile-wrapping.
        for log_n in [2u32, 3, 5, 6, 9, 12] {
            let n = 1usize << log_n;
            let w = omega(log_n);
            let x = sample(n);
            let mut direct = x.clone();
            best_fft(&mut direct, w, log_n);
            let got = four_step_ooc(&x, w, log_n as usize, &dir, 4).unwrap();
            assert_eq!(got, direct, "four_step_ooc != best_fft at log_n={log_n}");
        }
    }
}
