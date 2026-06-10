//! Out-of-core KZG polynomial commitment.
//!
//! A KZG commitment to `f(X) = Sum_i a_i X^i` is `C = Sum_i a_i * [tau^i] g`, the
//! MSM of the coefficient vector against the powers-of-tau SRS. That is *all* it
//! is — arkworks' [`KZG10::commit`](ark_poly_commit) with no hiding term reduces
//! to a single `VariableBaseMSM::msm` over `powers_of_g`. So the SRS is the
//! prover's memory wall here too: `deg + 1` curve points, gigabytes once the
//! polynomial degree reaches `2^26..2^28`.
//!
//! Which means the out-of-core MSM from [`crate::msm`] already gives an
//! out-of-core KZG commitment, with nothing new to invent: stream the SRS from
//! disk, hold only the Pippenger buckets resident, and the commitment that comes
//! out is the *same group element* a stock in-RAM KZG prover would produce. The
//! contribution is the composition and the bit-for-bit equality with the deployed
//! path (verified in the tests against arkworks' own `KZG10::commit`), not a new
//! commitment scheme.

use std::io;
use std::path::Path;

use ark_bn254::G1Affine;
use ark_ec::CurveGroup;

use crate::msm::msm_ooc;

/// Out-of-core KZG commitment: `C = Sum_i coeffs[i] * SRS[i]`, with the
/// `num_coeffs`-point SRS streamed from `srs_file` and the coefficients from
/// `coeffs_file`, never more than `block` of each resident at once.
///
/// The SRS records are the `[tau^i] g` powers in arkworks `CanonicalSerialize`
/// uncompressed layout (what [`crate::msm::write_points`] writes); the
/// coefficients are field elements in the same layout. The returned point is
/// identical to arkworks `KZG10::commit(powers, poly, None, None)` on the same
/// SRS and polynomial — see `kzg::tests::commit_ooc_matches_kzg10`.
pub fn commit_ooc(
    srs_file: &Path,
    coeffs_file: &Path,
    num_coeffs: usize,
    block: usize,
) -> io::Result<G1Affine> {
    Ok(msm_ooc(srs_file, coeffs_file, num_coeffs, block)?.into_affine())
}

/// In-core KZG commitment over an explicit SRS, used by the demo's OOM-bound
/// path. This is the genuine deployed convention — `Sum_i coeffs[i] * srs[i]` is
/// exactly what arkworks `KZG10::commit` computes with no hiding — but with the
/// whole SRS held in RAM, so it is the one that gets OOM-killed under a tight
/// cgroup while [`commit_ooc`] survives.
pub fn commit_incore(srs: &[G1Affine], coeffs: &[ark_bn254::Fr]) -> G1Affine {
    crate::msm::msm_incore(srs, coeffs).into_affine()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    use ark_bn254::{Bn254, Fr};
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial};
    use ark_poly_commit::kzg10::{Powers, KZG10};
    use ark_std::test_rng;

    use crate::msm::{write_points, write_scalars};

    type Kzg = KZG10<Bn254, DensePolynomial<Fr>>;

    // The headline correctness check: our streamed commitment must equal what the
    // *deployed* arkworks KZG prover produces, point for point. (Internal oracle
    // only — the outward proof that it works is the cgroup demo, not this test.)
    #[test]
    fn commit_ooc_matches_kzg10() {
        let mut rng = test_rng();
        let dir = std::env::temp_dir();

        for &deg in &[1usize, 2, 7, 64, 255] {
            let pp = Kzg::setup(deg, false, &mut rng).unwrap();
            let n = deg + 1; // SRS / coefficient count

            // A real, all-nonzero-coefficient polynomial of exactly degree `deg`,
            // so KZG10::commit skips no leading zeros and the alignment is direct.
            let coeffs: Vec<Fr> = (0..n).map(|i| Fr::from(i as u64 + 1)).collect();
            let poly = DensePolynomial::from_coefficients_vec(coeffs.clone());

            let powers = Powers::<Bn254> {
                powers_of_g: Cow::Borrowed(&pp.powers_of_g),
                powers_of_gamma_g: Cow::Owned(pp.powers_of_gamma_g.values().cloned().collect()),
            };
            let (comm, _) = Kzg::commit(&powers, &poly, None, None).unwrap();

            // Persist the SRS powers and the coefficients, then commit out of core.
            let srs_path = dir.join(format!("ss_kzg_srs_{deg}"));
            let coeff_path = dir.join(format!("ss_kzg_coeff_{deg}"));
            write_points(&srs_path, &pp.powers_of_g[..n]).unwrap();
            write_scalars(&coeff_path, &coeffs).unwrap();

            // block=5 forces the streaming chunk loop to wrap for the larger degrees.
            let got = commit_ooc(&srs_path, &coeff_path, n, 5).unwrap();
            assert_eq!(
                got, comm.0,
                "out-of-core KZG != arkworks KZG10 at deg={deg}"
            );

            std::fs::remove_file(&srs_path).ok();
            std::fs::remove_file(&coeff_path).ok();
        }
    }
}
