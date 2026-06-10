//! Out-of-core SNARK proving.
//!
//! The same thesis as the STARK work (a prover's RAM is a dial, not a wall),
//! carried into the pairing/SNARK world. A SNARK prover's memory is dominated by
//! the multi-scalar multiplication (MSM) over a fixed set of curve points — the
//! SRS / proving key, `O(N)` group elements running to gigabytes. This crate
//! streams those bases from disk so peak resident memory tracks a tile size
//! rather than `N`.
//!
//! - [`mod msm`](msm) — variable-base MSM whose bases stream from disk through a
//!   fixed buffer (Pippenger buckets over a windowed base file).
//! - [`mod kzg`](kzg) — KZG polynomial commitment over an out-of-core SRS, which
//!   is just that MSM against the powers-of-tau and is bit-identical to arkworks'
//!   deployed `KZG10::commit`.
//!
//! The scalar-field FFT and the halo2 integration live in the `halo2-stream`
//! crate, written over `halo2curves::bn256::Fr`.
//!
//! Discipline (carried over verbatim): every out-of-core result is checked
//! bit-for-bit against an in-core arkworks reference, and every memory figure is
//! measured under an enforced cgroup with swap off.

pub mod kzg;
pub mod msm;
