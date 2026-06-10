//! Out-of-core primitives for a deployed halo2 KZG prover, used by the patch in
//! `integration/`. The proof stays bit-identical; only where the buffers live
//! changes.
//!
//! - [`msm`] streams the powers-of-tau from disk. Every prover-time MSM over the
//!   SRS funnels through `ParamsKZG::commit` / `commit_lagrange`, so the patch
//!   routes those to [`msm::msm_ooc_bases`] when the bases are disk-backed and the
//!   base footprint stops tracking `2^k`.
//! - [`fft`] is the four-step `coeff_to_extended`, leaving the extended-domain
//!   result on disk; [`tiled`] is the window arithmetic the quotient evaluator
//!   reads through; [`disk`] holds an extended coset on disk and hands back one
//!   tile-sized window at a time. Together they let the quotient be evaluated
//!   tile by tile instead of with every coset resident.

pub mod disk;
pub mod fft;
pub mod msm;
pub mod tiled;
