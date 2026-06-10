<!--
  snark-stream fork notice
  ------------------------
  This directory is a trimmed, vendored fork of zcash/halo2 at commit 261faac,
  kept here so the out-of-core quotient check builds without a separate checkout.
  Only the `halo2_proofs` crate is included (the only one the change touches); the
  gadgets, book, and other workspace members are omitted. The sole change from
  upstream is an out-of-core twin of the `poly::Ast` quotient evaluator
  (`evaluate_ooc`); the exact diff against 261faac is
  `../halo2-stream/integration/halo2-zcash-quotient-ooc.patch`.

  Verify:  cd halo2-zcash && SS_CHECK_OOC=1 cargo test -p halo2_proofs --release -- --nocapture
  Upstream license (MIT OR Apache-2.0) is preserved in LICENSE-* / COPYING.md.
  Original zcash/halo2 README follows.
-->

# halo2

## Usage

This repository contains the [halo2_proofs](https://github.com/zcash/halo2/blob/main/halo2_proofs/README.md) and
[halo2_gadgets](https://github.com/zcash/halo2/blob/main/halo2_gadgets/README.md) crates, which should be used directly.

## Minimum Supported Rust Version

Requires Rust **1.60** or higher.

Minimum supported Rust version can be changed in the future, but it will be done with a
minor version bump.

## Controlling parallelism

`halo2` currently uses [rayon](https://github.com/rayon-rs/rayon) for parallel computation.
The `RAYON_NUM_THREADS` environment variable can be used to set the number of threads.

You can disable `rayon` by disabling the `"multicore"` feature.
Warning! Halo2 will lose access to parallelism if you disable the `"multicore"` feature.
This will significantly degrade performance.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
