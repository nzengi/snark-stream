# Out-of-core proving for a deployed halo2 KZG/SHPLONK prover

Two halves, both **without changing the protocol, proof, or verifier** — the proof
is byte-for-byte identical to stock halo2 and verifies against the stock verifier:

- **the SRS** (`g` / `g_lagrange`) streamed out of the prover, so the bases leave RAM;
- **the quotient** evaluated tile-by-tile over the extended domain, reusing halo2's
  own gate evaluator on coset *windows*, so the extended buffers needn't be resident.

## The SRS hook

Every prover-time MSM over the SRS in halo2 (the column commitments *and* the
SHPLONK opening, which calls `self.params.commit(..)`) funnels through two methods:
`ParamsKZG::commit` / `commit_lagrange`. The patch gives `ParamsKZG` an optional
disk-backed base source and routes those two methods to
`halo2_stream::msm::msm_ooc_bases` instead of the in-RAM `engine.msm(scalars,
&self.g)`. The SHPLONK/GWC multiopen prover is hard-bound to the concrete
`ParamsKZG<E>`, so a pure external crate can't carry the full proof; this is a
small surgical fork, not a memory-model rewrite. Everything else — keygen, the
prover, SHPLONK, the transcript, the verifier — is stock.

## The quotient

`Evaluator::evaluate_h` reads each extended coset at `(idx + rot·rot_scale).rem_
euclid(size)`. `evaluate_h_ooc` (added by the same patch) evaluates the quotient in
tiles: for a tile `[s, s+len)` it builds the window `[s-halo, s+len+halo)` of each
coset (`halo = max_rotation · rot_scale`, a small constant) and calls halo2's
**unchanged** gate evaluator with that window, a tile-local index, and `isize` set
to the window length — which makes the `rem_euclid` a no-op in-window, so the
existing constraint code reads the right values. The permutation, lookup, and
shuffle arguments fold in the same way (their `z(ωX)` +1, `±1`, and last-row
`-(blinding+1)` reads set the halo). The prover, under `SS_CHECK_OOC_H`, recomputes
the quotient this way and asserts it byte-identical to the in-core one
(`SS_OOC_TILE` sets the tile size). Verified on a custom-gate circuit, an add/mul
chain with copy constraints, a lookup circuit, and a shuffle circuit, across tile
sizes down to one row — the proof still verifies. That is every term Halo2's
quotient has, so `evaluate_h_ooc` covers any Halo2 circuit. (The coset windows can
be sourced from disk via `halo2_stream::fft::coeff_to_extended_ooc`, itself checked
bit-identical to `EvaluationDomain::coeff_to_extended`.)

## Reproduce

The patched fork is vendored at `halo2-fork/` in the repo root, so there is
nothing to clone or apply — it is PSE halo2 at commit `198e9ae` (toolchain 1.82,
pinned in `halo2-fork/rust-toolchain`) with these changes already in place. The
file `halo2-params-streaming.patch` here is the exact diff against upstream
`198e9ae`, for review.

```sh
cd halo2-fork            # from the repo root
cargo build --release -p ooc-demo
```

Then, two processes so the one-time setup's transient SRS-in-RAM doesn't pollute
the prover's peak RSS (run from `halo2-fork/`):

```sh
BIN=./target/release/ooc-demo
# SRS hook: A/B peak RSS (two processes so the one-time setup's transient
# SRS-in-RAM doesn't pollute the prover's peak).
$BIN gen 20 /var/tmp            # trusted setup + keygen; SRS + pk to disk
$BIN prove stock  /var/tmp      # prove with the SRS loaded into RAM (stock halo2)
$BIN prove stream /var/tmp      # prove with the SRS streamed from disk

# Quotient: prove a circuit; the prover recomputes the quotient out-of-core
# (tiled, windowed) and asserts it byte-identical to in-core, then verifies.
# Default circuit (perm) is the add/mul chain WITH copy constraints, so the
# permutation argument is exercised; "mul" is custom-gates-only. SS_OOC_TILE
# forces the tile size (small tiles exercise the halo + wrap).
$BIN provecheck 12 perm         # -> "out-of-core quotient == in-core ..."  verify=OK
SS_OOC_TILE=8 $BIN provecheck 10 perm
$BIN provecheck 10 mul          # custom gates only
$BIN provecheck 8 shuffle       # shuffle argument
$BIN provecheck 10 lookup       # lookup argument

# Disk-backed quotient: fresh cosets spilled to disk, read a window per tile,
# accumulator streamed to disk — still asserted byte-identical to in-core.
$BIN provedisk 10 perm
SS_OOC_TILE=4 $BIN provedisk 8 lookup
```

## The quotient, in a bounded budget

`evaluate_h_ooc` still materialises the cosets in RAM; `evaluate_h_ooc_disk` (same
patch) spills each fresh coset to a scratch file one at a time, reads back a window
per tile, and streams the accumulator to disk — so it holds at most one coset plus
an `O(tile)` window, never the whole extended domain. `qmem` runs the quotient *in
isolation* (a wide custom-gate circuit, advice filled with pseudo-random values — a
memory benchmark, not a valid proof) so its peak RSS is the quotient's alone:

```sh
# no cgroup: compare peak RSS (k=18, 96 advice columns, tile=8192)
$BIN qmem stock 18 96 8192     # -> peakRSS 2458 MB
$BIN qmem disk  18 96 8192     # -> peakRSS  939 MB, identical fingerprint

# under an enforced cgroup (swap off): in-core OOM-kills, disk fits
CG="systemd-run --user --scope -q -p MemoryMax=1536M -p MemorySwapMax=0 --"
$CG $BIN qmem stock 18 96 8192   # -> Killed (exit 137)
$CG $BIN qmem disk  18 96 8192   # -> 939 MB, fingerprint 23dd8f9ed0effe87
```

| quotient | budget | peak RSS | same `h` |
|----------|-------:|---------:|:--------:|
| in-core (`evaluate_h`)            | 1536 MB | 2458 MB → **OOM-killed** | ✓ `23dd…fe87` |
| disk (`evaluate_h_ooc_disk`)      | 1536 MB | **939 MB** | ✓ `23dd…fe87` |

`qmem-demo.sh` is the exact two-process run captured for the README GIF
(`docs/qmem-demo.gif`), polling each worker's RSS live. The 939 MB is mostly the
resident witness (the advice columns themselves); the
quotient's *own* allocation falls from ~1.7 GB of extended cosets to one coset plus
the tile. Inside Halo2's full prover this saving does **not** lower the process peak
— the quotient is not the binding phase there, and glibc keeps earlier phases'
high-water mark — which is why the bounded run is the quotient in isolation, the same
way the SRS streaming win was ~5 % of a full prove. (`provedisk` / `qmem`
honour `SS_OOC_TILE`, `SS_OOC_DIR`; `qmem` takes `<k> <cols> <tile>`.)

## Result (BN254, this circuit)

| k | prover | peak RSS | proof | verify | time |
|---|--------|---------:|-------|:------:|-----:|
| 2^20 | stock (SRS in RAM)   |  2762 MB | `1cd85508ad28194c` | OK | 10.8 s |
| 2^20 | stream (SRS on disk) |  2627 MB | `1cd85508ad28194c` | OK | 20.7 s |
| 2^22 | stock (SRS in RAM)   | 10980 MB | `ab517ff69d689d35` | OK | 39.3 s |
| 2^22 | stream (SRS on disk) | 10378 MB | `ab517ff69d689d35` | OK | 80.9 s |

**Identical proof, both verify.** Streaming the SRS cuts peak RSS by 135 MB at
k=20 and 602 MB at k=22 — each ≈ the SRS itself (`g` + `g_lagrange` =
2 × 2^k × 64 B: 128 MB, 512 MB), scaling ~4× with k as the SRS does. So the hook
does precisely what it claims, bit-identically.

But the saving stays ~5 % of the prover peak: the **proving key** (1.7 GB at k=20,
6.8 GB at k=22 — fixed columns in extended-coset form, `l0/l_last/l_active`,
permutation cosets) plus the witness and the extended-domain quotient dominate.
Streaming is ~2× slower (the SRS is re-read per commitment) — the expected
"trade time for RAM". And it does not meaningfully raise the max k before OOM,
because the proving key, not the SRS, is the binding wall.

**Honest conclusion:** out-of-core MSM alone does *not* make halo2 bounded-RAM —
it removes the SRS wall (a real, bit-identical, protocol-preserving win on a
deployed prover), and the next walls are the proving key and the extended-domain
FFT/quotient. Those are the subject of the rest of this document: the streaming
`coeff_to_extended` and the tiled quotient evaluator below.

## Upstream zcash/halo2 (IPA)

Everything above is the PSE KZG/SHPLONK fork. Upstream `zcash/halo2` (IPA, no
trusted setup, so no SRS) computes the quotient with a different engine: the
`poly::Ast` evaluator. The prover registers every coset with `poly::Evaluator`,
builds an `Ast` of the gate / permutation / lookup expressions, and calls
`Evaluator::evaluate`, which evaluates the `Ast` over the extended domain in
parallel chunks — `get_chunk_of_rotated` pulls each registered coset's chunk. The
wall is `Evaluator.polys`: every coset resident at once.

That engine is *already* chunked, so the out-of-core mapping is even more direct.
`evaluate_ooc` (in `halo2-zcash-quotient-ooc.patch`, 215 lines across
`poly/evaluator.rs`, `poly/domain.rs`, `plonk/vanishing/prover.rs`) spills each
registered coset to a scratch file and reads only the chunk window each `Ast::Poly`
leaf needs — the same range `get_chunk_of_rotated_extended` would take, the
rotation handled as a cyclic window — so the cosets need not all be resident. No
extra dependency; it reuses the field's `PrimeField::to_repr` for the scratch
records.

Verified **byte-for-byte identical** to in-core `evaluate` across the whole
`halo2_proofs` test suite (`plonk_api` — custom gates + copy constraints + lookup —
plus every unit/integration proof), via `SS_CHECK_OOC`:

The patched fork is vendored at `halo2-zcash/` in the repo root (zcash/halo2 at
`261faac`, trimmed to the `halo2_proofs` crate), so there is nothing to clone or
apply — the patch here is the exact diff against upstream, for review.

```sh
cd halo2-zcash             # from the repo root
SS_CHECK_OOC=1 cargo test -p halo2_proofs --release -- --nocapture
#   -> "SS_CHECK_OOC: out-of-core quotient == in-core, N extended values"  (every proof)
```

So the quotient tiling is not specific to the KZG fork or the per-row evaluator —
it holds on the original halo2 prover and its `Ast` evaluator too, bit-identically.
