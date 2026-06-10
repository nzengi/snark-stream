# A SNARK prover's bases don't have to be in RAM

The expensive thing in almost every SNARK prover is a multi-scalar
multiplication: take a fixed set of curve points — the SRS, or the proving key —
and a vector of field-element scalars, and compute `Σ sᵢ·Pᵢ`. The bases are the
same every time; only the scalars change. For a circuit with a few hundred million
constraints over BN254, that fixed set is tens of gigabytes of points, and on a
normal machine it lives in RAM for the whole proof.

I spent a while on the STARK side ([zk-stream](https://github.com/nzengi/zk-stream))
convincing myself that a prover's memory is a dial, not a wall — that the dominant
buffer, the NTT, can be streamed through storage with a footprint set by a tile
size instead of by the problem size. The obvious next question was whether the
same is true on the SNARK side, where the dominant buffer is the MSM bases rather
than a transform.

## Pippenger is already a streaming algorithm

It is, and you barely have to do anything to see it.

Pippenger's bucket method splits each scalar into `W` windows of `c` bits. For
each window, every base lands in one of `2^c` buckets according to its digit in
that window; you sum each bucket, reduce with a running sum, and recombine the
windows with `c` doublings between them. The buckets — `W · 2^c` points — are a
few tens of megabytes and their size doesn't depend on how many bases you have.

So if you hold every window's buckets resident at once, each base's contribution
to all `W` windows is known from its scalar alone, and you can route a base into
all its buckets the instant it arrives and then forget it. The bases get read
once, in order, in fixed-size blocks. Group addition is commutative, so the order
you read them in doesn't change the answer — the streamed result is the same group
element, bit for bit, that you'd get with everything in RAM.

That last point is the whole reason this is worth doing rather than just clever:
there is no approximation, no different proof, no soundness argument to make. It's
the identical computation with a different memory schedule.

Measured against arkworks' `VariableBaseMSM`, under a hard memory cgroup with swap
off, spilling to NVMe:

| MSM | budget | in-core | out-of-core |
|-----|-------:|--------:|------------:|
| 2²² | 256 MB | 1038 MB → killed | 99 MB |
| 2²⁶ | 512 MB | 11.3 GB → killed | 120 MB |

A 67-million-point MSM over four gigabytes of bases, computed in 120 MB, producing
the same commitment the in-core path produces (I checked by running the in-core
path without the cgroup and matching the fingerprint). A KZG commitment is just
this MSM against the powers-of-tau, so an out-of-core KZG commitment falls straight
out and matches arkworks' `KZG10::commit` exactly.

The cost is time. Streaming is slower than RAM — you're reading gigabytes off a
disk instead of touching cache. This is a lever for when RAM is the thing you
don't have, not a way to go faster.

## Does it survive a real prover?

A microbenchmark proving its own MSM is not very convincing. The test I actually
cared about: take a prover people run — PSE's Halo2, KZG with SHPLONK openings,
the BN254 stack — and make *it* stream the SRS, without changing the proof.

This turned out to be a small patch. Every MSM the Halo2 prover does over the SRS,
including the SHPLONK opening, funnels through two methods on the params object:
`commit` and `commit_lagrange`. Point those two at a streaming MSM when the bases
live on disk, leave everything else — keygen, the prover, the opening argument, the
verifier — exactly as it is, and you're done. About a hundred lines in one file.

It works. The proof comes out byte-for-byte identical to stock Halo2 and verifies
against the stock verifier. Streaming the SRS changes nothing the protocol can
see, which is the point.

Then I measured the RAM, expecting a satisfying drop, and got a lesson instead:

| k | SRS in RAM | SRS on disk |
|---|-----------:|------------:|
| 2²⁰ | 2762 MB | 2627 MB |
| 2²² | 10980 MB | 10378 MB |

The saving is real, it's bit-identical, and it's exactly the size of the SRS — 135
MB at k=20, 602 MB at k=22, scaling with the circuit as the SRS does. But it's
about five percent of the prover's footprint. The SRS is simply not where a
deployed Halo2 prover spends its memory. The proving key is — the fixed columns in
extended-coset form, the Lagrange selector polynomials, the permutation cosets, all
held at the blown-up extended-domain size — and so is the quotient polynomial. At
k=22 the proving key alone is 6.8 GB.

I could have guessed some of this from a memory breakdown, and I did sketch one
before starting. But the breakdown tells you what's *claimed* to be big; the
measurement tells you what actually binds. Streaming the bases out of a real Halo2
prover, and watching the peak barely move, is a much sharper statement than a
table of estimates: on a univariate KZG prover, the MSM bases are a solved problem
the moment you're willing to stream them, and the wall everyone actually hits is
the proving key and the FFT.

## What that means

Out-of-core MSM is a clean, honest win on its own terms — it removes the SRS from a
deployed prover's RAM, bit-identically, without touching the proof or the verifier,
and the contribution is the engineering, not a new scheme. For provers whose
memory really is dominated by the bases, or for the standalone MSM/KZG case, that's
the whole story, and it's a big one: gigabytes down to a flat hundred-odd
megabytes.

For making *Halo2 itself* fit in a small budget, the same streaming idea has to
reach the other buffers: the proving key's extended-domain cosets, and the
extended-domain FFT (`coeff_to_extended`) the prover runs on every advice and
permutation polynomial. Those primitives are now built and checked bit-for-bit
against Halo2's own: the four-step transform from the STARK side ported to the
scalar field (identical to `best_fft`), and `coeff_to_extended` with its extended
output left on disk (identical to `EvaluationDomain`'s).

I expected the integration to be a rewrite of the prover's memory model. It isn't.
The quotient evaluator reads every coset at `(idx + rot·rot_scale) mod size`, so
if you hand it a *window* of the coset and a tile-local index — with `size` set to
the window length, which makes the modulo a no-op inside the window — the existing
constraint code runs unchanged on bounded memory. The rotations are small, so the
window is the tile plus a small margin.

So I tried it. `evaluate_h_ooc` evaluates the quotient over the extended domain in
tiles, builds each tile's coset windows, and calls Halo2's own gate evaluator on
them — the evaluator untouched — then folds in the permutation, lookup, and shuffle
arguments the same way. On real KZG/SHPLONK proofs — a custom-gate circuit, an
add/mul chain with copy constraints, a lookup circuit, and a shuffle circuit — the
out-of-core quotient comes out byte-for-byte identical to the in-core one, down to a
one-row tile, and the proof verifies. The check runs inside the prover, so it's on
the real inputs, not a reconstruction. (The permutation's last-row read and the
lookup's `±1` rotations are genuine, so the small-tile runs exercise the wrapping
halo end to end, not just in a unit test.) That's every term Halo2's quotient has —
custom gates, permutation, lookup, shuffle — so `evaluate_h_ooc` now produces the
quotient of *any* Halo2 circuit out of core.

So the dial turns the rest of the way too. The bases were the easy half and came
out for free; the quotient is the harder half, and it comes out the same — tiled,
bit-identical, reusing the prover's own evaluator. Not a rewrite after all.

The last step is to make the tiling actually *bound* the memory, not just compute
the right answer. `evaluate_h_ooc_disk` spills each fresh extended coset to a scratch
file one at a time and reads back a window per tile, and streams the accumulator out
the same way — so where the in-core evaluator holds every coset and the whole
accumulator at once, this holds one coset plus a tile. Run the quotient on its own
under a memory cgroup with swap off — a wide circuit at `k = 18`, 96 advice columns —
and the in-core path is OOM-killed at 1536 MB while the disk path finishes in 939 MB,
producing the identical `h` to the last byte.

One honest wrinkle worth stating plainly: drop that disk quotient back into Halo2's
*full* prover and the process peak doesn't move — the quotient isn't the phase that
binds there (the witness, held in several forms, and the commitment MSMs peak higher),
and the allocator keeps the earlier high-water mark even after the quotient frees. It's
the same lesson the SRS taught: the streamed buffer is genuinely out of core,
bit-for-bit, but whether *that* buffer is the wall depends on the prover. The bounded
number is the quotient computation itself — which is exactly the thing a prover built
out-of-core from the start gets to keep small.

---

*Code, the cgroup demo, and the Halo2 patch:
[github.com/nzengi/snark-stream](https://github.com/nzengi/snark-stream). Sibling
of the STARK-side [zk-stream](https://github.com/nzengi/zk-stream).*
