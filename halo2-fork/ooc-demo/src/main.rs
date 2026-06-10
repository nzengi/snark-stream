//! Measurement harness: a real halo2 KZG/SHPLONK prover, the SRS in RAM vs streamed.
//!
//!   gen   <k> <dir>            trusted setup + keygen; write pk + SRS files to <dir>
//!   prove stock  <k> <dir>     prove with the SRS loaded into RAM (stock halo2)
//!   prove stream <k> <dir>     prove with the SRS streamed from disk
//!
//! Run `prove stock` and `prove stream` as separate processes (so the one-time
//! setup's transient SRS-in-RAM does not pollute the prover's peak-RSS), then
//! compare their peak RSS — and check the proofs are byte-for-byte identical (the
//! prover uses a deterministic rng, so the only difference is the MSM path).
//!
//! The circuit is the standard PLONK add/mul chain that fills ~2^k rows, so the
//! prover does full-size MSMs over the 2^k SRS regardless of how sparse it is.

use std::io::{BufReader, BufWriter, Read, Write};
use std::marker::PhantomData;
use std::path::Path;
use std::time::Instant;

use group::ff::Field;
use halo2_backend::plonk::verifier::verify_proof_multi;
use halo2_debug::test_rng;
use halo2_proofs::circuit::{Cell, Layouter, SimpleFloorPlanner, Value};
use halo2_proofs::plonk::*;
use halo2_proofs::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::poly::Rotation;
use halo2_proofs::transcript::{
    Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
};
use halo2_proofs::SerdeFormat;
use halo2curves::bn256::{Bn256, Fr, G1Affine, G2Affine};
use halo2curves::serde::SerdeObject;

const BLOCK: usize = 1 << 14;

// ---- the circuit: PLONK add/mul chain filling ~2^k rows (from the halo2 bench) ----

#[derive(Clone)]
struct PlonkConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    sa: Column<Fixed>,
    sb: Column<Fixed>,
    sc: Column<Fixed>,
    sm: Column<Fixed>,
}

trait StandardCs<FF: Field> {
    fn raw_multiply<F>(
        &self,
        layouter: &mut impl Layouter<FF>,
        f: F,
    ) -> Result<(Cell, Cell, Cell), ErrorFront>
    where
        F: FnMut() -> Value<(Assigned<FF>, Assigned<FF>, Assigned<FF>)>;
    fn raw_add<F>(
        &self,
        layouter: &mut impl Layouter<FF>,
        f: F,
    ) -> Result<(Cell, Cell, Cell), ErrorFront>
    where
        F: FnMut() -> Value<(Assigned<FF>, Assigned<FF>, Assigned<FF>)>;
    fn copy(&self, layouter: &mut impl Layouter<FF>, a: Cell, b: Cell) -> Result<(), ErrorFront>;
}

#[derive(Clone)]
struct MyCircuit<F: Field> {
    a: Value<F>,
    k: u32,
}

struct StandardPlonk<F: Field> {
    config: PlonkConfig,
    _marker: PhantomData<F>,
}

impl<FF: Field> StandardPlonk<FF> {
    fn new(config: PlonkConfig) -> Self {
        StandardPlonk {
            config,
            _marker: PhantomData,
        }
    }
}

impl<FF: Field> StandardCs<FF> for StandardPlonk<FF> {
    fn raw_multiply<F>(
        &self,
        layouter: &mut impl Layouter<FF>,
        mut f: F,
    ) -> Result<(Cell, Cell, Cell), ErrorFront>
    where
        F: FnMut() -> Value<(Assigned<FF>, Assigned<FF>, Assigned<FF>)>,
    {
        layouter.assign_region(
            || "raw_multiply",
            |mut region| {
                let mut value = None;
                let lhs = region.assign_advice(
                    || "lhs",
                    self.config.a,
                    0,
                    || {
                        value = Some(f());
                        value.unwrap().map(|v| v.0)
                    },
                )?;
                let rhs =
                    region.assign_advice(|| "rhs", self.config.b, 0, || value.unwrap().map(|v| v.1))?;
                let out =
                    region.assign_advice(|| "out", self.config.c, 0, || value.unwrap().map(|v| v.2))?;
                region.assign_fixed(|| "a", self.config.sa, 0, || Value::known(FF::ZERO))?;
                region.assign_fixed(|| "b", self.config.sb, 0, || Value::known(FF::ZERO))?;
                region.assign_fixed(|| "c", self.config.sc, 0, || Value::known(FF::ONE))?;
                region.assign_fixed(|| "a * b", self.config.sm, 0, || Value::known(FF::ONE))?;
                Ok((lhs.cell(), rhs.cell(), out.cell()))
            },
        )
    }
    fn raw_add<F>(
        &self,
        layouter: &mut impl Layouter<FF>,
        mut f: F,
    ) -> Result<(Cell, Cell, Cell), ErrorFront>
    where
        F: FnMut() -> Value<(Assigned<FF>, Assigned<FF>, Assigned<FF>)>,
    {
        layouter.assign_region(
            || "raw_add",
            |mut region| {
                let mut value = None;
                let lhs = region.assign_advice(
                    || "lhs",
                    self.config.a,
                    0,
                    || {
                        value = Some(f());
                        value.unwrap().map(|v| v.0)
                    },
                )?;
                let rhs =
                    region.assign_advice(|| "rhs", self.config.b, 0, || value.unwrap().map(|v| v.1))?;
                let out =
                    region.assign_advice(|| "out", self.config.c, 0, || value.unwrap().map(|v| v.2))?;
                region.assign_fixed(|| "a", self.config.sa, 0, || Value::known(FF::ONE))?;
                region.assign_fixed(|| "b", self.config.sb, 0, || Value::known(FF::ONE))?;
                region.assign_fixed(|| "c", self.config.sc, 0, || Value::known(FF::ONE))?;
                region.assign_fixed(|| "a * b", self.config.sm, 0, || Value::known(FF::ZERO))?;
                Ok((lhs.cell(), rhs.cell(), out.cell()))
            },
        )
    }
    fn copy(
        &self,
        layouter: &mut impl Layouter<FF>,
        left: Cell,
        right: Cell,
    ) -> Result<(), ErrorFront> {
        layouter.assign_region(|| "copy", |mut region| region.constrain_equal(left, right))
    }
}

impl<F: Field> Circuit<F> for MyCircuit<F> {
    type Config = PlonkConfig;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self {
            a: Value::unknown(),
            k: self.k,
        }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> PlonkConfig {
        meta.set_minimum_degree(5);
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        meta.enable_equality(a);
        meta.enable_equality(b);
        meta.enable_equality(c);
        let sm = meta.fixed_column();
        let sa = meta.fixed_column();
        let sb = meta.fixed_column();
        let sc = meta.fixed_column();
        meta.create_gate("Combined add-mult", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            let sa = meta.query_fixed(sa, Rotation::cur());
            let sb = meta.query_fixed(sb, Rotation::cur());
            let sc = meta.query_fixed(sc, Rotation::cur());
            let sm = meta.query_fixed(sm, Rotation::cur());
            vec![a.clone() * sa + b.clone() * sb + a * b * sm - (c * sc)]
        });
        PlonkConfig {
            a,
            b,
            c,
            sa,
            sb,
            sc,
            sm,
        }
    }

    fn synthesize(
        &self,
        config: PlonkConfig,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), ErrorFront> {
        let cs = StandardPlonk::new(config);
        for _ in 0..((1 << (self.k - 1)) - 3) {
            let a: Value<Assigned<_>> = self.a.into();
            let mut a_squared = Value::unknown();
            let (a0, _, c0) = cs.raw_multiply(&mut layouter, || {
                a_squared = a.square();
                a.zip(a_squared).map(|(a, a_squared)| (a, a, a_squared))
            })?;
            let (a1, b1, _) = cs.raw_add(&mut layouter, || {
                let fin = a_squared + a;
                a.zip(a_squared)
                    .zip(fin)
                    .map(|((a, a_squared), fin)| (a, a_squared, fin))
            })?;
            cs.copy(&mut layouter, a0, a1)?;
            cs.copy(&mut layouter, b1, c0)?;
        }
        Ok(())
    }
}

// ---- a custom-gates-only circuit (no copy/lookup/shuffle) so the
// quotient is exactly the custom-gates fold, isolating evaluate_h_ooc. ----

#[derive(Clone)]
struct MulGateConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    q: Column<Fixed>,
}

#[derive(Clone, Default)]
struct MulGate {
    k: u32,
}

impl Circuit<Fr> for MulGate {
    type Config = MulGateConfig;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self { k: self.k }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> MulGateConfig {
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        let q = meta.fixed_column();
        // No enable_equality, no lookups, no shuffles -> custom gates only.
        meta.create_gate("q*(a*b - c)", |meta| {
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            let q = meta.query_fixed(q, Rotation::cur());
            vec![q * (a * b - c)]
        });
        MulGateConfig { a, b, c, q }
    }

    fn synthesize(
        &self,
        config: MulGateConfig,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let n = 1usize << self.k;
        // Leave a comfortable margin of unconstrained rows at the end (q = 0 there,
        // so the gate is not enforced on the blinding rows).
        let usable = n.saturating_sub(16);
        layouter.assign_region(
            || "fill",
            |mut region| {
                for row in 0..usable {
                    region.assign_fixed(|| "q", config.q, row, || Value::known(Fr::ONE))?;
                    region.assign_advice(|| "a", config.a, row, || Value::known(Fr::from(2)))?;
                    region.assign_advice(|| "b", config.b, row, || Value::known(Fr::from(3)))?;
                    region.assign_advice(|| "c", config.c, row, || Value::known(Fr::from(6)))?;
                }
                Ok(())
            },
        )
    }
}

// ---- a minimal shuffle circuit (adapted from halo2's shuffle_api test) to
// exercise the shuffle term of evaluate_h_ooc. ----

#[derive(Clone)]
struct ShuffleConfig {
    input_0: Column<Advice>,
    input_1: Column<Fixed>,
    shuffle_0: Column<Advice>,
    shuffle_1: Column<Advice>,
    s_input: Selector,
    s_shuffle: Selector,
}

#[derive(Clone, Default)]
struct ShuffleCircuit;

impl Circuit<Fr> for ShuffleCircuit {
    type Config = ShuffleConfig;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> ShuffleConfig {
        let input_0 = meta.advice_column();
        let input_1 = meta.fixed_column();
        let shuffle_0 = meta.advice_column();
        let shuffle_1 = meta.advice_column();
        let s_input = meta.complex_selector();
        let s_shuffle = meta.complex_selector();
        meta.shuffle("shuffle", |meta| {
            let s_input = meta.query_selector(s_input);
            let s_shuffle = meta.query_selector(s_shuffle);
            let i0 = meta.query_advice(input_0, Rotation::cur());
            let i1 = meta.query_fixed(input_1, Rotation::cur());
            let sh0 = meta.query_advice(shuffle_0, Rotation::cur());
            let sh1 = meta.query_advice(shuffle_1, Rotation::cur());
            vec![
                (s_input.clone() * i0, s_shuffle.clone() * sh0),
                (s_input * i1, s_shuffle * sh1),
            ]
        });
        ShuffleConfig {
            input_0,
            input_1,
            shuffle_0,
            shuffle_1,
            s_input,
            s_shuffle,
        }
    }

    fn synthesize(
        &self,
        config: ShuffleConfig,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let input_0 = [1u64, 2, 4, 1].map(Fr::from);
        let input_1 = [10u64, 20, 40, 10].map(Fr::from);
        let shuffle_0 = [4u64, 1, 1, 2].map(Fr::from);
        let shuffle_1 = [40u64, 10, 10, 20].map(Fr::from);
        layouter.assign_region(
            || "input",
            |mut region| {
                for i in 0..4 {
                    region.assign_advice(|| "i0", config.input_0, i, || Value::known(input_0[i]))?;
                    region.assign_fixed(|| "i1", config.input_1, i, || Value::known(input_1[i]))?;
                    config.s_input.enable(&mut region, i)?;
                }
                Ok(())
            },
        )?;
        layouter.assign_region(
            || "shuffle",
            |mut region| {
                for i in 0..4 {
                    region.assign_advice(|| "s0", config.shuffle_0, i, || Value::known(shuffle_0[i]))?;
                    region.assign_advice(|| "s1", config.shuffle_1, i, || Value::known(shuffle_1[i]))?;
                    config.s_shuffle.enable(&mut region, i)?;
                }
                Ok(())
            },
        )
    }
}

// ---- a "wide" custom-gates circuit: many advice columns, one gate per row
// pinning the last to the sum of the rest. Custom-gates-only (no perm/lookup/
// shuffle), so the quotient's *fresh* extended cosets — one per advice column —
// are the dominant allocation, which is what the disk-backed path removes. ----

#[derive(Clone)]
struct WideConfig {
    adv: Vec<Column<Advice>>,
    q: Column<Fixed>,
}

#[derive(Clone)]
struct WideCircuit {
    k: u32,
    cols: usize,
}

impl Circuit<Fr> for WideCircuit {
    type Config = WideConfig;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> WideConfig {
        let adv: Vec<Column<Advice>> = (0..meta_cols()).map(|_| meta.advice_column()).collect();
        let q = meta.fixed_column();
        meta.create_gate("sum", |meta| {
            let q = meta.query_fixed(q, Rotation::cur());
            let last = meta.query_advice(adv[adv.len() - 1], Rotation::cur());
            let mut sum = meta.query_advice(adv[0], Rotation::cur());
            for &a in adv.iter().take(adv.len() - 1).skip(1) {
                sum = sum + meta.query_advice(a, Rotation::cur());
            }
            vec![q * (sum - last)]
        });
        WideConfig { adv, q }
    }

    fn synthesize(
        &self,
        config: WideConfig,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let n = 1usize << self.k;
        let usable = n.saturating_sub(16);
        let cols = config.adv.len();
        layouter.assign_region(
            || "fill",
            |mut region| {
                for row in 0..usable {
                    region.assign_fixed(|| "q", config.q, row, || Value::known(Fr::ONE))?;
                    let mut sum = Fr::ZERO;
                    for (j, &a) in config.adv.iter().take(cols - 1).enumerate() {
                        let v = Fr::from((row + j + 1) as u64);
                        sum += v;
                        region.assign_advice(|| "a", a, row, || Value::known(v))?;
                    }
                    region.assign_advice(|| "last", config.adv[cols - 1], row, || Value::known(sum))?;
                }
                Ok(())
            },
        )
    }
}

// Number of advice columns for WideCircuit, from SS_WIDE_COLS (default 32). Read
// at configure time so keygen and proving agree.
fn meta_cols() -> usize {
    std::env::var("SS_WIDE_COLS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32)
        .max(2)
}

// ---- a minimal lookup circuit (advice values constrained into a fixed table)
// to exercise the lookup term of evaluate_h_ooc. ----

#[derive(Clone)]
struct LookupConfig {
    advice: Column<Advice>,
    table: TableColumn,
    sel: Selector,
}

#[derive(Clone, Default)]
struct LookupCircuit {
    k: u32,
}

impl Circuit<Fr> for LookupCircuit {
    type Config = LookupConfig;
    type FloorPlanner = SimpleFloorPlanner;
    #[cfg(feature = "circuit-params")]
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self { k: self.k }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> LookupConfig {
        let advice = meta.advice_column();
        let table = meta.lookup_table_column();
        let sel = meta.complex_selector();
        meta.lookup("in-table", |meta| {
            let a = meta.query_advice(advice, Rotation::cur());
            let s = meta.query_selector(sel);
            // When the selector is off the compressed input is 0, which the table
            // contains, so unused rows lookup-validate trivially.
            vec![(s * a, table)]
        });
        LookupConfig { advice, table, sel }
    }

    fn synthesize(
        &self,
        config: LookupConfig,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        // Table small enough to fit under 2^k (leaving room for blinding rows).
        let t = (1usize << self.k).saturating_sub(32).min(256).max(4);
        layouter.assign_table(
            || "table",
            |mut table| {
                for i in 0..t {
                    table.assign_cell(
                        || "t",
                        config.table,
                        i,
                        || Value::known(Fr::from(i as u64)),
                    )?;
                }
                Ok(())
            },
        )?;
        let n = 1usize << self.k;
        let usable = n.saturating_sub(16);
        layouter.assign_region(
            || "vals",
            |mut region| {
                for row in 0..usable {
                    config.sel.enable(&mut region, row)?;
                    region.assign_advice(
                        || "a",
                        config.advice,
                        row,
                        || Value::known(Fr::from((row % t) as u64)),
                    )?;
                }
                Ok(())
            },
        )
    }
}

// ---- measurement plumbing ----

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

fn g_path(dir: &str) -> std::path::PathBuf {
    Path::new(dir).join("h2_g.bin")
}
fn gl_path(dir: &str) -> std::path::PathBuf {
    Path::new(dir).join("h2_g_lagrange.bin")
}
fn pk_path(dir: &str) -> std::path::PathBuf {
    Path::new(dir).join("h2_pk.bin")
}
fn meta_path(dir: &str) -> std::path::PathBuf {
    Path::new(dir).join("h2_meta.bin")
}

// The prover's witness + blinding are deterministic so the stock and streaming
// proofs are byte-for-byte comparable: any difference would be the MSM path.
fn circuit(k: u32) -> MyCircuit<Fr> {
    MyCircuit {
        a: Value::known(Fr::from(0x1234_5678u64)),
        k,
    }
}

fn gen(k: u32, dir: &str) {
    let mut rng = test_rng();
    let mut params = ParamsKZG::<Bn256>::setup(k, &mut rng);
    let circ = circuit(k);

    let vk = keygen_vk(&params, &circ).expect("keygen_vk");
    let pk = keygen_pk(&params, vk, &circ).expect("keygen_pk");

    let mut w = BufWriter::new(std::fs::File::create(pk_path(dir)).unwrap());
    pk.write(&mut w, SerdeFormat::RawBytes).unwrap();
    w.flush().unwrap();

    // Stash the (tiny) G2 elements the verifier needs, then move the SRS to disk.
    let (g2, s_g2) = (params.g2(), params.s_g2());
    let mut m = BufWriter::new(std::fs::File::create(meta_path(dir)).unwrap());
    m.write_all(&k.to_le_bytes()).unwrap();
    g2.write_raw(&mut m).unwrap();
    s_g2.write_raw(&mut m).unwrap();
    m.flush().unwrap();

    params
        .enable_streaming(g_path(dir), gl_path(dir), BLOCK)
        .expect("enable_streaming");

    println!("gen: k={k}, pk + SRS ({} G1 points each) written to {dir}", 1u64 << k);
}

fn read_meta(dir: &str) -> (u32, G2Affine, G2Affine) {
    let mut r = BufReader::new(std::fs::File::open(meta_path(dir)).unwrap());
    let mut kb = [0u8; 4];
    r.read_exact(&mut kb).unwrap();
    let k = u32::from_le_bytes(kb);
    let g2 = G2Affine::read_raw_unchecked(&mut r);
    let s_g2 = G2Affine::read_raw_unchecked(&mut r);
    (k, g2, s_g2)
}

fn load_pk(dir: &str, k: u32) -> ProvingKey<G1Affine> {
    let circ = circuit(k);
    let mut r = BufReader::new(std::fs::File::open(pk_path(dir)).unwrap());
    pk_read::<G1Affine, _, MyCircuit<Fr>>(&mut r, SerdeFormat::RawBytes, k, &circ, true)
        .expect("pk_read")
}

fn prove(mode: &str, dir: &str) {
    let (k, g2, s_g2) = read_meta(dir);
    let n = 1usize << k;
    let pk = load_pk(dir, k);

    let params = match mode {
        "stock" => {
            // A side: pull the whole SRS into RAM, the stock halo2 prover.
            let g = halo2_stream::msm::read_bases::<G1Affine>(&g_path(dir), n).unwrap();
            let g_lagrange = halo2_stream::msm::read_bases::<G1Affine>(&gl_path(dir), n).unwrap();
            ParamsKZG::<Bn256>::from_raw_srs(k, g, g_lagrange, g2, s_g2)
        }
        "stream" => {
            // B side: bases stay on disk, streamed in commit/commit_lagrange.
            ParamsKZG::<Bn256>::streaming_params(k, g2, s_g2, g_path(dir), gl_path(dir), BLOCK)
        }
        _ => {
            eprintln!("mode must be stock|stream");
            return;
        }
    };

    let circ = circuit(k);
    let t = Instant::now();
    let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
    create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<'_, Bn256>, _, _, _, _>(
        &params,
        &pk,
        &[circ],
        &[vec![]],
        test_rng(),
        &mut transcript,
    )
    .expect("create_proof");
    let proof = transcript.finalize();
    let elapsed = t.elapsed().as_secs_f64();

    // Verify against stock verifier params (unchanged by streaming).
    let verifier_params = params.verifier_params();
    let mut vr = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
    let ok = verify_proof_multi::<
        KZGCommitmentScheme<Bn256>,
        VerifierSHPLONK<Bn256>,
        _,
        _,
        SingleStrategy<_>,
    >(&verifier_params, pk.get_vk(), &[vec![]], &mut vr);

    let fp: String = proof.iter().take(8).map(|b| format!("{b:02x}")).collect();
    println!(
        "{mode:<6} k=2^{k}  time={elapsed:.2}s  peakRSS={} MB  verify={}  proofLen={}  proof={fp}",
        rss_mb(),
        if ok { "OK" } else { "FAIL" },
        proof.len(),
    );
}

// Build the circuit named by `kind`, run the prover (which honours whatever
// SS_CHECK_OOC_* env vars the caller set — the in-prover quotient asserts fire
// there), and return whether the proof verifies.
fn prove_circuit_check(k: u32, kind: &str) -> bool {
    prove_circuit(k, kind).0
}

// Returns (verify_ok, first-8-bytes proof fingerprint).
fn prove_circuit(k: u32, kind: &str) -> (bool, String) {
    let mut rng = test_rng();
    let params = ParamsKZG::<Bn256>::setup(k, &mut rng);
    let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
    let pk;
    if kind == "mul" {
        let circ = MulGate { k };
        let vk = keygen_vk(&params, &circ).expect("keygen_vk");
        pk = keygen_pk(&params, vk, &circ).expect("keygen_pk");
        create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<'_, Bn256>, _, _, _, _>(
            &params, &pk, &[circ], &[vec![]], test_rng(), &mut transcript,
        )
        .expect("create_proof");
    } else if kind == "shuffle" {
        let circ = ShuffleCircuit;
        let vk = keygen_vk(&params, &circ).expect("keygen_vk");
        pk = keygen_pk(&params, vk, &circ).expect("keygen_pk");
        create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<'_, Bn256>, _, _, _, _>(
            &params, &pk, &[circ], &[vec![]], test_rng(), &mut transcript,
        )
        .expect("create_proof");
    } else if kind == "lookup" {
        let circ = LookupCircuit { k };
        let vk = keygen_vk(&params, &circ).expect("keygen_vk");
        pk = keygen_pk(&params, vk, &circ).expect("keygen_pk");
        create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<'_, Bn256>, _, _, _, _>(
            &params, &pk, &[circ], &[vec![]], test_rng(), &mut transcript,
        )
        .expect("create_proof");
    } else if kind == "wide" {
        let circ = WideCircuit {
            k,
            cols: meta_cols(),
        };
        let vk = keygen_vk(&params, &circ).expect("keygen_vk");
        pk = keygen_pk(&params, vk, &circ).expect("keygen_pk");
        create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<'_, Bn256>, _, _, _, _>(
            &params, &pk, &[circ], &[vec![]], test_rng(), &mut transcript,
        )
        .expect("create_proof");
    } else {
        let circ = MyCircuit {
            a: Value::known(Fr::from(7)),
            k,
        };
        let vk = keygen_vk(&params, &circ).expect("keygen_vk");
        pk = keygen_pk(&params, vk, &circ).expect("keygen_pk");
        create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<'_, Bn256>, _, _, _, _>(
            &params, &pk, &[circ], &[vec![]], test_rng(), &mut transcript,
        )
        .expect("create_proof");
    }
    let proof = transcript.finalize();
    let fp: String = proof.iter().take(8).map(|b| format!("{b:02x}")).collect();
    let verifier_params = params.verifier_params();
    let mut vr = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
    let ok = verify_proof_multi::<
        KZGCommitmentScheme<Bn256>,
        VerifierSHPLONK<Bn256>,
        _,
        _,
        SingleStrategy<_>,
    >(&verifier_params, pk.get_vk(), &[vec![]], &mut vr);
    (ok, fp)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("gen") => {
            let k: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(18);
            let dir = args.get(3).map_or("/var/tmp", String::as_str);
            gen(k, dir);
        }
        Some("prove") => {
            let mode = args.get(2).map_or("stock", String::as_str);
            let dir = args.get(3).map_or("/var/tmp", String::as_str);
            prove(mode, dir);
        }
        Some("provecheck") => {
            // Prove a real circuit; the prover (with SS_CHECK_OOC_H) recomputes the
            // quotient out-of-core (tiled, windowed, cosets in RAM) and asserts it
            // byte-identical to the in-core one, then the proof is verified.
            std::env::set_var("SS_CHECK_OOC_H", "1");
            let k: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
            // "mul" custom-gates-only; "shuffle"/"lookup" those args; anything else
            // the add/mul chain (MyCircuit) with copy constraints (permutation).
            let kind = args.get(3).map_or("perm", String::as_str);
            let ok = prove_circuit_check(k, kind);
            println!("provecheck k=2^{k}  verify={}", if ok { "OK" } else { "FAIL" });
        }
        Some("provedisk") => {
            // Aggregate-RAM: the prover (with SS_CHECK_OOC_DISK) recomputes the
            // quotient with the fresh cosets spilled to disk and the accumulator
            // streamed to disk — never the whole extended domain resident — and
            // asserts it byte-identical to the in-core one. SS_OOC_DIR sets the
            // scratch dir, SS_OOC_TILE the tile size.
            std::env::set_var("SS_CHECK_OOC_DISK", "1");
            let k: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
            let kind = args.get(3).map_or("perm", String::as_str);
            let ok = prove_circuit_check(k, kind);
            println!("provedisk k=2^{k}  verify={}", if ok { "OK" } else { "FAIL" });
        }
        Some("qmem") => {
            // Isolated quotient benchmark: keygen a wide custom-gates pk, fill the
            // advice columns with deterministic pseudo-random values (a memory
            // benchmark — not a valid proof), and evaluate the quotient ALONE,
            // in-core or disk-backed. Peak RSS reflects the quotient: in-core holds
            // every advice extended coset at once; the disk path holds one plus an
            // O(tile) window. Run two processes (stock / disk) under a cgroup — the
            // in-core one OOM-kills while the disk one fits — and check the
            // fingerprints match.
            //   qmem stock|disk <k> <cols> [tile]
            let mode = args.get(2).map_or("stock", String::as_str);
            let disk = mode == "disk";
            let k: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(18);
            let cols: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
            let tile: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1 << 13);
            let dir = std::env::var("SS_OOC_DIR").unwrap_or_else(|_| "/var/tmp".to_string());
            std::env::set_var("SS_WIDE_COLS", cols.to_string());

            let mut rng = test_rng();
            let params = ParamsKZG::<Bn256>::setup(k, &mut rng);
            let circ = WideCircuit { k, cols };
            let vk = keygen_vk(&params, &circ).expect("keygen_vk");
            let pk = keygen_pk(&params, vk, &circ).expect("keygen_pk");
            let domain = pk.get_vk().get_domain();
            let n = 1usize << k;

            // Deterministic pseudo-random advice columns (the resident witness).
            let mut x = Fr::from(0x9e37_79b9u64);
            let advice: Vec<_> = (0..cols)
                .map(|_| {
                    let col: Vec<Fr> = (0..n)
                        .map(|_| {
                            x = x * Fr::from(6364136223846793005u64) + Fr::from(1u64);
                            x
                        })
                        .collect();
                    domain.coeff_from_vec(col)
                })
                .collect();

            let t = Instant::now();
            let fp = halo2_backend::plonk::ss_bench_quotient(
                &pk,
                &advice,
                &[],
                Fr::from(11),
                Fr::from(22),
                Fr::from(33),
                Fr::from(44),
                disk,
                tile,
                Path::new(&dir),
            );
            let fph: String = fp.iter().map(|b| format!("{b:02x}")).collect();
            println!(
                "qmem {mode:<5} k=2^{k} cols={cols} tile={tile}  peakRSS={} MB  time={:.2}s  h={fph}",
                rss_mb(),
                t.elapsed().as_secs_f64(),
            );
        }
        Some("provemem") => {
            // Aggregate-RAM measurement. With SS_OOC_REPLACE set in the environment
            // the prover computes the quotient *only* via the disk-backed path (the
            // in-core evaluate_h is skipped), so peak RSS reflects the out-of-core
            // quotient. Run two processes — with and without SS_OOC_REPLACE — and
            // compare peak RSS and the (identical) proof fingerprint.
            let k: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(18);
            let kind = args.get(3).map_or("perm", String::as_str);
            let mode = if std::env::var("SS_OOC_REPLACE").is_ok() {
                "disk "
            } else {
                "stock"
            };
            let t = Instant::now();
            let (ok, fp) = prove_circuit(k, kind);
            println!(
                "provemem {mode} k=2^{k} {kind}  peakRSS={} MB  time={:.2}s  verify={}  proof={fp}",
                rss_mb(),
                t.elapsed().as_secs_f64(),
                if ok { "OK" } else { "FAIL" },
            );
        }
        Some("fftcheck") => {
            // Verify coeff_to_extended_ooc (disk output) == halo2's coeff_to_extended.
            use halo2_proofs::poly::EvaluationDomain;
            use halo2curves::ff::WithSmallOrderMulGroup;
            let k: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
            let dir = args.get(3).map_or("/var/tmp", String::as_str);
            let domain = EvaluationDomain::<Fr>::new(4, k); // quotient degree 3 -> ext_k = k+2
            let n = 1usize << k;
            let coeffs: Vec<Fr> = (0..n)
                .map(|i| Fr::from(i as u64).square() + Fr::from(7))
                .collect();
            let poly = domain.coeff_from_vec(coeffs.clone());
            let oracle = domain.coeff_to_extended(poly).values;

            let out = Path::new(dir).join("hs_c2e_out.bin");
            halo2_stream::fft::coeff_to_extended_ooc(
                &coeffs,
                <Fr as WithSmallOrderMulGroup<3>>::ZETA,
                domain.get_extended_omega(),
                k as usize,
                domain.extended_k() as usize,
                &out,
                Path::new(dir),
                256,
            )
            .unwrap();
            let got = halo2_stream::fft::read_to_vec(&out, domain.extended_len()).unwrap();
            std::fs::remove_file(&out).ok();
            println!(
                "fftcheck k=2^{k} ext_k={} ext_len={}  match={}",
                domain.extended_k(),
                domain.extended_len(),
                if got == oracle { "OK" } else { "FAIL" }
            );
        }
        Some("selftest") => {
            // Canonical halo2 flow, one process, no disk/streaming — isolates the
            // harness from the streaming patch.
            let k: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12);
            let mut rng = test_rng();
            let params = ParamsKZG::<Bn256>::setup(k, &mut rng);
            let circ = circuit(k);
            let vk = keygen_vk(&params, &circ).expect("keygen_vk");
            let pk = keygen_pk(&params, vk, &circ).expect("keygen_pk");
            let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
            create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<'_, Bn256>, _, _, _, _>(
                &params,
                &pk,
                &[circ],
                &[vec![]],
                test_rng(),
                &mut transcript,
            )
            .expect("create_proof");
            let proof = transcript.finalize();
            let verifier_params = params.verifier_params();
            let mut vr = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
            let ok = verify_proof_multi::<
                KZGCommitmentScheme<Bn256>,
                VerifierSHPLONK<Bn256>,
                _,
                _,
                SingleStrategy<_>,
            >(&verifier_params, pk.get_vk(), &[vec![]], &mut vr);
            println!("selftest k=2^{k}  verify={}", if ok { "OK" } else { "FAIL" });
        }
        _ => eprintln!("usage: gen <k> <dir> | prove <stock|stream> <dir>"),
    }
}
