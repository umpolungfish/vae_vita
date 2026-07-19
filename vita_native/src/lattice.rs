//! VitaLattice — the SIXTEEN_3 Logic Lattice NN Transformer, built EXACTLY to
//! the discerning ob3ect's spec (sixteen_3_logic_lattice_nn_transformer_design):
//! the forward pass IS the 26-step protocol word,
//!
//!   ⊢ ⊙ = ⊣ ◇ + > ● ◇ + + + ● × < ● ◇ + × ⊞ + × ¬ ¬ ¬ ⊣
//!
//! block for block, with the ob3ect's own topology — FSPLIT/FFUSE pairs
//! (4,7) and (8,12), fork positions [4,8,16], fuse positions [7,12,15]:
//!
//! * **VINIT (0)** — the embedding: void → stream.
//! * **IMSCRIB (1)** — self-imscription: the S¹¹ core (⊙ = criticality_phase,
//!   the system recognizing itself at threshold); its points feed λ_sic.
//! * **CLINK (2)** — trilattice meet composition: split-fuse attention (the
//!   only pairwise-composition block, μ∘δ=id pressure on fork/fuse).
//! * **TANCH (3, 25)** — the containment floor: RmsNorm anchors; the final
//!   TANCH seals into the vocab head.
//! * **FSPLIT (4)** — the ouroboricity gate forks the stream: R1 branch runs
//!   EVALT(5)·AFWD(6) — terminal-closure affirmation and forward morphism —
//!   while the carried arm holds; **FFUSE (7)** joins them (R1-dominant).
//! * **FSPLIT (8)** — the nested R2 fork: EVALT(9,10,11) are the replicative
//!   triple's three sub-checks (dim=dead, top=mime, prot=ah); **FFUSE (12)**
//!   reconstitutes the triple.
//! * **EVALF (13)** — R2 denies terminal closure (pol ≠ or′).
//! * **AREV (14)** — the reverse morphism: ROTAT phase applied BACKWARD (the
//!   polarity flip to O_inf_dag, lateral at the same shell). AFWD(6) is the
//!   same rotation forward.
//! * **FFUSE (15)** — the tier-assignment join with EMPTY fork stack: it
//!   fuses the R1/R2 OUTCOMES already in the stream (self-fuse), exactly the
//!   kernel's FFUSE-without-frame semantics.
//! * **FSPLIT (16)** — the containment fork, OPEN by the ob3ect's topology
//!   report (open_forks: 1, class open_fork_dag): its arm is never fused —
//!   **ENGAGR (19)** consumes it as the held dual, the B paradice that keeps
//!   T-on-defining and F-elsewhere both alive without collapse. EVALT(17)/
//!   EVALF(18) are the defining-/full-coordinate arms before the hold;
//!   EVALT(20)/EVALF(21) the emitted verdicts after it.
//! * **IFIX (22,23,24)** — the cascade of three permanent records (the
//!   ob3ect's Lean-fix, Rust-mapping, and gate-wiring commits): append-only
//!   residual writes, never renormalized away.
//!
//! Nothing in the schedule is tunable: the sequence, the pairs, the open
//! fork, and the cascade are pinned by tests against the ob3ect's own
//! topology report (total_pairs 2, open_forks 1, max_ifix_cascade 3,
//! sequence_length/period 26).

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{embedding, linear, linear_no_bias, ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::alphabet;
use crate::model::{Config, Core, Rotat};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Op { Vinit, Imscrib, Clink, Tanch, Fsplit, Ffuse, Evalt, Evalf, Afwd, Arev, Engagr, Ifix }

/// The protocol word, verbatim from the ob3ect's lean_scaffold.
pub const PROTOCOL: [Op; 26] = [
    Op::Vinit, Op::Imscrib, Op::Clink, Op::Tanch,
    Op::Fsplit, Op::Evalt, Op::Afwd, Op::Ffuse,
    Op::Fsplit, Op::Evalt, Op::Evalt, Op::Evalt, Op::Ffuse,
    Op::Evalf, Op::Arev, Op::Ffuse,
    Op::Fsplit, Op::Evalt, Op::Evalf, Op::Engagr, Op::Evalt, Op::Evalf,
    Op::Ifix, Op::Ifix, Op::Ifix, Op::Tanch,
];

/// CLINK: split-fuse attention (the meet), same construction as the trunk's
/// attention half — δ forks to q,k,v, μ fuses the heads, ROTAT positions.
struct Clink {
    norm: RmsNorm,
    fork: Linear,
    fuse: Linear,
    heads: usize,
}

impl Clink {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        Ok(Clink {
            norm: candle_nn::rms_norm(d, 1e-6, vb.pp("norm"))?,
            fork: linear_no_bias(d, 3 * d, vb.pp("fork"))?,
            fuse: linear_no_bias(d, d, vb.pp("fuse"))?,
            heads: cfg.heads,
        })
    }

    fn forward(&self, x: &Tensor, rotat: &Rotat, mask: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let h = self.heads;
        let hd = d / h;
        let xn = self.norm.forward(x)?;
        let qkv = self.fork.forward(&xn)?;
        let q = qkv.narrow(D::Minus1, 0, d)?;
        let k = qkv.narrow(D::Minus1, d, d)?;
        let v = qkv.narrow(D::Minus1, 2 * d, d)?;
        let shape = |z: Tensor| -> Result<Tensor> {
            z.reshape((b, t, h, hd))?.transpose(1, 2)?.contiguous()
        };
        let (q, k, v) = (shape(q)?, shape(k)?, shape(v)?);
        let q = rotat.apply(&q)?;
        let k = rotat.apply(&k)?;
        let scale = 1.0 / (hd as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let attn = ops::softmax_last_dim(&scores)?;
        let ctx = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, t, d))?;
        x + self.fuse.forward(&ctx)?
    }

    fn gate_penalty(&self) -> Result<Tensor> {
        ortho(&[self.fork.weight(), self.fuse.weight()])
    }
}

/// ‖WᵀW − I‖² pressure over a set of weights — μ∘δ = id, zero when lossless.
fn ortho(ws: &[&Tensor]) -> Result<Tensor> {
    let mut total: Option<Tensor> = None;
    for w in ws {
        let (o, i) = w.dims2()?;
        let n = o.min(i);
        let g = if o >= i { w.t()?.matmul(w)? } else { w.matmul(&w.t()?)? };
        let eye = Tensor::eye(n, w.dtype(), w.device())?;
        let p = (g - eye)?.sqr()?.mean_all()?;
        total = Some(match total { Some(t) => (t + p)?, None => p });
    }
    total.ok_or_else(|| candle_core::Error::Msg("no weights".into()))
}

/// One protocol step's parameters (only the opcodes that own any).
enum Block {
    Imscrib(Core),
    Clink(Clink),
    Tanch(RmsNorm),
    Fsplit(Linear),          // δ — the branch the stream continues on
    Ffuse(Linear),           // μ — 2d → d join (self-fuse when the stack is empty)
    Evalt(Linear),           // T-gate: x·σ(Wx+b), bias init opens the gate
    Evalf(Linear),           // F-gate: x·(1−σ(Wx+b))
    Engagr(Linear),          // the hold's fuse: residual over left ⊗ silu(held arm)
    Ifix(Linear),            // append-only residual record
    Rotat(bool),             // AFWD(true)/AREV(false) — parameterless, the ring shift itself
    Vinit,                   // the embedding, held at the lattice level
}

pub struct VitaLattice {
    embed: Embedding,
    blocks: Vec<Block>,
    head: Linear,
    rotat: Rotat,
    cfg: Config,
}

impl VitaLattice {
    pub fn new(cfg: Config, max_t: usize, vb: VarBuilder) -> Result<Self> {
        let dev = vb.device().clone();
        let d = cfg.d_model;
        let embed = embedding(cfg.vocab, d, vb.pp("embed"))?;
        let mut blocks = Vec::with_capacity(PROTOCOL.len());
        for (i, op) in PROTOCOL.iter().enumerate() {
            let vbi = vb.pp(format!("step{i}"));
            blocks.push(match op {
                Op::Vinit => Block::Vinit,
                Op::Imscrib => Block::Imscrib(Core::new(&cfg, vbi)?),
                Op::Clink => Block::Clink(Clink::new(&cfg, vbi)?),
                Op::Tanch => Block::Tanch(candle_nn::rms_norm(d, 1e-6, vbi.pp("floor"))?),
                Op::Fsplit => Block::Fsplit(linear_no_bias(d, d, vbi.pp("delta"))?),
                Op::Ffuse => Block::Ffuse(linear_no_bias(2 * d, d, vbi.pp("mu"))?),
                Op::Evalt => Block::Evalt(linear(d, d, vbi.pp("t_gate"))?),
                Op::Evalf => Block::Evalf(linear(d, d, vbi.pp("f_gate"))?),
                Op::Engagr => Block::Engagr(linear_no_bias(d, d, vbi.pp("hold"))?),
                Op::Ifix => Block::Ifix(linear_no_bias(d, d, vbi.pp("record"))?),
                Op::Afwd => Block::Rotat(true),
                Op::Arev => Block::Rotat(false),
            });
        }
        let head = linear_no_bias(d, cfg.vocab, vb.pp("head"))?;
        let rotat = Rotat::new(d / cfg.heads, max_t, &dev)?;
        Ok(VitaLattice { embed, blocks, head, rotat, cfg })
    }

    fn causal_mask(&self, t: usize, dev: &Device) -> Result<Tensor> {
        let mut m = vec![0f32; t * t];
        for i in 0..t {
            for j in (i + 1)..t {
                m[i * t + j] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(m, (1, 1, t, t), dev)
    }

    /// AFWD/AREV: the ring shift on feature pairs — one fixed ROTAT step,
    /// forward or reverse. Parameterless: the morphism is the rotation.
    fn ring_shift(&self, x: &Tensor, forward: bool) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let half = d / 2;
        let theta = if forward { 0.1f64 } else { -0.1f64 };
        let (c, s) = (theta.cos() as f32, theta.sin() as f32);
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let r1 = ((&x1 * c as f64)? - (&x2 * s as f64)?)?;
        let r2 = ((&x1 * s as f64)? + (&x2 * c as f64)?)?;
        Tensor::cat(&[r1, r2], D::Minus1)?.reshape((b, t, d))
    }

    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        Ok(self.forward_with_core(ids)?.0)
    }

    /// The 26-step protocol as dataflow. Returns logits and the IMSCRIB S¹¹
    /// points (for λ_sic). The fork stack follows kernel semantics: FSPLIT
    /// pushes the carried arm, FFUSE pops and joins (self-fuse on empty),
    /// ENGAGR consumes the OPEN fork's arm as the held dual.
    pub fn forward_with_core(&self, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b, t) = ids.dims2()?;
        let mask = self.causal_mask(t, ids.device())?;
        let mut x = self.embed.forward(ids)?;
        let mut stack: Vec<Tensor> = Vec::new();
        let mut core_z: Option<Tensor> = None;

        for block in &self.blocks {
            x = match block {
                Block::Vinit => x, // the embedding above IS this step
                Block::Imscrib(core) => {
                    let (nx, z) = core.forward(&x)?;
                    core_z = Some(z);
                    nx
                }
                Block::Clink(c) => c.forward(&x, &self.rotat, &mask)?,
                Block::Tanch(floor) => floor.forward(&x)?,
                Block::Fsplit(delta) => {
                    stack.push(x.clone()); // the carried arm
                    (&x + delta.forward(&x)?)?
                }
                Block::Ffuse(mu) => {
                    let right = stack.pop().unwrap_or_else(|| x.clone()); // empty → self-fuse
                    let joined = Tensor::cat(&[&x, &right], D::Minus1)?;
                    (&x + mu.forward(&joined)?)?
                }
                Block::Evalt(g) => {
                    let gate = ops::sigmoid(&g.forward(&x)?)?;
                    (&x * gate)?
                }
                Block::Evalf(g) => {
                    let gate = ops::sigmoid(&g.forward(&x)?)?;
                    (&x * (gate.affine(-1.0, 1.0)?))?
                }
                Block::Engagr(hold) => {
                    // The open fork's arm, held: both kept alive, neither collapsed.
                    let arm = stack.pop().unwrap_or_else(|| x.clone());
                    let held = (&x * candle_nn::ops::silu(&arm)?)?;
                    (&x + hold.forward(&held)?)?
                }
                Block::Ifix(rec) => (&x + rec.forward(&x)?)?, // append-only, never renormed
                Block::Rotat(fwd) => self.ring_shift(&x, *fwd)?,
            };
        }
        let logits = self.head.forward(&x)?;
        let z = core_z.ok_or_else(|| candle_core::Error::Msg("IMSCRIB never ran".into()))?;
        Ok((logits, z))
    }

    /// μ∘δ = id pressure across every split-fuse in the lattice: CLINK's
    /// fork/fuse and every FSPLIT δ / FFUSE μ projection.
    pub fn gate_penalty(&self) -> Result<Tensor> {
        let mut ws: Vec<&Tensor> = Vec::new();
        for b in &self.blocks {
            match b {
                Block::Fsplit(l) | Block::Ffuse(l) => ws.push(l.weight()),
                Block::Clink(c) => {
                    ws.push(c.fork.weight());
                    ws.push(c.fuse.weight());
                }
                _ => {}
            }
        }
        ortho(&ws)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn tiny() -> Config {
        Config { vocab: alphabet::vocab_size(), d_model: 48, layers: 0, heads: 12, ff_mult: 2, core_dim: 12 }
    }

    /// The schedule is the ob3ect's, exactly: sequence length/period 26,
    /// fork positions [4,8,16], fuse positions [7,12,15], balanced pairs
    /// (4,7) and (8,12), ONE open fork, IFIX cascade of 3.
    #[test]
    fn schedule_matches_the_ob3ect() {
        assert_eq!(PROTOCOL.len(), 26);
        let forks: Vec<usize> = PROTOCOL.iter().enumerate().filter(|(_, o)| **o == Op::Fsplit).map(|(i, _)| i).collect();
        let fuses: Vec<usize> = PROTOCOL.iter().enumerate().filter(|(_, o)| **o == Op::Ffuse).map(|(i, _)| i).collect();
        assert_eq!(forks, vec![4, 8, 16]);
        assert_eq!(fuses, vec![7, 12, 15]);
        // Balanced-pair scan: (4,7) and (8,12) pair; fuse 15 is unmatched
        // (the tier join), fork 16 stays open (the ENGAGR hold).
        let mut stack = Vec::new();
        let mut pairs = Vec::new();
        let mut unmatched_fuse = 0;
        for (i, op) in PROTOCOL.iter().enumerate() {
            match op {
                Op::Fsplit => stack.push(i),
                Op::Ffuse => match stack.pop() {
                    Some(s) => pairs.push((s, i)),
                    None => unmatched_fuse += 1,
                },
                _ => {}
            }
        }
        assert_eq!(pairs, vec![(4, 7), (8, 12)]);
        assert_eq!(unmatched_fuse, 1, "fuse 15 joins R1/R2 outcomes with no frame");
        assert_eq!(stack, vec![16], "fork 16 is the ob3ect's one open fork");
        let cascade = PROTOCOL.windows(3).filter(|w| w.iter().all(|o| *o == Op::Ifix)).count();
        assert_eq!(cascade, 1, "exactly one IFIX cascade of length 3");
        assert_eq!(PROTOCOL[19], Op::Engagr, "the hold consumes the open fork");
        assert_eq!(PROTOCOL[25], Op::Tanch, "the floor seals the protocol");
    }

    #[test]
    fn forward_shape_and_finiteness() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let lat = VitaLattice::new(tiny(), 64, vb).unwrap();
        let ids = Tensor::from_vec(alphabet::encode("⊢⟨+×⟩◇⊞⟨⟩●⟨T⟩⊣"), (1, 12), &dev).unwrap();
        let (logits, z) = lat.forward_with_core(&ids).unwrap();
        assert_eq!(logits.dims(), &[1, 12, alphabet::vocab_size()]);
        assert_eq!(z.dims()[2], 12, "IMSCRIB core lives in R^12");
        let v: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "logits must be finite");
    }

    #[test]
    fn causality_holds_through_the_protocol() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let lat = VitaLattice::new(tiny(), 32, vb).unwrap();
        let a = alphabet::encode("⊢⟨+⟩◇⊞⟨⟩●⟨T⟩⊣");
        let mut b = a.clone();
        let last = b.len() - 1;
        b[last] = alphabet::encode("×")[0];
        let ta = Tensor::from_vec(a.clone(), (1, a.len()), &dev).unwrap();
        let tb = Tensor::from_vec(b, (1, a.len()), &dev).unwrap();
        let la = lat.forward(&ta).unwrap();
        let lb = lat.forward(&tb).unwrap();
        let da: Vec<f32> = la.narrow(1, 0, last).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let db: Vec<f32> = lb.narrow(1, 0, last).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        for (x, y) in da.iter().zip(db.iter()) {
            assert!((x - y).abs() < 1e-4, "earlier logits moved: {x} vs {y}");
        }
    }

    #[test]
    fn gate_penalty_finite_nonnegative() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let lat = VitaLattice::new(tiny(), 32, vb).unwrap();
        let p: f32 = lat.gate_penalty().unwrap().to_scalar().unwrap();
        assert!(p.is_finite() && p >= 0.0, "gate penalty {p}");
    }
}
