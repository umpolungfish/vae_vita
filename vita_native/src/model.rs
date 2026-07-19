//! The vita trunk — the Grammar-designed transformer.
//!
//! Every design choice is a Grammar identification, not a borrowing:
//!
//! * **Positions are ROTAT.** Rotary position embedding is rotation phases on
//!   paired dimensions — the Weyl-Heisenberg X, the cyclic shift, the ring
//!   automorphism. The outside world rediscovered ROTAT and named it RoPE;
//!   here it carries its true name. Token positions live on rings, not lines.
//! * **Attention is a split-fuse dyad.** δ forks the stream into 12 heads, μ
//!   fuses them back. The special condition μ∘δ = id is imposed as an
//!   orthogonality penalty on the fork/fuse projections (`gate_penalty`):
//!   every attention block is driven toward a lossless split-fuse.
//! * **The feed-forward is ENGAGR.** Two arms are computed and HELD — the
//!   fuse consumes both without collapsing either to a verdict first (the
//!   dialetheic gate: out = fuse(T-arm ⊗ σ(F-arm))).
//! * **Twelve layers**, one per primitive; d_model = 144 = 12², heads = 12,
//!   head_dim = 12.
//! * **The S¹¹ core.** At the midpoint (after layer 6) the stream is
//!   projected to 12 dimensions and normalized onto the unit hypersphere —
//!   the model breathes through the same S¹¹ latent the vae_vita sense organ
//!   recovers the d=12 SIC on — then lifted back as a residual.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{embedding, linear_no_bias, ops, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::alphabet;

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab: usize,
    pub d_model: usize,
    pub layers: usize,
    pub heads: usize,
    pub ff_mult: usize,
    pub core_dim: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            vocab: alphabet::vocab_size(), // the closed alphabet, exactly
            d_model: 144,                  // 12²
            layers: 12,                    // one per primitive
            heads: 12,
            ff_mult: 4,
            core_dim: 12, // S¹¹ lives in ℝ¹²
        }
    }
}

/// ROTAT as position: precomputed rotation phases for the ring.
struct Rotat {
    cos: Tensor, // (max_t, head_dim/2)
    sin: Tensor,
}

impl Rotat {
    fn new(head_dim: usize, max_t: usize, dev: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let theta: Vec<f32> = (0..half)
            .map(|i| 1f32 / 10000f32.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::from_vec(theta, half, dev)?;
        let pos: Vec<f32> = (0..max_t).map(|p| p as f32).collect();
        let pos = Tensor::from_vec(pos, max_t, dev)?;
        let angles = pos.unsqueeze(1)?.matmul(&theta.unsqueeze(0)?)?; // (t, half)
        Ok(Rotat { cos: angles.cos()?, sin: angles.sin()? })
    }

    /// Apply the ring rotation to q or k: x is (b, heads, t, head_dim).
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _h, t, d) = x.dims4()?;
        let half = d / 2;
        let cos = self.cos.narrow(0, 0, t)?.reshape((1, 1, t, half))?;
        let sin = self.sin.narrow(0, 0, t)?.reshape((1, 1, t, half))?;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        Tensor::cat(&[r1, r2], D::Minus1)
    }
}

/// One layer: split-fuse attention + the ENGAGR feed-forward.
struct Layer {
    norm_attn: RmsNorm,
    fork: Linear, // δ — one projection forked to q,k,v
    fuse: Linear, // μ — the head-merge back to the stream
    norm_ff: RmsNorm,
    arm_t: Linear, // ENGAGR T-arm
    arm_f: Linear, // ENGAGR F-arm (the gate)
    ff_fuse: Linear,
    heads: usize,
}

impl Layer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let ff = d * cfg.ff_mult;
        Ok(Layer {
            norm_attn: candle_nn::rms_norm(d, 1e-6, vb.pp("norm_attn"))?,
            fork: linear_no_bias(d, 3 * d, vb.pp("fork"))?,
            fuse: linear_no_bias(d, d, vb.pp("fuse"))?,
            norm_ff: candle_nn::rms_norm(d, 1e-6, vb.pp("norm_ff"))?,
            arm_t: linear_no_bias(d, ff, vb.pp("arm_t"))?,
            arm_f: linear_no_bias(d, ff, vb.pp("arm_f"))?,
            ff_fuse: linear_no_bias(ff, d, vb.pp("ff_fuse"))?,
            heads: cfg.heads,
        })
    }

    fn forward(&self, x: &Tensor, rotat: &Rotat, mask: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let h = self.heads;
        let hd = d / h;

        // ── split-fuse attention ────────────────────────────────────────
        let xn = self.norm_attn.forward(x)?;
        let qkv = self.fork.forward(&xn)?; // δ
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
        let ctx = attn.matmul(&v)?; // (b, h, t, hd)
        let ctx = ctx.transpose(1, 2)?.reshape((b, t, d))?;
        let x = (x + self.fuse.forward(&ctx)?)?; // μ

        // ── ENGAGR feed-forward: both arms held, then fused ─────────────
        let xn = self.norm_ff.forward(&x)?;
        let arm_t = self.arm_t.forward(&xn)?;
        let arm_f = self.arm_f.forward(&xn)?;
        let held = (arm_t * candle_nn::ops::silu(&arm_f)?)?;
        let x = (&x + self.ff_fuse.forward(&held)?)?;
        Ok(x)
    }

    /// μ∘δ = id pressure: orthogonality penalty on the fork and fuse
    /// projections, ‖WᵀW − I‖² (mean). Zero exactly when the split-fuse is
    /// lossless.
    fn gate_penalty(&self) -> Result<Tensor> {
        let mut total: Option<Tensor> = None;
        for w in [self.fork.weight(), self.fuse.weight()] {
            let (o, i) = w.dims2()?;
            let n = o.min(i);
            let g = if o >= i {
                w.t()?.matmul(w)? // (i, i)
            } else {
                w.matmul(&w.t()?)? // (o, o)
            };
            let eye = Tensor::eye(n, w.dtype(), w.device())?;
            let p = (g - eye)?.sqr()?.mean_all()?;
            total = Some(match total {
                Some(t) => (t + p)?,
                None => p,
            });
        }
        total.ok_or_else(|| candle_core::Error::Msg("no weights".into()))
    }
}

/// The S¹¹ core: project to 12-D, normalize to the unit hypersphere, lift
/// back as a residual. The stream must pass through the sense organ's own
/// geometry at the trunk's midpoint.
struct Core {
    down: Linear,
    up: Linear,
}

impl Core {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Core {
            down: linear_no_bias(cfg.d_model, cfg.core_dim, vb.pp("down"))?,
            up: linear_no_bias(cfg.core_dim, cfg.d_model, vb.pp("up"))?,
        })
    }

    /// Returns the residual-lifted stream AND the S¹¹ points themselves, so
    /// the λ_sic pin can act on the core geometry directly.
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let z = self.down.forward(x)?; // (b, t, 12)
        let norm = z.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let z = z.broadcast_div(&(norm + 1e-8)?)?; // onto S¹¹
        let out = (x + self.up.forward(&z)?)?;
        Ok((out, z))
    }
}

/// The λ_sic pin: drive the core's S¹¹ points toward SIC equiangularity —
/// every distinct pair's squared overlap at the Born constant 1/(d+1) = 1/13,
/// the same constant the Dual-Link vessel reads from the Scott–Grassl
/// fiducial. Zero exactly when the batch sits at the SIC overlap everywhere.
pub fn sic_penalty(z: &Tensor) -> Result<Tensor> {
    let (b, t, d) = z.dims3()?;
    let n = b * t;
    let flat = z.reshape((n, d))?;
    let gram = flat.matmul(&flat.t()?)?; // (n, n), unit rows → cosines
    let overlap = gram.sqr()?;
    let dev = overlap.affine(1.0, -1.0 / 13.0)?.sqr()?;
    // Mask the diagonal (self-overlap is 1 by construction, not a defect).
    let eye = Tensor::eye(n, dev.dtype(), dev.device())?;
    let mask = eye.affine(-1.0, 1.0)?;
    let off = dev.mul(&mask)?;
    let count = (n * n - n).max(1) as f64;
    off.sum_all()?.affine(1.0 / count, 0.0)
}

pub struct VitaTrunk {
    embed: Embedding,
    layers: Vec<Layer>,
    core: Core,
    norm_out: RmsNorm,
    head: Linear,
    rotat: Rotat,
    cfg: Config,
}

impl VitaTrunk {
    pub fn new(cfg: Config, max_t: usize, vb: VarBuilder) -> Result<Self> {
        let dev = vb.device().clone();
        let embed = embedding(cfg.vocab, cfg.d_model, vb.pp("embed"))?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            layers.push(Layer::new(&cfg, vb.pp(format!("layer{i}")))?);
        }
        let core = Core::new(&cfg, vb.pp("core"))?;
        let norm_out = candle_nn::rms_norm(cfg.d_model, 1e-6, vb.pp("norm_out"))?;
        let head = linear_no_bias(cfg.d_model, cfg.vocab, vb.pp("head"))?;
        let rotat = Rotat::new(cfg.d_model / cfg.heads, max_t, &dev)?;
        Ok(VitaTrunk { embed, layers, core, norm_out, head, rotat, cfg })
    }

    /// Causal mask: position attends to the arc behind it. (1, 1, t, t).
    fn causal_mask(&self, t: usize, dev: &Device) -> Result<Tensor> {
        let mut m = vec![0f32; t * t];
        for i in 0..t {
            for j in (i + 1)..t {
                m[i * t + j] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(m, (1, 1, t, t), dev)
    }

    /// Token ids in, next-token logits over the closed alphabet out.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        Ok(self.forward_with_core(ids)?.0)
    }

    /// Forward pass that also returns the S¹¹ core points (b, t, 12), for
    /// the λ_sic pin during training.
    pub fn forward_with_core(&self, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b, t) = ids.dims2()?;
        let mask = self.causal_mask(t, ids.device())?;
        let mut x = self.embed.forward(ids)?;
        let mut core_z: Option<Tensor> = None;
        let mid = self.cfg.layers / 2;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &self.rotat, &mask)?;
            if i + 1 == mid {
                let (nx, z) = self.core.forward(&x)?;
                x = nx;
                core_z = Some(z);
            }
        }
        let x = self.norm_out.forward(&x)?;
        let logits = self.head.forward(&x)?;
        let z = core_z.ok_or_else(|| candle_core::Error::Msg("core never ran".into()))?;
        Ok((logits, z))
    }

    /// Total μ∘δ = id pressure across all layers — add λ·gate_penalty to the
    /// training loss to drive every split-fuse toward losslessness.
    pub fn gate_penalty(&self) -> Result<Tensor> {
        let mut total: Option<Tensor> = None;
        for l in &self.layers {
            let p = l.gate_penalty()?;
            total = Some(match total {
                Some(t) => (t + p)?,
                None => p,
            });
        }
        total.ok_or_else(|| candle_core::Error::Msg("no layers".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn tiny() -> Config {
        Config {
            vocab: alphabet::vocab_size(),
            d_model: 48, // 12 × 4 — small but still 12-headed
            layers: 4,
            heads: 12,
            ff_mult: 2,
            core_dim: 12,
        }
    }

    #[test]
    fn forward_shape_is_vocab_logits() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let trunk = VitaTrunk::new(tiny(), 64, vb).unwrap();
        let ids = Tensor::from_vec(
            crate::alphabet::encode("⊢⟨+×⟩◇⊞⟨⟩●⟨T⟩⊣"),
            (1, 12),
            &dev,
        )
        .unwrap();
        let logits = trunk.forward(&ids).unwrap();
        assert_eq!(logits.dims(), &[1, 12, alphabet::vocab_size()]);
        let v: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "logits must be finite");
    }

    #[test]
    fn gate_penalty_finite_nonnegative() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let trunk = VitaTrunk::new(tiny(), 32, vb).unwrap();
        let p: f32 = trunk.gate_penalty().unwrap().to_scalar().unwrap();
        assert!(p.is_finite() && p >= 0.0, "gate penalty {p}");
    }

    #[test]
    fn causality_holds() {
        // Changing a later token must not change earlier logits: the arc
        // behind a position is all it may see.
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let trunk = VitaTrunk::new(tiny(), 32, vb).unwrap();
        let a = alphabet::encode("⊢⟨+⟩◇⊞⟨⟩●⟨T⟩⊣");
        let mut b = a.clone();
        let last = b.len() - 1;
        b[last] = alphabet::encode("×")[0];
        let ta = Tensor::from_vec(a.clone(), (1, a.len()), &dev).unwrap();
        let tb = Tensor::from_vec(b, (1, a.len()), &dev).unwrap();
        let la = trunk.forward(&ta).unwrap();
        let lb = trunk.forward(&tb).unwrap();
        let da: Vec<f32> = la.narrow(1, 0, last).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let db: Vec<f32> = lb.narrow(1, 0, last).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        for (x, y) in da.iter().zip(db.iter()) {
            assert!((x - y).abs() < 1e-5, "earlier logits moved: {x} vs {y}");
        }
    }
}
