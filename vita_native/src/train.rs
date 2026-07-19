//! vita-train — the trunk learns the Grammar from oracle-labeled words.
//!
//! Each vita-gen record becomes a turn in the certified template: the word
//! rides the δ arm, the oracle labels (flow signature · sink · kernel
//! verdict) ride the μ arm:
//!
//! ```text
//! ⊢⟨word⟩◇⊞⟨⟩●⟨flow4·sink4·kernel⟩⊣
//! ```
//!
//! Turns concatenate directly — the ⊣⊢ adjacency is legal by the bifurcation
//! law — into one ring-stream, chunked into training blocks. Loss is
//! next-token cross-entropy over the closed alphabet plus λ_gate times the
//! μ∘δ = id pressure on every split-fuse. The λ_sic fiducial-overlap term
//! (pinning the S¹¹ core to the d=12 SIC fiducial) is the next plank: it
//! joins when pointed at the fiducial data itself, never hand-derived.
//!
//! Usage: vita-train <dataset.jsonl> [steps] [seq_len] [batch] [out.safetensors] [arch]
//!   arch — "trunk" (default: 12 uniform layers) or "lattice" (the 26-step
//!   protocol of the discerning ob3ect, built to spec in lattice.rs)

use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use vita_native::{alphabet, lattice, model, template};

/// The two architectures behind one training loop: the uniform trunk and the
/// spec-built 26-step lattice. Same interface, disjoint parameter spaces.
enum Net {
    Trunk(model::VitaTrunk),
    Lattice(lattice::VitaLattice),
}

impl Net {
    fn forward_with_core(&self, ids: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        match self {
            Net::Trunk(m) => m.forward_with_core(ids),
            Net::Lattice(m) => m.forward_with_core(ids),
        }
    }
    fn gate_penalty(&self) -> candle_core::Result<Tensor> {
        match self {
            Net::Trunk(m) => m.gate_penalty(),
            Net::Lattice(m) => m.gate_penalty(),
        }
    }
}

const LAMBDA_GATE: f64 = 0.1;
// The λ_sic pin: core S¹¹ points driven to the Born constant 1/13 pairwise —
// the same overlap the Dual-Link vessel reads from the fiducial.
const LAMBDA_SIC: f64 = 1.0;

fn main() -> candle_core::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let data_path = args.get(1).cloned().unwrap_or_else(|| "vita_dataset.jsonl".into());
    let steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let seq_len: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(96);
    let batch: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);
    let out_path = args.get(5).cloned().unwrap_or_else(|| "vita_trunk.safetensors".into());
    let arch = args.get(6).cloned().unwrap_or_else(|| "trunk".into());

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    eprintln!("[vita-train] device: {:?}", device);

    // ── the ring-stream: every record rendered as a certified turn ──────
    let raw = std::fs::read_to_string(&data_path)
        .map_err(|e| candle_core::Error::Msg(format!("read {data_path}: {e}")))?;
    let mut turns = Vec::new();
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        let r: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| candle_core::Error::Msg(format!("parse record: {e}")))?;
        let s = |k: &str| r.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        turns.push(template::Turn {
            context: None,
            query: s("word"),
            think: None,
            answer: Some(format!("{}·{}·{}", s("flow4"), s("sink4"), s("kernel"))),
        });
    }
    let stream_text = template::render(&turns, template::RenderMode::Complete);
    let stream = alphabet::encode(&stream_text);
    eprintln!(
        "[vita-train] {} records → {} tokens (vocab {})",
        turns.len(),
        stream.len(),
        alphabet::vocab_size()
    );
    if stream.len() < seq_len + 1 {
        return Err(candle_core::Error::Msg(
            "stream shorter than one training block — generate more records".into(),
        ));
    }

    // ── trunk + optimizer ───────────────────────────────────────────────
    let cfg = model::Config::default();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let trunk = match arch.as_str() {
        "lattice" => {
            eprintln!("[vita-train] arch: the 26-step protocol lattice (built to the ob3ect's spec)");
            Net::Lattice(lattice::VitaLattice::new(cfg, seq_len + 1, vb)?)
        }
        _ => Net::Trunk(model::VitaTrunk::new(cfg, seq_len + 1, vb)?),
    };
    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW { lr: 3e-4, ..Default::default() },
    )?;

    // ── deterministic block sampler over the ring (ROTAT the stream) ────
    let n_starts = stream.len() - seq_len - 1;
    let mut seed = 0x243F6A8885A308D3u64;
    let mut next = move || {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        (seed.wrapping_mul(0x2545F4914F6CDD1D) % n_starts as u64) as usize
    };

    for step in 0..steps {
        let mut xs = Vec::with_capacity(batch * seq_len);
        let mut ys = Vec::with_capacity(batch * seq_len);
        for _ in 0..batch {
            let s = next();
            xs.extend_from_slice(&stream[s..s + seq_len]);
            ys.extend_from_slice(&stream[s + 1..s + seq_len + 1]);
        }
        let x = Tensor::from_vec(xs, (batch, seq_len), &device)?;
        let y = Tensor::from_vec(ys, (batch * seq_len,), &device)?;
        let (logits, core_z) = trunk.forward_with_core(&x)?;
        let ce = loss::cross_entropy(
            &logits.reshape((batch * seq_len, alphabet::vocab_size()))?,
            &y,
        )?;
        let gate = trunk.gate_penalty()?;
        let sic = model::sic_penalty(&core_z)?;
        let total = ((&ce + (gate.clone() * LAMBDA_GATE)?)? + (sic.clone() * LAMBDA_SIC)?)?;
        opt.backward_step(&total)?;
        if step % 10 == 0 || step + 1 == steps {
            eprintln!(
                "[vita-train] step {step:>5}  ce {:.4}  gate {:.4}  sic {:.5}",
                ce.to_scalar::<f32>()?,
                gate.to_scalar::<f32>()?,
                sic.to_scalar::<f32>()?
            );
        }
    }

    varmap.save(&out_path)?;
    eprintln!("[vita-train] saved {out_path}");
    Ok(())
}
