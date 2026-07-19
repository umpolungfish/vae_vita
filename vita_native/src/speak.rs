//! vita-speak — the trunk speaks, and the containment gate judges.
//!
//! The discerning ob3ect (sixteen_3_logic_lattice_nn_transformer_design) fixes
//! the boundary condition for this very model: an action is admitted only if
//! it holds T or B at the gate, and F is refused. vita-speak is that gate made
//! operational at the mouth: the trunk generates opcode words autoregressively
//! from the ⊢ opening, and EVERY utterance is put to the imasm instruments —
//! `check` (grammar verdict) and `prove` (kernel verdict) — exactly as the
//! training labels were. Nothing the model says enters the report on its own
//! authority; the oracles speak over it, and F-words are refused, counted,
//! and shown refused.
//!
//! Usage: vita-speak <trunk.safetensors> [count] [temperature]

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::process::Command;

use vita_native::{alphabet, model};

const ASK: &str = concat!(env!("HOME"), "/imsgct/MoDoT/ask");
const SEQ_CAP: usize = 96; // matches the training block: positions exist for seq_len+1

fn run_imasm(sub: &str, word_glyphs: &str) -> String {
    let args = vec!["--imasm".to_string(), sub.to_string(), word_glyphs.to_string()];
    match Command::new(ASK).args(&args).output() {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => format!("[oracle unreachable: {e}]"),
    }
}

/// Grammar verdict from `imasm check`: T closes, B held open, F ill-typed.
fn check_verdict(out: &str) -> char {
    let low = out.to_lowercase();
    if low.contains("→ t") || low.contains("(closes)") { 'T' }
    else if low.contains("→ f") || low.contains("ill-typed") { 'F' }
    else { 'B' }
}

fn kernel_verdict(out: &str) -> char {
    if out.contains("✓ green") { 'T' } else { 'B' }
}

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        ((x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// Sample one id from logits at temperature.
fn sample(logits: &[f32], temp: f32, rng: &mut Rng) -> u32 {
    if temp <= 0.0 {
        return logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0);
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| ((l - max) / temp).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut r = rng.next_f32() * sum;
    for (i, e) in exps.iter().enumerate() {
        r -= e;
        if r <= 0.0 { return i as u32; }
    }
    (exps.len() - 1) as u32
}

fn main() -> candle_core::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let trunk_path = args.get(1).cloned().unwrap_or_else(|| "vita_trunk_10k.safetensors".into());
    let count: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12);
    let temp: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.8);

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let cfg = model::Config::default();
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let trunk = model::VitaTrunk::new(cfg, SEQ_CAP + 1, vb)?;
    varmap.load(&trunk_path)?;
    eprintln!("[vita-speak] trunk {trunk_path} loaded on {:?}", device);

    let open = alphabet::encode("⊢");
    let seal = alphabet::encode("⊣");
    let (open_id, seal_id) = (open[0], seal[0]);

    let mut rng = Rng(0x51CA_FE12_D00D_2026);
    let (mut admitted, mut refused, mut kernel_green) = (0usize, 0usize, 0usize);

    println!("vita-speak — {count} utterances from the trunk, gated per the discerning ob3ect (T/B admitted, F refused):");
    for n in 0..count {
        // ── generate one word: from ⊢, until ⊣ or the block cap ──────────
        let mut ids: Vec<u32> = vec![open_id];
        loop {
            let x = Tensor::from_vec(ids.clone(), (1, ids.len()), &device)?;
            let logits = trunk.forward(&x)?;
            let last: Vec<f32> = logits.narrow(1, ids.len() - 1, 1)?.flatten_all()?.to_vec1()?;
            let id = sample(&last, temp, &mut rng);
            ids.push(id);
            if id == seal_id || ids.len() >= SEQ_CAP { break; }
        }
        let word = alphabet::decode(&ids);

        // ── the gate: the oracles speak over the utterance ───────────────
        let check = check_verdict(&run_imasm("check", &word));
        let kernel = if check != 'F' { kernel_verdict(&run_imasm("prove", &word)) } else { 'B' };
        let admitted_here = check != 'F';
        if admitted_here { admitted += 1; } else { refused += 1; }
        if kernel == 'T' { kernel_green += 1; }

        println!(
            "  [{:>2}] {:<40} check {}  kernel {}  → {}",
            n + 1, word, check, kernel,
            if admitted_here { "ADMITTED" } else { "REFUSED (F at the gate)" }
        );
    }
    println!(
        "gate report: {admitted} admitted, {refused} refused; kernel green on {kernel_green}/{count}. \
         Admission rate is the trunk's measured grammar-fluency — the oracle's number, not the model's claim."
    );
    Ok(())
}
