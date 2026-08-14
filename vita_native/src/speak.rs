//! vita-speak — the trunk speaks a certified turn, and the gate judges.
//!
//! The discerning ob3ect (sixteen_3_logic_lattice_nn_transformer_design) fixes
//! the boundary condition for this very model: an action is admitted only if
//! it holds T or B at the gate, and F is refused. vita-speak is that gate made
//! operational at the mouth — and the mouth speaks in the spec's own skeleton
//! (template.rs: a dialogue turn IS an IMASM word):
//!
//! ```text
//! ⊢ ⟨query⟩ ◇ ⊞ ⟨⟩ ● ⟨flow·sink·kernel⟩ ⊣
//! ```
//!
//! The trunk generates the query arm under the kernel's valence discipline;
//! the empty-think skeleton is then rendered, and the trunk completes the
//! answer arm — its OWN predicted readout, spoken in the register glyph face
//! before the oracles are consulted. Every query word is put to the imasm
//! instruments — `check` (grammar) and `prove` (kernel) — exactly as the
//! training labels were, and the model's self-predicted kernel verdict is
//! compared against the oracle's. Nothing the model says enters the report
//! on its own authority; F-words are refused, counted, and shown refused.
//!
//! Usage: vita-speak <trunk.safetensors> [count] [temperature] [arch] [word_cap] [harvest.jsonl]
//!
//! With a harvest path, every utterance is appended as a full corpus-schema
//! record (oracle-labeled, `source: mouth`) — the autopoietic winding: the
//! trunk's own words, its own failures included, become the next diet.

use candle_core::{DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::process::Command;

use vita_native::{alphabet, device as ig_device, lattice, model};

enum Net {
    Trunk(model::VitaTrunk),
    Lattice(lattice::VitaLattice),
}
impl Net {
    fn forward(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Net::Trunk(m) => m.forward(ids),
            Net::Lattice(m) => m.forward(ids),
        }
    }
}

const ASK: &str = concat!(env!("HOME"), "/imsgct/MoDoT/ask");

const OPS: [(&str, char); 12] = [
    ("VINIT", '⊢'), ("TANCH", '⊣'), ("AFWD", '≻'), ("AREV", '≺'),
    ("CLINK", '='), ("IMSCRIB", '⊙'), ("FSPLIT", '◇'), ("FFUSE", '●'),
    ("EVALT", '+'), ("EVALF", '×'), ("ENGAGR", '⊞'), ("IFIX", '¬'),
];

/// The oracles' name-list calling convention (as vita-gen and vita-corpus
/// use for eval/eval16/prove labeling).
fn run_imasm_names(sub: &str, names: &[String]) -> String {
    let mut args = vec!["--imasm".to_string(), sub.to_string()];
    args.extend(names.iter().cloned());
    match Command::new(ASK).args(&args).output() {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => format!("[oracle unreachable: {e}]"),
    }
}

fn flow_signature(eval_out: &str) -> String {
    eval_out.lines()
        .filter(|l| l.contains("μ∘δ on values"))
        .map(|l| if l.contains("— id") { '⊙' } else { '×' })
        .collect()
}

fn sink_readout(eval_out: &str) -> String {
    eval_out.lines()
        .find(|l| l.contains("readout at sink"))
        .and_then(|l| l.rsplit_once('(').map(|(_, r)| r.trim_end_matches(')').trim().to_string()))
        .unwrap_or_default()
}
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
    let arch = args.get(4).cloned().unwrap_or_else(|| "trunk".into());
    // Query-arm cap, in tokens. Default 34 — the longest real organism in the
    // corpus; the mouth stays inside the length regime the trunk learned.
    let word_cap: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(34);
    // Optional harvest path: every utterance appended as a full corpus-schema
    // record, oracle-labeled — the mouth's own words become diet (the
    // autopoietic winding: speak → harvest → retrain → speak).
    let harvest = args.get(6).cloned();

    let device = ig_device::pick();
    let cfg = model::Config::default();
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let trunk = match arch.as_str() {
        "lattice" => Net::Lattice(lattice::VitaLattice::new(cfg, SEQ_CAP + 1, vb)?),
        _ => Net::Trunk(model::VitaTrunk::new(cfg, SEQ_CAP + 1, vb)?),
    };
    varmap.load(&trunk_path)?;
    eprintln!("[vita-speak] trunk {trunk_path} loaded on {:?}", device);

    let id_of = |g: &str| alphabet::encode(g)[0];
    let (open_id, seal_id) = (id_of("⊢"), id_of("⊣"));
    let (split_id, fuse_id) = (id_of("◇"), id_of("●"));
    let (lang_id, rang_id, dot_id) = (id_of("⟨"), id_of("⟩"), id_of("·"));
    // The query arm speaks only opcodes — organism words are opcode words,
    // and they carry their own ⊢…⊣ inside the ⟨⟩ (the corpus renders whole
    // protocol words as queries). Opcodes-only is also what disambiguates
    // the δ arm from the μ arm at ⊢⟨.
    let query_legal: Vec<u32> = "⊢⊣><=⊙◇●∈∋+×⊞~≁¬".chars()
        .map(|c| id_of(&c.to_string())).collect();
    // The answer arm speaks the readout syntax the labels taught: flow marks
    // ⊙/×, register glyph faces, the · separator, and the closing ⟩.
    let mut answer_legal: Vec<u32> = alphabet::reg_faces().iter().map(|(_, g)| id_of(g)).collect();
    answer_legal.extend([id_of("⊙"), id_of("×"), dot_id, rang_id]);
    // Forced skeleton between the arms: empty think, then the μ arm opens.
    let mid_ids: Vec<u32> = alphabet::encode("◇⊞⟨⟩●⟨");
    const ANS_CAP: usize = 16;

    let mut harvest_out = harvest.as_ref().map(|p| {
        std::fs::OpenOptions::new().create(true).append(true).open(p)
            .expect("open harvest file")
    });

    let mut rng = Rng(0x51CA_FE12_D00D_2026);
    let (mut admitted, mut refused, mut kernel_green, mut self_agree) = (0usize, 0usize, 0usize, 0usize);

    println!("vita-speak — {count} certified turns from the trunk, gated per the discerning ob3ect (T/B admitted, F refused):");
    for n in 0..count {
        // ── the δ arm: ⊢⟨ primes the turn; the trunk speaks the query word
        // under the kernel's valence discipline (fork stack as logit mask):
        // ● only over an open ◇; ⟩ closes the arm only at depth 0; when the
        // cap looms, open forks are FORCED closed then the arm is sealed, so
        // every word leaves the mouth δ/μ-balanced within its arm.
        let mut ids: Vec<u32> = vec![open_id, lang_id];
        let mut depth = 0usize;
        let mut sealed = false; // the word spoke its own ⊣
        loop {
            let x = Tensor::from_vec(ids.clone(), (1, ids.len()), &device)?;
            let logits = trunk.forward(&x)?;
            let mut last: Vec<f32> = logits.narrow(1, ids.len() - 1, 1)?.flatten_all()?.to_vec1()?;
            let qlen = ids.len() - 2;
            if sealed || qlen + depth + 2 >= word_cap {
                // The word is sealed, or out of room: close forks (●), seal
                // if unsealed (⊣), then close the arm (⟩).
                let force = if depth > 0 { fuse_id } else if !sealed { seal_id } else { rang_id };
                for (i, l) in last.iter_mut().enumerate() {
                    if i as u32 != force { *l = f32::NEG_INFINITY; }
                }
            } else {
                for (i, l) in last.iter_mut().enumerate() {
                    let id = i as u32;
                    if !query_legal.contains(&id) { *l = f32::NEG_INFINITY; }
                }
                if qlen == 0 {
                    // Word-initial is ⊢ or ⊙ in the entire real population;
                    // any other opening is off-manifold before it begins.
                    let word_initial = [open_id, id_of("⊙")];
                    for (i, l) in last.iter_mut().enumerate() {
                        if !word_initial.contains(&(i as u32)) { *l = f32::NEG_INFINITY; }
                    }
                } else {
                    last[open_id as usize] = f32::NEG_INFINITY;
                }
                if depth == 0 { last[fuse_id as usize] = f32::NEG_INFINITY; }
                if depth > 0 { last[seal_id as usize] = f32::NEG_INFINITY; }
            }
            let id = sample(&last, temp, &mut rng);
            if id == split_id { depth += 1; }
            if id == fuse_id { depth = depth.saturating_sub(1); }
            if id == seal_id { sealed = true; }
            ids.push(id);
            if id == rang_id { break; }
        }
        let word = alphabet::decode(&ids[2..ids.len() - 1]);

        // ── the skeleton: empty ⊞ block, μ arm opens ─────────────────────
        ids.extend_from_slice(&mid_ids);

        // ── the answer arm: the trunk pronounces its own readout, register
        // glyphs only, until it closes the arm (or the cap closes it).
        loop {
            let x = Tensor::from_vec(ids.clone(), (1, ids.len()), &device)?;
            let logits = trunk.forward(&x)?;
            let mut last: Vec<f32> = logits.narrow(1, ids.len() - 1, 1)?.flatten_all()?.to_vec1()?;
            let alen = ids.len() - (2 + word.chars().count() + 1 + mid_ids.len());
            for (i, l) in last.iter_mut().enumerate() {
                if !answer_legal.contains(&(i as u32)) { *l = f32::NEG_INFINITY; }
            }
            let id = if alen + 1 >= ANS_CAP { rang_id } else { sample(&last, temp, &mut rng) };
            ids.push(id);
            if id == rang_id { break; }
        }
        ids.push(seal_id);
        let turn = alphabet::decode(&ids);
        // The model's self-predicted kernel verdict: the last register token
        // in the answer arm, kept in its glyph face (no Latin anywhere).
        let self_kernel = ids[ids.len() - 3..ids.len() - 1]
            .first()
            .map(|id| alphabet::decode(&[*id]))
            .filter(|g| alphabet::reg_faces().iter().any(|(_, f)| f == g))
            .unwrap_or_else(|| "?".into());

        // ── the gate: the oracles speak over the query word ──────────────
        let check = check_verdict(&run_imasm("check", &word));
        let kernel = if check != 'F' { kernel_verdict(&run_imasm("prove", &word)) } else { 'B' };
        let admitted_here = check != 'F';
        if admitted_here { admitted += 1; } else { refused += 1; }
        if kernel == 'T' { kernel_green += 1; }

        // ── harvest: the utterance becomes a corpus-schema record ────────
        if let Some(out) = harvest_out.as_mut() {
            use std::io::Write as _;
            let names: Vec<String> = word.chars()
                .filter_map(|c| OPS.iter().find(|(_, g)| *g == c).map(|(n, _)| n.to_string()))
                .collect();
            let eval4 = run_imasm_names("eval", &names);
            let eval16 = run_imasm_names("eval16", &names);
            let ids = alphabet::encode(&word);
            let (canon_k, canon_ids) = alphabet::canonical_rotation(&ids);
            let record = serde_json::json!({
                "source": "mouth",
                "word": word,
                "flow4": flow_signature(&eval4),
                "sink4": sink_readout(&eval4),
                "flow16": flow_signature(&eval16),
                "sink16": sink_readout(&eval16),
                "kernel": kernel.to_string(),
                "self": self_kernel,
                "canonical_k": canon_k,
                "canonical_word": alphabet::decode(&canon_ids),
                "ids": ids,
            });
            writeln!(out, "{record}").expect("write harvest record");
        }
        let agree = self_kernel == alphabet::reg_face(&kernel.to_string());
        if agree { self_agree += 1; }

        println!(
            "  [{:>2}] {:<52} check {}  kernel {}  self {}{}  → {}",
            n + 1, turn, check, kernel, self_kernel,
            if agree { " ✓" } else { "" },
            if admitted_here { "ADMITTED" } else { "REFUSED (F at the gate)" }
        );
    }
    println!(
        "gate report: {admitted} admitted, {refused} refused; kernel green on {kernel_green}/{count}; \
         self-readout agrees with the oracle on {self_agree}/{count}. \
         Admission rate is the trunk's measured grammar-fluency — the oracle's number, not the model's claim."
    );
    Ok(())
}
