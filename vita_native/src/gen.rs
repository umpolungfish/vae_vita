//! vita-gen — the procedural training-data generator.
//!
//! Never hand-imscribed: every label on every record is emitted by the imasm
//! instruments (`ask --imasm eval|eval16|prove`), so the dataset is
//! tool-grounded end to end. The generator samples opcode words, puts each to
//! the three verdicts (grammar, kernel, flow), reads the flow signature per
//! dyad, and writes glyph-native JSONL records plus the canonical rotation.
//!
//! Usage: vita-gen <count> [max_len] [out.jsonl]
//!   count    — number of words to sample and label
//!   max_len  — maximum word length (default 16, min 4)
//!   out      — output path (default vita_dataset.jsonl)

use std::io::Write as _;
use std::process::Command;

#[path = "alphabet.rs"]
mod alphabet;

const ASK: &str = concat!(env!("HOME"), "/imsgct/MoDoT/ask");

/// The 12 classic node-opcodes, name and glyph, in imasm order.
const OPS: [(&str, char); 16] = [
    ("VINIT", '⊢'), ("TANCH", '⊣'), ("AFWD", '>'), ("AREV", '<'),
    ("CLINK", '='), ("IMSCRIB", '⊙'), ("FSPLIT", '◇'), ("FFUSE", '●'),
    ("EVALT", '+'), ("EVALF", '×'), ("ENGAGR", '⊞'), ("IFIX", '¬'),
    ("FSPLIT3", '∈'), ("FFUSE3", '∋'), ("TNEG", '~'), ("INEG", '≁'),
];

/// xorshift64* — deterministic, seedable, dependency-free.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Sample a word shaped like a program: opens with VINIT, seals with TANCH,
/// and every FSPLIT is eventually answered by an FFUSE (δ/μ balance), with
/// free WORK opcodes between. Shape, not verdict: the verdicts come from the
/// instruments, never from the sampler.
fn sample_word(rng: &mut Rng, max_len: usize) -> Vec<&'static str> {
    let body_len = 2 + rng.below(max_len.saturating_sub(4).max(1));
    let work = ["AFWD", "AREV", "CLINK", "IMSCRIB", "EVALT", "EVALF", "ENGAGR", "IFIX",
                "TNEG", "INEG"];
    let mut word = vec!["VINIT"];
    let mut open_forks = 0usize;   // classic ◇ awaiting ●
    let mut open_tris = 0usize;    // ∈ awaiting ∋ (3 arms reconverge on one fuse)
    for _ in 0..body_len {
        let roll = rng.below(12);
        if roll < 2 {
            word.push("FSPLIT");
            open_forks += 1;
        } else if roll < 3 {
            word.push("FSPLIT3");
            open_tris += 1;
        } else if roll < 5 && open_forks > 0 {
            word.push("FFUSE");
            open_forks -= 1;
        } else if roll < 6 && open_tris > 0 {
            word.push("FFUSE3");
            open_tris -= 1;
        } else {
            word.push(work[rng.below(work.len())]);
        }
    }
    while open_forks > 0 {
        word.push("FFUSE");
        open_forks -= 1;
    }
    while open_tris > 0 {
        word.push("FFUSE3");
        open_tris -= 1;
    }
    word.push("TANCH");
    word
}

fn glyphs(word: &[&str]) -> String {
    word.iter()
        .map(|n| OPS.iter().find(|(name, _)| name == n).map(|(_, g)| *g).unwrap_or('?'))
        .collect()
}

fn run_imasm(sub: &str, word: &[&str]) -> String {
    let mut args = vec!["--imasm".to_string(), sub.to_string()];
    args.extend(word.iter().map(|s| s.to_string()));
    match Command::new(ASK).args(&args).output() {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => format!("[oracle unreachable: {e}]"),
    }
}

/// Flow signature: per split-fuse dyad, ⊙ for id (lossless) and × for NOT-id
/// (value moved on the arms) — read off the eval output, in dyad order.
fn flow_signature(eval_out: &str) -> String {
    eval_out
        .lines()
        .filter(|l| l.contains("μ∘δ on values"))
        .map(|l| if l.contains("— id") { '⊙' } else { '×' })
        .collect()
}

/// Sink readout: the register name at the sink, e.g. "TANCH(B)" → "B".
fn sink_readout(eval_out: &str) -> String {
    eval_out
        .lines()
        .find(|l| l.contains("readout at sink"))
        .and_then(|l| l.rsplit_once('(').map(|(_, r)| r.trim_end_matches(')').trim().to_string()))
        .unwrap_or_default()
}

/// Kernel verdict as a Belnap token: T for green, B otherwise (held, a
/// frontier — the kernel's silence or refusal is never an F on our authority).
fn kernel_verdict(prove_out: &str) -> &'static str {
    if prove_out.contains("✓ green") {
        "T"
    } else {
        "B"
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let count: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let max_len: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16).max(4);
    let out_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "vita_dataset.jsonl".to_string());

    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut out = std::fs::File::create(&out_path).expect("create output");
    let mut written = 0usize;

    for i in 0..count {
        let word = sample_word(&mut rng, max_len);
        let g = glyphs(&word);
        let is_tri = word.iter().any(|n| matches!(*n, "FSPLIT3" | "FFUSE3" | "TNEG" | "INEG"));

        let ids = alphabet::encode(&g);
        let (canon_k, canon_ids) = alphabet::canonical_rotation(&ids);

        let record = if is_tri {
            // Tri-words go to the trilattice machine itself — the classic
            // eval verbs silently drop tokens they do not know, and a label
            // for a filtered word is a lie. The 16_3 machine gives the sink
            // register and the tri-ancestral verdict (T/N/B/F, the full
            // verdict space).
            let out16 = {
                let mut args = vec!["--imasm".to_string(), "16_3".into(), "check".into(), g.clone()];
                match Command::new(ASK).args(&args.drain(..).collect::<Vec<_>>()).output() {
                    Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
                    Err(e) => format!("[oracle unreachable: {e}]"),
                }
            };
            let sink = out16.lines()
                .find(|l| l.contains("Final register:"))
                .and_then(|l| l.rsplit(':').next().map(|s| s.trim().to_string()))
                .unwrap_or_default();
            let verdict = out16.lines()
                .find(|l| l.contains("Tri-ancestral verdict:"))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|s| s.trim().chars().next())
                .map(|c| c.to_string())
                .unwrap_or_else(|| "N".into());
            serde_json::json!({
                "word": g,
                "flow4": "", "sink4": "",
                "flow16": "", "sink16": sink,
                "kernel": verdict,
                "canonical_k": canon_k,
                "canonical_word": alphabet::decode(&canon_ids),
                "ids": ids,
            })
        } else {
            let eval4 = run_imasm("eval", &word);
            let eval16 = run_imasm("eval16", &word);
            let prove = run_imasm("prove", &word);
            serde_json::json!({
                "word": g,
                "flow4": flow_signature(&eval4),
                "sink4": sink_readout(&eval4),
                "flow16": flow_signature(&eval16),
                "sink16": sink_readout(&eval16),
                "kernel": kernel_verdict(&prove),
                "canonical_k": canon_k,
                "canonical_word": alphabet::decode(&canon_ids),
                "ids": ids,
            })
        };
        writeln!(out, "{record}").expect("write record");
        written += 1;
        if (i + 1) % 10 == 0 {
            eprintln!("[vita-gen] {}/{count}", i + 1);
        }
    }
    eprintln!(
        "[vita-gen] wrote {written} records to {out_path} (vocab {} tokens)",
        alphabet::vocab_size()
    );
}
