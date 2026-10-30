//! vita-corpus — render the REAL corpus: every discerned ob3ect's protocol
//! word, oracle-labeled, as training records.
//!
//! The synthetic bootstrap (vita-gen) taught the trunk the texture of
//! grammar-speech from RNG babble; this feeds the words the Grammar actually
//! produced when it discerned something. Each ob3ect's lean_scaffold opens
//! with its IGProtocol sequence; that word goes to the SAME three instruments
//! that labeled the synthetic set (eval, eval16, prove), so the record schema
//! is identical and vita-train consumes it unchanged. Never hand-imscribed:
//! the scaffold gives only the word; every label is the oracles'.
//!
//! Usage: vita-corpus [ob3ect_dir] [out.jsonl]

use std::io::Write as _;
use std::process::Command;

use vita_native::alphabet;

const ASK: &str = concat!(env!("HOME"), "/imsgct/MoDoT/ask");

const OPS: [(&str, char); 12] = [
    ("VINIT", '⊢'), ("TANCH", '⊣'), ("AFWD", '≻'), ("AREV", '≺'),
    ("CLINK", '='), ("IMSCRIB", '⊙'), ("FSPLIT", '◇'), ("FFUSE", '●'),
    ("EVALT", '+'), ("EVALF", '×'), ("ENGAGR", '⊞'), ("IFIX", '¬'),
];

fn run_imasm(sub: &str, word: &[String]) -> String {
    let mut args = vec!["--imasm".to_string(), sub.to_string()];
    args.extend(word.iter().cloned());
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

fn kernel_verdict(prove_out: &str) -> &'static str {
    if prove_out.contains("✓ green") { "T" } else { "B" }
}

/// The protocol word from an ob3ect JSON's lean_scaffold header line:
/// `-- IGProtocol scaffold: VINIT → IMSCRIB → …`. Returns token names.
fn protocol_of(json: &serde_json::Value) -> Option<Vec<String>> {
    let scaffold = json.get("lean_scaffold")?.as_str()?;
    let first = scaffold.lines().next()?;
    let seq = first.strip_prefix("-- IGProtocol scaffold: ")?;
    let toks: Vec<String> = seq.split('→').map(|t| t.trim().to_string()).collect();
    if !toks.is_empty() && toks.iter().all(|t| OPS.iter().any(|(n, _)| n == t)) {
        Some(toks)
    } else {
        None
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).cloned()
        .unwrap_or_else(|| format!("{}/imsgct/ob3ect/digital", std::env::var("HOME").unwrap()));
    let out_path = args.get(2).cloned().unwrap_or_else(|| "vita_corpus.jsonl".to_string());

    let mut out = std::fs::File::create(&out_path).expect("create output");
    let (mut written, mut skipped) = (0usize, 0usize);

    let mut entries: Vec<_> = std::fs::read_dir(&dir).expect("read ob3ect dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();

    for sub in entries {
        let Some(json_path) = std::fs::read_dir(&sub).ok().and_then(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .find(|p| p.file_name().map(|n| n.to_string_lossy().ends_with("_ob3ect.json")).unwrap_or(false))
        }) else { skipped += 1; continue };
        let Ok(text) = std::fs::read_to_string(&json_path) else { skipped += 1; continue };
        let Ok(j) = serde_json::from_str::<serde_json::Value>(&text) else { skipped += 1; continue };
        let Some(word) = protocol_of(&j) else { skipped += 1; continue };

        let g: String = word.iter()
            .map(|n| OPS.iter().find(|(name, _)| name == n).map(|(_, c)| *c).unwrap())
            .collect();

        let eval4 = run_imasm("eval", &word);
        let eval16 = run_imasm("eval16", &word);
        let prove = run_imasm("prove", &word);

        let ids = alphabet::encode(&g);
        let (canon_k, canon_ids) = alphabet::canonical_rotation(&ids);
        let name = j.get("name").and_then(|v| v.as_str()).unwrap_or("");

        let record = serde_json::json!({
            "name": name,
            "word": g,
            "flow4": flow_signature(&eval4),
            "sink4": sink_readout(&eval4),
            "flow16": flow_signature(&eval16),
            "sink16": sink_readout(&eval16),
            "kernel": kernel_verdict(&prove),
            "canonical_k": canon_k,
            "canonical_word": alphabet::decode(&canon_ids),
            "ids": ids,
        });
        writeln!(out, "{record}").expect("write record");
        written += 1;
        if written % 50 == 0 {
            eprintln!("[vita-corpus] {written} rendered…");
        }
    }
    eprintln!("[vita-corpus] wrote {written} records to {out_path} ({skipped} skipped) — the real organisms, oracle-labeled.");
}
