//! vita-probe — the seed atlas: host-side survey of the trunk's language.
//!
//! The forward pass, sampler, and gate are the SAME scalar f32 code that runs
//! on mOMonadOS (src/vita.rs there), reading the SAME baked vita_weights.bin,
//! so seed↔word parity with the metal is bit-exact: any specimen found here
//! can be re-spoken on the OS by its (seed, temp) coordinates.
//!
//! The probe: enumerate seeds in order and discard already-seen words — the
//! dedup IS the skip past the numerically optimized core. A temperature
//! schedule walks outward in shells. Around each novel specimen the gate
//! (imasm_core, microseconds, no forward pass) spiders single-glyph edits,
//! charting how much of the neighborhood closes, holds paradox, or breaks.
//!
//! Usage:
//!   vita-probe --one 666 0.9                      # speak one turn (parity check)
//!   vita-probe --seeds 1000 --temps 0.8,1.2,1.6 --spider --out atlas.jsonl

use std::collections::BTreeMap;
use std::io::Write as _;

use imasm_core::check;
use imasm_core::classic::Token as CTok;
use imasm_core::imasm16_3::{self, Token16_3, ALL_TOKENS};

// ── the derived alphabet (identical derivation to the metal) ──────────────

fn vocab() -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    for t in ALL_TOKENS {
        if t == Token16_3::Fsplit3 {
            v.push(CTok::Fsplit.code().to_string());
            v.push(CTok::Ffuse.code().to_string());
        }
        v.push(t.glyph().to_string());
    }
    for s in imasm_core::state_order() {
        v.push(s.glyph().to_string());
    }
    v.push("⊤".to_string());
    for p in ["Ð", "Þ", "Ř", "Φ", "ƒ", "Ç", "Γ", "ɢ", "Ħ", "Σ", "Ω"] {
        v.push(p.to_string());
    }
    for c in 0x10450..=0x1047F_u32 {
        v.push(char::from_u32(c).unwrap().to_string());
    }
    for s in ["⟨", "⟩", "·", "⋈", "|", " ", "\n", "?"] {
        v.push(s.to_string());
    }
    v
}

fn id_of(voc: &[String], g: &str) -> u32 {
    voc.iter().position(|e| e == g).unwrap_or(voc.len() - 1) as u32
}

fn decode(voc: &[String], ids: &[u32]) -> String {
    ids.iter().map(|&i| voc.get(i as usize).map(|s| s.as_str()).unwrap_or("?")).collect()
}

// ── the baked tensor store ────────────────────────────────────────────────

struct Store {
    tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Store {
    fn load(b: &[u8]) -> Option<Store> {
        if b.len() < 8 || &b[0..4] != b"VITA" {
            return None;
        }
        let mut off = 4;
        let rd_u32 = |b: &[u8], o: &mut usize| -> u32 {
            let v = u32::from_le_bytes([b[*o], b[*o + 1], b[*o + 2], b[*o + 3]]);
            *o += 4;
            v
        };
        let count = rd_u32(b, &mut off) as usize;
        let mut tensors = BTreeMap::new();
        for _ in 0..count {
            let nl = rd_u32(b, &mut off) as usize;
            let name = std::str::from_utf8(&b[off..off + nl]).ok()?.to_string();
            off += nl;
            let nd = rd_u32(b, &mut off) as usize;
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(rd_u32(b, &mut off) as usize);
            }
            let n: usize = dims.iter().product();
            let mut data = Vec::with_capacity(n);
            for _ in 0..n {
                data.push(f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]));
                off += 4;
            }
            tensors.insert(name, (dims, data));
        }
        Some(Store { tensors })
    }

    fn t(&self, name: &str) -> &(Vec<usize>, Vec<f32>) {
        self.tensors.get(name).expect("baked tensor missing")
    }
    fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

// ── minimal f32 math — byte-identical to the metal (same libm) ────────────

fn expf(x: f32) -> f32 {
    libm::expf(x)
}

fn linear(x: &[f32], t: usize, din: usize, w: &[f32], dout: usize, bias: Option<&[f32]>) -> Vec<f32> {
    let mut y = vec![0f32; t * dout];
    for r in 0..t {
        for o in 0..dout {
            let mut acc = if let Some(b) = bias { b[o] } else { 0.0 };
            let xr = &x[r * din..(r + 1) * din];
            let wr = &w[o * din..(o + 1) * din];
            for i in 0..din {
                acc += xr[i] * wr[i];
            }
            y[r * dout + o] = acc;
        }
    }
    y
}

fn rms_norm(x: &mut [f32], t: usize, d: usize, w: &[f32]) {
    for r in 0..t {
        let row = &mut x[r * d..(r + 1) * d];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / libm::sqrtf(ms + 1e-6);
        for i in 0..d {
            row[i] = row[i] * inv * w[i];
        }
    }
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + expf(-v))
}

fn rope(row: &mut [f32], pos: usize) {
    let half = HD / 2;
    for i in 0..half {
        let theta = 1.0 / libm::powf(10000.0, 2.0 * i as f32 / HD as f32);
        let a = pos as f32 * theta;
        let (c, s) = (libm::cosf(a), libm::sinf(a));
        let x1 = row[i];
        let x2 = row[half + i];
        row[i] = x1 * c - x2 * s;
        row[half + i] = x1 * s + x2 * c;
    }
}

fn ring_shift(x: &mut [f32], t: usize, forward: bool) {
    let half = D / 2;
    let theta: f32 = if forward { 0.1 } else { -0.1 };
    let (c, s) = (libm::cosf(theta), libm::sinf(theta));
    for r in 0..t {
        for i in 0..half {
            let a = x[r * D + i];
            let b = x[r * D + half + i];
            x[r * D + i] = a * c - b * s;
            x[r * D + half + i] = a * s + b * c;
        }
    }
}

// ── the 26-step protocol forward pass (mirror of the metal) ──────────────

const D: usize = 144;
const HEADS: usize = 12;
const HD: usize = D / HEADS;

struct Vita {
    store: Store,
    voc: Vec<String>,
}

impl Vita {
    fn load(weights: &[u8]) -> Option<Vita> {
        Some(Vita { store: Store::load(weights)?, voc: vocab() })
    }

    fn forward_last(&self, ids: &[u32]) -> Vec<f32> {
        let s = &self.store;
        let t = ids.len();
        let vocab_n = self.voc.len();
        let (_, emb) = s.t("embed.weight");
        let mut x = vec![0f32; t * D];
        for (r, &id) in ids.iter().enumerate() {
            x[r * D..(r + 1) * D].copy_from_slice(&emb[id as usize * D..(id as usize + 1) * D]);
        }
        let mut stack: Vec<Vec<f32>> = Vec::new();

        for step in 0..26 {
            let pre = |n: &str| format!("step{step}.{n}");
            if s.has(&pre("down.weight")) {
                let (_, wd) = s.t(&pre("down.weight"));
                let (_, wu) = s.t(&pre("up.weight"));
                let mut z = linear(&x, t, D, wd, 12, None);
                for r in 0..t {
                    let row = &mut z[r * 12..(r + 1) * 12];
                    let n = libm::sqrtf(row.iter().map(|v| v * v).sum::<f32>()) + 1e-8;
                    for v in row.iter_mut() {
                        *v /= n;
                    }
                }
                let up = linear(&z, t, 12, wu, D, None);
                for i in 0..t * D {
                    x[i] += up[i];
                }
            } else if s.has(&pre("fork.weight")) {
                let (_, wn) = s.t(&pre("norm.weight"));
                let (_, wf) = s.t(&pre("fork.weight"));
                let (_, wu) = s.t(&pre("fuse.weight"));
                let mut xn = x.clone();
                rms_norm(&mut xn, t, D, wn);
                let qkv = linear(&xn, t, D, wf, 3 * D, None);
                let mut ctx = vec![0f32; t * D];
                for h in 0..HEADS {
                    let mut q = vec![0f32; t * HD];
                    let mut k = vec![0f32; t * HD];
                    let mut v = vec![0f32; t * HD];
                    for r in 0..t {
                        for i in 0..HD {
                            q[r * HD + i] = qkv[r * 3 * D + h * HD + i];
                            k[r * HD + i] = qkv[r * 3 * D + D + h * HD + i];
                            v[r * HD + i] = qkv[r * 3 * D + 2 * D + h * HD + i];
                        }
                    }
                    for r in 0..t {
                        rope(&mut q[r * HD..(r + 1) * HD], r);
                        rope(&mut k[r * HD..(r + 1) * HD], r);
                    }
                    let scale = 1.0 / libm::sqrtf(HD as f32);
                    for r in 0..t {
                        let mut sc = vec![0f32; r + 1];
                        let mut mx = f32::NEG_INFINITY;
                        for c in 0..=r {
                            let mut a = 0f32;
                            for i in 0..HD {
                                a += q[r * HD + i] * k[c * HD + i];
                            }
                            a *= scale;
                            sc[c] = a;
                            if a > mx {
                                mx = a;
                            }
                        }
                        let mut sum = 0f32;
                        for c in 0..=r {
                            sc[c] = expf(sc[c] - mx);
                            sum += sc[c];
                        }
                        for c in 0..=r {
                            let w = sc[c] / sum;
                            for i in 0..HD {
                                ctx[r * D + h * HD + i] += w * v[c * HD + i];
                            }
                        }
                    }
                }
                let fused = linear(&ctx, t, D, wu, D, None);
                for i in 0..t * D {
                    x[i] += fused[i];
                }
            } else if s.has(&pre("floor.weight")) {
                let (_, wn) = s.t(&pre("floor.weight"));
                rms_norm(&mut x, t, D, wn);
            } else if s.has(&pre("delta.weight")) {
                stack.push(x.clone());
                let (_, w) = s.t(&pre("delta.weight"));
                let dx = linear(&x, t, D, w, D, None);
                for i in 0..t * D {
                    x[i] += dx[i];
                }
            } else if s.has(&pre("mu.weight")) {
                let right = stack.pop().unwrap_or_else(|| x.clone());
                let (_, w) = s.t(&pre("mu.weight"));
                let mut joined = vec![0f32; t * 2 * D];
                for r in 0..t {
                    joined[r * 2 * D..r * 2 * D + D].copy_from_slice(&x[r * D..(r + 1) * D]);
                    joined[r * 2 * D + D..(r + 1) * 2 * D]
                        .copy_from_slice(&right[r * D..(r + 1) * D]);
                }
                let mx = linear(&joined, t, 2 * D, w, D, None);
                for i in 0..t * D {
                    x[i] += mx[i];
                }
            } else if s.has(&pre("t_gate.weight")) {
                let (_, w) = s.t(&pre("t_gate.weight"));
                let (_, b) = s.t(&pre("t_gate.bias"));
                let g = linear(&x, t, D, w, D, Some(b));
                for i in 0..t * D {
                    x[i] *= sigmoid(g[i]);
                }
            } else if s.has(&pre("f_gate.weight")) {
                let (_, w) = s.t(&pre("f_gate.weight"));
                let (_, b) = s.t(&pre("f_gate.bias"));
                let g = linear(&x, t, D, w, D, Some(b));
                for i in 0..t * D {
                    x[i] *= 1.0 - sigmoid(g[i]);
                }
            } else if s.has(&pre("hold.weight")) {
                let arm = stack.pop().unwrap_or_else(|| x.clone());
                let (_, w) = s.t(&pre("hold.weight"));
                let mut held = vec![0f32; t * D];
                for i in 0..t * D {
                    let a = arm[i];
                    held[i] = x[i] * (a * sigmoid(a)); // silu
                }
                let hx = linear(&held, t, D, w, D, None);
                for i in 0..t * D {
                    x[i] += hx[i];
                }
            } else if s.has(&pre("record.weight")) {
                let (_, w) = s.t(&pre("record.weight"));
                let rx = linear(&x, t, D, w, D, None);
                for i in 0..t * D {
                    x[i] += rx[i];
                }
            } else {
                if step == 6 || step == 14 {
                    ring_shift(&mut x, t, step == 6);
                }
            }
        }
        let (_, wh) = s.t("head.weight");
        let last = &x[(t - 1) * D..t * D];
        let mut logits = vec![0f32; vocab_n];
        for o in 0..vocab_n {
            let mut a = 0f32;
            for i in 0..D {
                a += last[i] * wh[o * D + i];
            }
            logits[o] = a;
        }
        logits
    }

    /// One turn: identical control flow to the metal's speak_turn.
    /// Returns (query word, full turn, gate verdict).
    fn speak_turn(&self, seed: u64, temp: f32, word_cap: usize) -> (String, String, char) {
        let voc = &self.voc;
        let open_id = id_of(voc, "⊢");
        let seal_id = id_of(voc, "⊣");
        let split_id = id_of(voc, "◇");
        let fuse_id = id_of(voc, "●");
        let lang_id = id_of(voc, "⟨");
        let rang_id = id_of(voc, "⟩");
        let dot_id = id_of(voc, "·");
        let query_legal: Vec<u32> =
            "⊢⊣><=⊙◇●∈∋+×⊞~≁¬".chars().map(|c| id_of(voc, &c.to_string())).collect();
        let mut answer_legal: Vec<u32> = imasm_core::state_order()
            .iter()
            .map(|s| id_of(voc, &s.glyph().to_string()))
            .collect();
        answer_legal.extend([id_of(voc, "⊙"), id_of(voc, "×"), dot_id, rang_id]);
        let word_initial = [open_id, id_of(voc, "⊙")];

        let mut rng = {
            let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            if z == 0 { 1 } else { z }
        };
        let mut next_f32 = move || -> f32 {
            let mut z = rng;
            z ^= z >> 12;
            z ^= z << 25;
            z ^= z >> 27;
            rng = z;
            ((z.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / ((1u64 << 24) as f32)
        };
        let mut sample = |logits: &[f32], mask: &dyn Fn(u32) -> bool| -> u32 {
            let mut mx = f32::NEG_INFINITY;
            for (i, &l) in logits.iter().enumerate() {
                if mask(i as u32) && l > mx {
                    mx = l;
                }
            }
            let mut exps = vec![0f32; logits.len()];
            let mut sum = 0f32;
            for (i, &l) in logits.iter().enumerate() {
                if mask(i as u32) {
                    exps[i] = expf((l - mx) / temp);
                    sum += exps[i];
                }
            }
            let mut r = next_f32() * sum;
            for (i, &e) in exps.iter().enumerate() {
                if e > 0.0 {
                    r -= e;
                    if r <= 0.0 {
                        return i as u32;
                    }
                }
            }
            logits.len() as u32 - 1
        };

        // δ arm
        let mut ids: Vec<u32> = vec![open_id, lang_id];
        let mut depth = 0usize;
        let mut sealed = false;
        loop {
            let logits = self.forward_last(&ids);
            let qlen = ids.len() - 2;
            let id = if sealed || qlen + depth + 2 >= word_cap {
                if depth > 0 {
                    fuse_id
                } else if !sealed {
                    seal_id
                } else {
                    rang_id
                }
            } else {
                let ql = &query_legal;
                sample(&logits, &|i: u32| {
                    if !ql.contains(&i) {
                        return false;
                    }
                    if i == open_id {
                        return qlen == 0 && word_initial.contains(&i);
                    }
                    if qlen == 0 {
                        return word_initial.contains(&i);
                    }
                    if i == fuse_id && depth == 0 {
                        return false;
                    }
                    if i == seal_id && depth > 0 {
                        return false;
                    }
                    true
                })
            };
            if id == split_id {
                depth += 1;
            }
            if id == fuse_id {
                depth = depth.saturating_sub(1);
            }
            if id == seal_id {
                sealed = true;
            }
            ids.push(id);
            if id == rang_id {
                break;
            }
        }
        let word = decode(voc, &ids[2..ids.len() - 1]);

        // skeleton + μ arm
        for g in ["◇", "⊞", "⟨", "⟩", "●", "⟨"] {
            ids.push(id_of(voc, g));
        }
        let mut alen = 0usize;
        loop {
            let logits = self.forward_last(&ids);
            let al = &answer_legal;
            let id = if alen + 1 >= 16 {
                rang_id
            } else {
                sample(&logits, &|i: u32| al.contains(&i))
            };
            ids.push(id);
            alen += 1;
            if id == rang_id {
                break;
            }
        }
        ids.push(seal_id);
        let turn = decode(voc, &ids);
        (word.clone(), turn, gate_word(&word))
    }
}

// ── the gate (imasm_core, no forward pass) ────────────────────────────────

fn gate_word(word: &str) -> char {
    let has_tri = word.chars().any(|c| matches!(c, '∈' | '∋' | '~' | '≁'));
    if has_tri {
        let steps = imasm16_3::parse_glyph_word(word);
        let (v, _) = imasm16_3::tri_ancestral_verdict(&steps);
        v
    } else {
        let toks: Vec<CTok> = word.chars().filter_map(|c| CTok::parse(&c.to_string())).collect();
        check::word_verdict(&toks).0
    }
}

// ── the spider: gate-driven survey of a specimen's edit neighborhood ─────

const QUERY_ALPHABET: &str = "⊢⊣><=⊙◇●∈∋+×⊞~≁¬";

/// Single-glyph substitutions, deletions, and insertions over the word.
/// Returns verdict tallies (T,B,N,F) over all distinct neighbors.
fn spider(word: &str) -> (usize, usize, usize, usize, usize) {
    let chars: Vec<char> = word.chars().collect();
    let alpha: Vec<char> = QUERY_ALPHABET.chars().collect();
    let mut seen = std::collections::BTreeSet::new();
    for i in 0..chars.len() {
        // substitutions
        for &a in &alpha {
            if a != chars[i] {
                let mut w = chars.clone();
                w[i] = a;
                seen.insert(w.into_iter().collect::<String>());
            }
        }
        // deletion
        if chars.len() > 1 {
            let mut w = chars.clone();
            w.remove(i);
            seen.insert(w.into_iter().collect::<String>());
        }
        // insertion before i
        for &a in &alpha {
            let mut w = chars.clone();
            w.insert(i, a);
            seen.insert(w.into_iter().collect::<String>());
        }
    }
    let (mut t, mut b, mut n, mut f) = (0usize, 0usize, 0usize, 0usize);
    for w in &seen {
        match gate_word(w) {
            'T' => t += 1,
            'B' => b += 1,
            'N' => n += 1,
            _ => f += 1,
        }
    }
    (seen.len(), t, b, n, f)
}

// ── the atlas ─────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut weights_path =
        std::env::var("HOME").unwrap_or_default() + "/mOMonadOS/vita_weights.bin";
    let mut n_seeds = 256u64;
    let mut start = 1u64;
    let mut temps: Vec<f32> = vec![0.8];
    let mut cap = 24usize;
    let mut do_spider = false;
    let mut out_path = String::from("vita_atlas.jsonl");
    let mut one: Option<(u64, f32)> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => {
                i += 1;
                weights_path = args[i].clone();
            }
            "--seeds" => {
                i += 1;
                n_seeds = args[i].parse().expect("--seeds N");
            }
            "--start" => {
                i += 1;
                start = args[i].parse().expect("--start S");
            }
            "--temps" => {
                i += 1;
                temps = args[i].split(',').map(|s| s.parse().expect("temp")).collect();
            }
            "--cap" => {
                i += 1;
                cap = args[i].parse().expect("--cap N");
            }
            "--spider" => do_spider = true,
            "--out" => {
                i += 1;
                out_path = args[i].clone();
            }
            "--one" => {
                let s: u64 = args[i + 1].parse().expect("--one SEED TEMP");
                let t: f32 = args[i + 2].parse().expect("--one SEED TEMP");
                one = Some((s, t));
                i += 2;
            }
            other => panic!("unknown arg {other}"),
        }
        i += 1;
    }

    let bytes = std::fs::read(&weights_path).expect("read weights bin");
    let vita = Vita::load(&bytes).expect("VITA blob");

    if let Some((s, t)) = one {
        let (word, turn, v) = vita.speak_turn(s, t, cap);
        println!("{turn}  gate {v}  → {}", if v == 'F' { "REFUSED" } else { "ADMITTED" });
        println!("word: {word}");
        return;
    }

    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut atlas: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut per_temp: Vec<(f32, [usize; 4], usize)> = Vec::new();

    for &temp in &temps {
        // fan the seed range across threads; results keep seed order for
        // first-seen attribution (the minimal seed is the word's true name).
        let results: Vec<(u64, String, char)> = std::thread::scope(|sc| {
            let mut handles = Vec::new();
            for w in 0..threads {
                let vita = &vita;
                handles.push(sc.spawn(move || {
                    let mut out = Vec::new();
                    let mut s = start + w as u64;
                    while s < start + n_seeds {
                        let (word, _, v) = vita.speak_turn(s, temp, cap);
                        out.push((s, word, v));
                        s += threads as u64;
                    }
                    out
                }));
            }
            let mut all: Vec<(u64, String, char)> =
                handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
            all.sort_by_key(|r| r.0);
            all
        });

        let mut tally = [0usize; 4]; // T B N F over all seeds (the mass)
        let mut novel = 0usize;
        for (seed, word, v) in results {
            let vi = match v {
                'T' => 0,
                'B' => 1,
                'N' => 2,
                _ => 3,
            };
            tally[vi] += 1;
            if !atlas.contains_key(&word) {
                novel += 1;
                let mut rec = serde_json::json!({
                    "word": word,
                    "first_seed": seed,
                    "temp": temp,
                    "gate": v.to_string(),
                    "len": word.chars().count(),
                });
                if do_spider {
                    let (nn, t, b, n, f) = spider(&word);
                    rec["spider"] = serde_json::json!({
                        "neighbors": nn, "T": t, "B": b, "N": n, "F": f,
                    });
                }
                atlas.insert(word, rec);
            }
        }
        per_temp.push((temp, tally, novel));
    }

    let mut f = std::fs::File::create(&out_path).expect("create atlas");
    let mut recs: Vec<&serde_json::Value> = atlas.values().collect();
    recs.sort_by_key(|r| r["first_seed"].as_u64().unwrap_or(u64::MAX));
    for r in &recs {
        writeln!(f, "{r}").unwrap();
    }

    println!("═══ vita seed atlas — {} seeds/temp from seed {start}, cap {cap} ═══", n_seeds);
    for (temp, tally, novel) in &per_temp {
        let total: usize = tally.iter().sum();
        println!(
            "  temp {temp:>4}: mass T {:>5.1}%  B {:>5.1}%  N {:>5.1}%  F {:>5.1}%   novel words {novel}",
            100.0 * tally[0] as f64 / total as f64,
            100.0 * tally[1] as f64 / total as f64,
            100.0 * tally[2] as f64 / total as f64,
            100.0 * tally[3] as f64 / total as f64,
        );
    }
    println!("  distinct words total: {}  → {out_path}", atlas.len());
    let head: Vec<String> = recs
        .iter()
        .take(8)
        .map(|r| {
            format!(
                "    seed {:>4}  {}  gate {}",
                r["first_seed"], r["word"].as_str().unwrap_or(""), r["gate"].as_str().unwrap_or("")
            )
        })
        .collect();
    println!("  smallest true names:\n{}", head.join("\n"));
}
