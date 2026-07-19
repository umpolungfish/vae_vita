//! The closed IMASM symbol alphabet — the model's entire universe.
//!
//! One entry, one token, one meaning. No BPE, no byte fallback, no unknown-token
//! smoothing: the vocabulary IS the ontology, so tokenization is an imscription
//! (lossless R∧W∧X) and a malformed input surfaces as the explicit ? token
//! rather than passing as clean.
//!
//! Sections, in scripture order (ids are stable; extending appends, never
//! reorders):
//!   0. opcodes    — the 12 classic node-opcodes ∪ the SIXTEEN_3 extension
//!                   (☊ ☋ 3-way fork/fuse, ~ ≁ the two negations): 16 glyphs.
//!                   Glyph order follows imasm16_3::ALL_TOKENS with ◇ ● placed
//!                   beside their 3-way siblings.
//!   1. registers  — the 16 SIXTEEN_3 states P({T,F,t,f}), canonical names per
//!                   Reg16_3::name (N, A, and T-F-t-f order for the rest). The
//!                   FOUR slice {N, T, F, B} lives inside: B is the classical
//!                   render of TF, and doubles as the Belnap verdict token.
//!   2. primitives — the 12 primitive letters (⊙ already present as IMSCRIB:
//!                   glyph recurrence is structure surfacing, not collision).
//!   3. shavian    — the 48 Shavian letters (levels within a primitive scale).
//!   4. structure  — ⟨ ⟩ · ⋈ | space newline, and ? (malformed marker).

/// The vocabulary, in id order. Multi-char entries are single tokens
/// (register names like "Tf"); encoding is longest-match.
pub const VOCAB: &[&str] = &[
    // ── 0. opcodes (16) ─────────────────────────────────────────────────
    "⊢", "⊣", ">", "<", "=", "⊙", "◇", "●", "☊", "☋", "+", "×", "⊞", "~", "≁", "¬",
    // ── 1. registers (16 SIXTEEN_3 states) + B (FOUR/Belnap render of TF) ─
    "N", "A", "T", "F", "t", "f", "TF", "Tt", "Tf", "Ft", "Ff", "tf",
    "TFt", "TFf", "Ttf", "Ftf", "B",
    // ── 2. primitives (12; ⊙ appears once, in the opcode section) ───────
    "Ð", "Þ", "Ř", "Φ", "ƒ", "Ç", "Γ", "ɢ", "Ħ", "Σ", "Ω",
    // ── 3. shavian (48) ─────────────────────────────────────────────────
    "𐑐", "𐑑", "𐑒", "𐑓", "𐑔", "𐑕", "𐑖", "𐑗", "𐑘", "𐑙", "𐑚", "𐑛",
    "𐑜", "𐑝", "𐑞", "𐑟", "𐑠", "𐑡", "𐑢", "𐑣", "𐑤", "𐑥", "𐑦", "𐑧",
    "𐑨", "𐑩", "𐑪", "𐑫", "𐑬", "𐑭", "𐑮", "𐑯", "𐑰", "𐑱", "𐑲", "𐑳",
    "𐑴", "𐑵", "𐑶", "𐑷", "𐑸", "𐑹", "𐑺", "𐑻", "𐑼", "𐑽", "𐑾", "𐑿",
    // ── 4. structure ────────────────────────────────────────────────────
    "⟨", "⟩", "·", "⋈", "|", " ", "\n", "?",
];

/// Id of the malformed-input marker.
pub fn unknown_id() -> u32 {
    (VOCAB.len() - 1) as u32
}

pub fn vocab_size() -> usize {
    VOCAB.len()
}

/// Encode text to token ids by longest match at each position. A char that
/// matches no vocabulary entry becomes the ? token — malformed input is
/// carried as evidence, never dropped and never smoothed over.
pub fn encode(text: &str) -> Vec<u32> {
    let chars: Vec<char> = text.chars().collect();
    let mut ids = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let mut best: Option<(usize, u32)> = None; // (chars consumed, id)
        for (id, entry) in VOCAB.iter().enumerate() {
            let e: Vec<char> = entry.chars().collect();
            if e.len() <= chars.len() - i && chars[i..i + e.len()] == e[..] {
                if best.map_or(true, |(len, _)| e.len() > len) {
                    best = Some((e.len(), id as u32));
                }
            }
        }
        match best {
            Some((len, id)) => {
                ids.push(id);
                i += len;
            }
            None => {
                ids.push(unknown_id());
                i += 1;
            }
        }
    }
    ids
}

/// Decode token ids back to text. Exact inverse of `encode` on any text the
/// alphabet covers (μ∘δ = id: the tokenizer is itself a lossless split-fuse).
pub fn decode(ids: &[u32]) -> String {
    ids.iter()
        .map(|&id| VOCAB.get(id as usize).copied().unwrap_or("?"))
        .collect()
}

/// ROTAT on a token sequence: the cyclic shift, the ring automorphism.
pub fn rotat(ids: &[u32], k: usize) -> Vec<u32> {
    if ids.is_empty() {
        return Vec::new();
    }
    let s = k % ids.len();
    let mut out = Vec::with_capacity(ids.len());
    out.extend_from_slice(&ids[s..]);
    out.extend_from_slice(&ids[..s]);
    out
}

/// Canonical rotation: the lexicographically minimal rotation of the sequence
/// (a balanced tiling of a period-n cycle is unique up to ROTAT; this picks
/// its representative). Returns (k, canonical sequence).
pub fn canonical_rotation(ids: &[u32]) -> (usize, Vec<u32>) {
    if ids.is_empty() {
        return (0, Vec::new());
    }
    (0..ids.len())
        .map(|k| (k, rotat(ids, k)))
        .min_by(|a, b| a.1.cmp(&b.1))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_duplicate_entries() {
        let mut seen = std::collections::HashSet::new();
        for e in VOCAB {
            assert!(seen.insert(*e), "duplicate vocab entry: {e}");
        }
    }

    #[test]
    fn roundtrip_word() {
        let w = "⟨⊢⊙◇+><×●=¬⊣⟩";
        assert_eq!(decode(&encode(w)), w);
    }

    #[test]
    fn roundtrip_16_3_word() {
        let w = "⊢⊙☊+×⊞~≁☋⊣";
        assert_eq!(decode(&encode(w)), w);
    }

    #[test]
    fn registers_longest_match() {
        // "TFt" must be ONE token, not T·F·t.
        assert_eq!(encode("TFt").len(), 1);
        assert_eq!(decode(&encode("TFt")), "TFt");
        // "Tf" one token; "A" and "N" one each.
        assert_eq!(encode("Tf N A B").len(), 7);
    }

    #[test]
    fn tuple_roundtrip() {
        let t = "⟨𐑦𐑸𐑽𐑹𐑐𐑪𐑔𐑝⊙𐑫𐑳𐑭⟩";
        assert_eq!(decode(&encode(t)), t);
    }

    #[test]
    fn malformed_surfaces_as_question() {
        let ids = encode("⊢Z⊣");
        assert_eq!(ids[1], unknown_id());
        assert_eq!(decode(&ids), "⊢?⊣");
    }

    #[test]
    fn rotat_orbit_closes() {
        let ids = encode("⊢⊙◇+×●⊣");
        let n = ids.len();
        assert_eq!(rotat(&rotat(&ids, 3), n - 3), ids);
        let (_, canon) = canonical_rotation(&ids);
        // Every rotation has the same canonical representative.
        for k in 0..n {
            assert_eq!(canonical_rotation(&rotat(&ids, k)).1, canon);
        }
    }
}
