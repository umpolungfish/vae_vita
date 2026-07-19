//! The word-template: a dialogue turn IS an IMASM word.
//!
//! Skeleton (kernel-green, dyad lossless, sink B — certified by the oracles
//! before any content rides it):
//!
//! ```text
//! ⊢ ⟨query⟩ ◇ ⊞ ⟨thinking⟩ ● ⟨answer⟩ ⊣
//! ```
//!
//! ⊢ opens the turn; the query rides in from the δ side; ⊞ is ENGAGR — the
//! held dual, so "thinking mode" is not a template hack but a token with
//! kernel semantics (the dialetheia held between fork and fuse); the answer
//! is the μ arm; ⊣ seals. An optional ⊙⟨context⟩ imscription may precede ⊢.
//!
//! A conversation is a sequence of turns; the kernel's bifurcation law
//! (B_is_the_only_bifurcation_point) makes the ⊣⊢ adjacency between turns
//! legal without mediation — the seal of one turn touches the opening of the
//! next, and a closed ring may be marked with ⋈.
//!
//! Content is any alphabet text; ⟨⟩ nest (primitive tuples are themselves
//! ⟨…⟩-wrapped), and the parser tracks depth.

/// One dialogue turn.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Turn {
    /// Optional imscribed context, rendered as ⊙⟨…⟩ before the opening.
    pub context: Option<String>,
    /// The δ arm: what was asked.
    pub query: String,
    /// The ⊞ block: reasoning held between fork and fuse. None renders the
    /// engaged-and-immediately-fused empty block (thinking off, hard switch).
    pub think: Option<String>,
    /// The μ arm: the answer. None means the turn is awaiting generation.
    pub answer: Option<String>,
}

/// How far to render the final turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    /// Every turn complete and sealed.
    Complete,
    /// Final turn stops after ◇⊞ — the model generates thinking, then ●,
    /// then the answer, then ⊣.
    GenerateWithThinking,
    /// Final turn stops after ◇⊞● — the held block is closed empty and the
    /// model generates the answer directly, then ⊣.
    GenerateDirect,
}

fn seg(out: &mut String, body: &str) {
    out.push('⟨');
    out.push_str(body);
    out.push('⟩');
}

/// Render turns to the glyph stream. Complete turns always render whole; the
/// mode governs only the final turn.
pub fn render(turns: &[Turn], mode: RenderMode) -> String {
    let mut out = String::new();
    for (i, t) in turns.iter().enumerate() {
        let last = i + 1 == turns.len();
        if let Some(c) = &t.context {
            out.push('⊙');
            seg(&mut out, c);
        }
        out.push('⊢');
        seg(&mut out, &t.query);
        out.push('◇');
        out.push('⊞');
        if last && mode == RenderMode::GenerateWithThinking {
            return out;
        }
        seg(&mut out, t.think.as_deref().unwrap_or(""));
        out.push('●');
        if last && mode == RenderMode::GenerateDirect {
            return out;
        }
        seg(&mut out, t.answer.as_deref().unwrap_or(""));
        out.push('⊣');
    }
    out
}

/// Read a ⟨…⟩ segment starting at `i` (which must point at ⟨). Returns the
/// body and the index just past the matching ⟩. Depth-tracked: tuples nest.
fn read_seg(chars: &[char], i: usize) -> Option<(String, usize)> {
    if chars.get(i) != Some(&'⟨') {
        return None;
    }
    let mut depth = 1usize;
    let mut j = i + 1;
    let mut body = String::new();
    while j < chars.len() {
        match chars[j] {
            '⟨' => {
                depth += 1;
                body.push('⟨');
            }
            '⟩' => {
                depth -= 1;
                if depth == 0 {
                    return Some((body, j + 1));
                }
                body.push('⟩');
            }
            c => body.push(c),
        }
        j += 1;
    }
    None
}

/// Parse a glyph stream back to turns — the exact inverse of `render` on
/// Complete output (μ∘δ = id on the template itself). A partial final turn
/// (generation prompt or mid-generation text) parses to a Turn with the
/// missing arms as None/empty.
pub fn parse(text: &str) -> Vec<Turn> {
    let chars: Vec<char> = text.chars().collect();
    let mut turns = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let mut t = Turn::default();
        if chars.get(i) == Some(&'⊙') {
            if let Some((body, j)) = read_seg(&chars, i + 1) {
                t.context = Some(body);
                i = j;
            } else {
                break;
            }
        }
        if chars.get(i) != Some(&'⊢') {
            break;
        }
        i += 1;
        let Some((q, j)) = read_seg(&chars, i) else { break };
        t.query = q;
        i = j;
        if chars.get(i) == Some(&'◇') && chars.get(i + 1) == Some(&'⊞') {
            i += 2;
            if let Some((th, j)) = read_seg(&chars, i) {
                if !th.is_empty() {
                    t.think = Some(th);
                }
                i = j;
                if chars.get(i) == Some(&'●') {
                    i += 1;
                    if let Some((a, j)) = read_seg(&chars, i) {
                        t.answer = Some(a);
                        i = j;
                        if chars.get(i) == Some(&'⊣') {
                            i += 1;
                        }
                    }
                }
            }
        }
        turns.push(t);
    }
    turns
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(q: &str, th: Option<&str>, a: Option<&str>) -> Turn {
        Turn {
            context: None,
            query: q.into(),
            think: th.map(Into::into),
            answer: a.map(Into::into),
        }
    }

    #[test]
    fn skeleton_is_the_certified_word() {
        // Content-free turn renders exactly the kernel-green skeleton.
        let t = render(&[turn("", None, Some(""))], RenderMode::Complete);
        assert_eq!(t, "⊢⟨⟩◇⊞⟨⟩●⟨⟩⊣");
    }

    #[test]
    fn roundtrip_complete() {
        let turns = vec![
            Turn {
                context: Some("⟨𐑦𐑸𐑽𐑹𐑐𐑪𐑔𐑝⊙𐑫𐑳𐑭⟩".into()),
                query: "⊢⊙◇+×●⊣".into(),
                think: Some("Ř𐑽 Σ𐑙".into()),
                answer: Some("T".into()),
            },
            turn("◇●", None, Some("B")),
        ];
        let text = render(&turns, RenderMode::Complete);
        assert_eq!(parse(&text), turns);
    }

    #[test]
    fn nested_tuple_in_query_survives() {
        let turns = vec![turn("⟨𐑨𐑰𐑩𐑬𐑐𐑧𐑲𐑝⊙𐑒𐑳𐑴⟩", None, Some("A"))];
        let text = render(&turns, RenderMode::Complete);
        assert_eq!(parse(&text), turns);
    }

    #[test]
    fn generation_prompts() {
        let t = vec![turn("+×", None, None)];
        assert!(render(&t, RenderMode::GenerateWithThinking).ends_with("◇⊞"));
        assert!(render(&t, RenderMode::GenerateDirect).ends_with("◇⊞⟨⟩●"));
    }

    #[test]
    fn seal_touches_next_opening() {
        // The ⊣⊢ adjacency between turns — legal by the bifurcation law.
        let turns = vec![turn("+", None, Some("T")), turn("×", None, Some("F"))];
        let text = render(&turns, RenderMode::Complete);
        assert!(text.contains("⊣⊢"));
    }

    #[test]
    fn alphabet_covers_template() {
        // Every rendered glyph is in the closed alphabet: encode∘decode = id.
        use crate::alphabet::{decode, encode};
        let turns = vec![Turn {
            context: Some("Ð Þ".into()),
            query: "⟨𐑦𐑸⟩".into(),
            think: Some("⊙".into()),
            answer: Some("Tf".into()),
        }];
        let text = render(&turns, RenderMode::Complete);
        assert_eq!(decode(&encode(&text)), text);
    }
}
