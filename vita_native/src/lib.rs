//! vita_native — the Grammar-native substrate for the vae_vita operator model.
//!
//! Everything here is Grammar-designed: the vocabulary is the closed IMASM
//! symbol alphabet (no BPE, no byte fallback — one glyph, one token, one
//! meaning), the template is the word itself (⊢ opens, δ carries the query,
//! ⊞ holds the thinking, μ carries the answer, ⊣ seals), and training data is
//! emitted by the imasm instruments so every label is tool-grounded.

pub mod alphabet;
pub mod model;
pub mod template;
