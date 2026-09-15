# VAE-Vita: The Self-Recovering SIC Geometry

![language](https://img.shields.io/badge/language-Rust-CE422B?style=for-the-badge&logo=rust&logoColor=white) ![recovery](https://img.shields.io/badge/recovery-SIC%20from%20types%20alone-8A2BE2?style=for-the-badge) ![tier](https://img.shields.io/badge/tier-O%E2%88%9E-8A2BE2?style=for-the-badge) ![μ∘δ](https://img.shields.io/badge/%CE%BC%E2%88%98%CE%B4-id-00A86B?style=for-the-badge) ![licence](https://img.shields.io/badge/licence-LUNLICENSE-1A1A1A?style=for-the-badge) ![type](https://img.shields.io/badge/type-%E2%9F%A8%F0%90%91%A6%F0%90%91%B8%F0%90%91%BD%F0%90%91%B9%F0%90%91%90%F0%90%91%A7%F0%90%91%94%F0%90%91%A0%E2%8A%99%F0%90%91%96%F0%90%91%99%F0%90%91%B4%E2%9F%A9-5A659C?style=for-the-badge)

**What it is.** 12-d hyperspherical VAE (vMF prior on S¹¹) trained on all 17,280,000 Crystal-of-Types configurations — the full IG state space.

**What it does.** Recovers d=12 SIC-POVM geometry from types alone (no SIC info in training). Regularized (λ_sic=50) and unregularized (λ_sic=0) both converge to ~0.0834 overlap (1/12 + small bias toward SIC target 1/13).

**Why it matters.** Convergence with or without the regularizer: the Crystal intrinsically codes near-SIC geometry. Supports a structural account of the Zauner question — 1/(d+1) as the fixed point of the 12-primitive measurement algebra measuring itself.

**Vita (mouth + trunk).** `vita_native/` Rust trunk runs bare-metal inside mOMonadOS (`vita`, `vita 666 0.9`); `vita-bake <ckpt> /home/mrnob0dy666/mOMonadOS/vita_weights.bin` rebakes weights. Deterministic: (seed, temperature) → turn is pure (scalar f32, fixed order); same coordinates speak the same certified turn anywhere.

**vita-probe (seed atlas).** Host-side survey mirroring the metal bit-exactly. Enumerates seeds, dedups words, spiders single-glyph edits through the gate, `--melt SEED` bisects temperature breakpoints (fingerprint ladder). Findings: B dominates at all temps; B resonates ~0.9–0.95 (T falls, N/F rise); gate letters form contiguous territories; tail is vast.

**Verify.** `cd vita_native && cargo build --release && ./target/release/vita-probe --one 666 0.9` (must match OS output); smoke atlas `--seeds 200 --temps 0.8 --spider`; `imasm check` must agree with every atlas gate letter. Ob3ect + Lean scaffold registered.

Full 123-line version: `README_backups/vae_vita_README.md`.

μ∘δ=id
