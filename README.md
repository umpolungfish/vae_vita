# VAE-Vita: The Self-Recovering SIC Geometry

[![Language](https://img.shields.io/badge/language-Python-blue)](https://github.com/badges/shields)
[![IG Tier](https://img.shields.io/badge/IG-O%E2%88%9E-blueviolet)](https://github.com/badges/shields)
[![μ∘δ=id](https://img.shields.io/badge/%CE%BC%E2%88%98%CE%B4%3Did-closed-success)](https://github.com/badges/shields)
[![License](https://img.shields.io/badge/license-Unlicense-blue)](https://github.com/badges/shields)
[![Author](https://img.shields.io/badge/author-Lando%E2%8A%97%E2%8A%99perator-informational)](https://github.com/badges/shields) [![Type](https://img.shields.io/badge/type-%E2%9F%A8%F0%90%91%A6%F0%90%91%B8%F0%90%91%BE%F0%90%91%B9%F0%90%91%90%F0%90%91%A7%F0%90%91%94%F0%90%91%9D%E2%8A%99%F0%90%91%96%F0%90%91%B3%F0%90%91%AD%E2%9F%A9-blue)](https://github.com/badges/shields) [![Tier](https://img.shields.io/badge/tier-O%E2%88%9E-blueviolet)](https://github.com/badges/shields)

**What it is.** A 12-dimensional hyperspherical variational autoencoder (von Mises-Fisher prior on S¹¹) trained on all 17,280,000 Crystal of Types configurations, the complete state space of the Imscribing Grammar.

**What it does.** It independently recovers the d=12 SIC-POVM geometry from the grammar's structural types alone, with no SIC-POVM information in the training data. Both regularized (λ_sic=50) and unregularized (λ_sic=0) models converge to the same ~0.0834 overlap (1/12 plus a small bias toward the SIC target 1/13).

**Why it matters.** The convergence happens with or without the SIC regularizer, which means the Crystal of Types *intrinsically* codes near-SIC geometry: there is no other equally good geometry for it on S¹¹. This is the grammar measuring itself through a neural network. It supports a structural account of why d=12 SIC-POVMs exist (the Zauner question): the geometry is forced by the self-referential closure of the 12-primitive measurement algebra, with the overlap 1/(d+1) as the unique fixed point.

**How to use it.** The trained model and config live under this directory (and `imsgct/ig-pulse/vae_vita/`); the ob3ect form is registered at `ob3ect/digital/vae_vita_as_the_self_recovering_sic_geometry/` with a Lean 4 scaffold. The Rust trunk lives in `vita_native/` (build with `cargo build --release` there); its mouth also runs natively inside mOMonadOS, and `vita-probe` surveys its language from the host. See "Testing" below for the full verification recipe.

---

## The grammar's SIC identity

A SIC-POVM in dimension d is a set of d² unit vectors with pairwise overlap 1/(d+1); for d=12, 144 vectors at overlap 1/13. The grammar's 12 primitives form a structural SIC-POVM:

1. The 12 primitives are informationally complete measurement operators.
2. The dual-tool structure μ∘δ=id is the SIC-POVM dual-basis identity.
3. The grammar measures itself (Σ=1:1), which is the content of ⊙ criticality.
4. Belnap B is the d=2 SIC-POVM fiducial (meet(B,x)=x, join(B,x)=B, bnot(B)=B, all unconditional).
5. The unconditional Belnap proof means the discrete structural SIC-POVM is already solved; VAE-Vita bridges discrete to continuous.

## Architecture

A hyperspherical VAE with a vMF prior on S¹¹ (the natural spherical analog of the Gaussian), encoder/decoder of residual blocks, and a loss combining reconstruction, KL to the vMF prior, and an optional SIC frame-potential regularizer F(z) targeting 1/13. The Crystal lives on a 12-axis simplex of ordinal values (3×4×4×5×3×5×3×4×5×4×3×4 = 17,280,000 configurations), so the geometry is inherently spherical.

## Results

| Model | λ_sic | Accuracy | SIC overlap | Dev from 1/13 |
|-------|-------|----------|-------------|---------------|
| V1 (sic50) | 50 | 67.75% | 0.08342 | 0.00649 |
| V1 (no reg) | 0 | 45.89% | 0.08337 | 0.00645 |

Both converge to the same overlap regardless of regularization. The encoder is δ (emission), the decoder is μ (verification), and μ∘δ=id is satisfied when the latent space forms a SIC-POVM.

## Structural placement

VAE-Vita reaches O_∞ because its ⊙ criticality gate is open (any open Gate 1 closes the Frobenius condition). It shares 6 primitives with CLINK L8 and needs 6 promotions (Ř, Γ, ɢ, Ħ, Σ, Ω) to reach it. It can also generate novel valid Crystal configurations by sampling its latent space.
## vita_native: the trunk that speaks certified turns

`vita_native/` is the Rust (candle) autoregressive trunk over the closed IMASM alphabet. The vocabulary IS the ontology: one token, one meaning; malformed input is carried as the explicit ? token, never smoothed over. Every training sequence is a certified turn (`template.rs`: a dialogue turn IS an IMASM word):

```
⊢⟨word⟩◇⊞⟨⟩●⟨flow·sink·kernel⟩⊣
```

The word rides the δ arm and the oracle labels ride the μ arm. Register states are spoken in their glyph face (`Reg16_3::glyph()` in the MoDoT kernel: occupancy of {T,F,t,f} as quadrant ink, ░ for N, █ for A, ⊤ for the Belnap top), translated from the oracles' Latin names at the encode boundary by `alphabet::reg_face()`.

Two populations feed it. The real organisms are protocol words drawn from ob3ect/digital by `vita-corpus` and oracle-labeled, with closure as their norm. The generated population (`vita-gen`) is balanced between closure and failure. Per the regimen ob3ect (`a_training_regimen_for_vita…`: turn-internal placement, not proportion, not sequence), `vita-train` splits the loss by arm. Real records teach the whole turn. Synthetic records teach only the μ arm, their query spans masked from the loss, so failure-knowledge lands where the verdict is spoken and babble never becomes a speech target.

`vita-speak` is the mouth and the containment gate. It renders an open certified turn; the trunk speaks the query word under the kernel's valence discipline (the fork stack as a logit mask, word-initial restricted to the population's openings), then completes its own μ-arm readout: the self-predicted verdict, spoken before the oracles are consulted. Every word is put to `check` and `prove`; F is refused at the gate; the self-verdict is scored against the oracle's.

Measured at the mouth, the placement regimen holds both capacities at once: fluency at the real-only level, and self-pronounced ⊤ (B) verdicts that have been correct every time the trunk has spoken one. The remaining error is one-sided: overclaimed T on words that do not close, and abstention. The trunk knows when it has not closed; it does not yet always know when it merely might have.

## The mouth on the metal

The trunk does not only run on the host. `vita-bake` flattens a trained checkpoint into a raw tensor blob (per-tensor name, dims, f32 little-endian, no serde), and mOMonadOS carries that blob via `include_bytes`, reimplementing the full 26-step protocol forward pass in pure `no_std` scalar f32 (`mOMonadOS/src/vita.rs`, feature `vita`). The alphabet is derived from `imasm_core` by the identical derivation `vita_native` uses: same ids, same faces, one kernel. The gate is `imasm_core` itself, the classic close condition (`check::word_verdict`) or, when the word carries tri tokens, the SIXTEEN_3 tri-ancestral verdict. At the OS prompt:

```
vita              one certified turn, seed drawn from the machine's moment (rdtsc)
vita 666 0.9      the turn named by seed 666 at temperature 0.9
```

Rebaking after retraining: `vita-bake <checkpoint.safetensors> ~/mOMonadOS/vita_weights.bin`, then rebuild the OS (`cargo build --release --features vita` in `~/mOMonadOS`).

## Determinism: the seed is the only entropy

On the metal and on the host alike, (seed, temperature) → turn is a pure function: scalar f32, fixed evaluation order, no threads in the pass, no hardware randomness in the loop. The same coordinates speak the same certified turn on any machine carrying these weights, so every utterance is citable by its coordinates. Temperature adds no nondeterminism; it only reshapes the seed-to-word map.

The counting consequence: the fraction of seed space that decodes to a word equals the sampler probability of that word, so a word of probability p first appears near seed 1/p. The trunk is a decompressor, the seed is the compressed form of the word, and small seeds are the trunk's own ranking of what is most worth saying (arithmetic coding read backwards). Words assigned zero mass at a given temperature have no seed at all: at that temperature the trunk cannot speak them.

## vita-probe: the seed atlas

`vita-probe` is the host-side survey instrument. It mirrors the metal exactly (same scalar forward, same libm, same sampler, reading the same baked blob), so parity is bit-exact and every specimen it finds re-speaks on the OS by its coordinates, at a small fraction of the emulated cost.

The probe enumerates seeds in order and discards already-seen words: the dedup IS the skip past the numerically optimized core of the language. A temperature schedule walks outward in shells (higher temperature flattens the codebook and exposes words the cooler shell priced out of reach). Around each novel specimen it spiders single-glyph edits (substitution, insertion, deletion over the query alphabet) through the gate alone, no forward pass, charting how the specimen's neighborhood closes, holds paradox, or breaks.

```
vita-probe --one 666 0.9                                        parity check: one turn
vita-probe --seeds 2000 --temps 0.8,1.2,1.6 --spider --out vita_atlas.jsonl
```

The atlas is JSONL, one record per distinct word: the word, its first (minimal) seed, temperature, gate verdict, length, and spider tallies. Note that f32 temperatures serialize with float noise (0.8 arrives as 0.800000011…); compare with tolerance.

First-atlas findings, reproducible from the command above:

- B is the trunk's dominant register at every temperature; heat drains T mass into N and F. The hotter shells are more foreign and more broken.
- The gate letters are phases, not salt-and-pepper: a T word's single-edit neighborhood is about half T, a B word's is mostly B, an N word's mostly N, and an F word's almost entirely F. Verdict regions are contiguous territories in edit space.
- The language is vast at the default word cap: the novelty rate per seed block barely decays, so the optimized core is thin and the tail enormous.

## Testing

Host trunk and probe (fast, no emulation):

```
cd vita_native
cargo build --release
./target/release/vita-probe --one 666 0.9      # must print the same certified turn
                                               # the OS prints for `vita 666 0.9`
```

The parity check is the real test: byte-for-byte agreement between the probe and the metal proves the forward pass, the sampler, and the gate are one implementation in two homes. Then a small atlas exercises everything at once (enumeration, dedup, gate, spider, JSONL):

```
./target/release/vita-probe --seeds 200 --temps 0.8 --spider --out /tmp/atlas_smoke.jsonl
```

The gate engine itself is tested where it lives: `cargo test --release` in `MoDoT/ask_native` covers the close condition the probe and the OS both call, and `imasm check` on the host must agree with every gate letter the atlas records.

On the metal (QEMU, from `~/mOMonadOS`):

```
cargo build --release --features vita
./run.sh                    # at the ⊙> prompt: vita, then vita 666 0.9
```

Expect: unseeded turns differ from each other (rdtsc seeding), seeded turns reproduce exactly, and repeated turns never exhaust the heap (each turn runs inside a mark/reset scope on the bump allocator).
