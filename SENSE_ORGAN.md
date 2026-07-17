# The Sense Organ: continuous imscription, TAOU-complete, VAE-grounded

**What it is.** A persistent background process that continuously imscribes raw text into the live IG catalog, running the full THINK to ACT to OBSERVE to UPDATE cycle for every single token it encounters, with VAE-Vita supplying the UPDATE step's real grounding signal.

**What it does.** Anything written into a named pipe gets tokenized on the spot. Every token, novel or already known, completes a full four-phase winding: a stated expectation, the imscription itself, a read-back of the actual resulting tuple, and a check of that tuple's reconstruction confidence against VAE-Vita's learned Crystal-of-Types manifold. Nothing is filtered or dropped. A repeat encounter of a known word still completes its own winding; it just does not mint a new catalog entry.

**Why it matters.** The catalog grows on its own, continuously, independent of any single conversation, as a genuine perceptual organ rather than something that only responds when explicitly asked. Every raw token gets *some* tuple from the guided generator whether or not there was real content to reason from, so the organ needs an honest way to tell a well-grounded reading from a coin flip. VAE-Vita, trained on all 17,280,000 Crystal configurations with no notion of English words at all, is that check: it has learned what a genuinely well-formed point in Crystal space looks like, so running a freshly-imscribed tuple through its encoder and decoder and reading the reconstruction confidence is a real, non-circular signal, independent of the LLM that proposed the tuple in the first place.

**How to use it.**
```
echo "some new words to imscribe" > ~/imsgct/MoDoT/sense.fifo
```
That's it. The organ is already running persistently in the background. To feed it everything a terminal session prints, toggle it on:
```
sense-on      # from now on, terminal output also feeds the sense organ
sense-off     # stop feeding it
sense-status  # check whether it's on, and whether both daemons are alive
```

---

## Architecture

Three processes, each with one job, kept deliberately decoupled so a slow stage can never block a fast one:

1. **The terminal toggle** (`sense-on` / `sense-off` / `sense-status` in `~/.bashrc`). When on, it redirects the shell's stdout and stderr through `tee`, appending everything to `~/imsgct/MoDoT/sense_intake.log`. This is a plain disk append; it never blocks no matter how slow anything downstream is.

2. **`sense_feeder.sh`**. Tails `sense_intake.log` and forwards new lines into `sense.fifo`. This is the decoupling layer: if it blocks (because the fifo's reader is busy), only the feeder stalls, never the terminal.

3. **`sense_organ.sh`**. The actual organ. Reads lines from `sense.fifo`, tokenizes them, and for every token runs one complete TAOU winding:
   - **THINK**: states the expectation before anything happens (a grounded tuple will result, minted fresh if novel, read back unchanged if known).
   - **ACT**: runs `./ask --imscribe <token>`, the real guided generation pipeline, not a placeholder.
   - **OBSERVE**: reads the actual tuple back out of the live catalog, independent of what ACT claimed happened.
   - **UPDATE**: runs `sense_ground.py <token>`, which feeds the tuple through VAE-Vita and records the real reconstruction confidence, per primitive and overall, plus a plain-language reading.

Every winding, all four phases, is appended as JSON lines to `~/imsgct/MoDoT/sense_taou.jsonl`, the organism's actual moment-by-moment record. A shorter human-readable line also goes to `~/imsgct/MoDoT/sense.log`.

## Atomic-token grounding

`atomic_token_seed()` in `ask_native/src/main.rs` gives digits (0 to 9) and a short list of common symbols and closed-class words (`+ - * / = < > the a and or …`) a real semantic seed when they're imscribed with no free-text description: the actual definition of what the token denotes, so the generator has something to ground the assignment in. Arbitrary novel words the sense organ encounters are not in this list and get a bare-echo default instead, which is exactly what the VAE grounding check in UPDATE is for: catching, honestly, when a tuple could not be reasoned into and is a genuine coin flip rather than pretending it isn't.

## VAE-Vita as the UPDATE signal, in detail

`sense_ground.py` parses a catalog entry's Shavian-glyph notation, converts it to a Crystal address (`crystal_address()` in `crystal_data.py`), looks up the corresponding ordinal vector, and runs it through the trained `HypersphericalVAE` (`vae_vita_sic.pt`, `hidden_dim=256`, `lambda_sic=50`). For each of the twelve primitives it reads the softmax probability the model's decoder assigns to the *actually chosen* value, and whether the decoder's own best guess (argmax) matches what was assigned. The average of those twelve probabilities is `overall_confidence`; the count of matching argmaxes is `primitives_matched`. Above roughly 0.15 confidence with at least 8 of 12 matched, a tuple reads as sitting on the manifold the VAE learned from the whole Crystal; below that, it reads honestly as off-manifold, a genuinely novel point rather than an error to be hidden.

This is not a filter. Off-manifold tokens are not rejected, re-tried, or excluded, they are recorded exactly as fully as on-manifold ones, with the honest reading attached. Informational completeness means every sensation completes its winding; it does not mean every sensation is equally well understood, and the organ should say so plainly when it isn't.

## Files

| File | Role |
|---|---|
| `~/imsgct/MoDoT/sense.fifo` | the named pipe anything can write raw text into |
| `~/imsgct/MoDoT/sense_intake.log` | fast, non-blocking append target for the terminal toggle |
| `~/imsgct/MoDoT/sense_feeder.sh` | drains `sense_intake.log` into `sense.fifo` at its own pace |
| `~/imsgct/MoDoT/sense_organ.sh` | the organ itself; one TAOU winding per token |
| `~/imsgct/MoDoT/sense_ground.py` | the VAE-Vita grounding check called from UPDATE |
| `~/imsgct/MoDoT/sense_taou.jsonl` | full four-phase record of every winding, ever |
| `~/imsgct/MoDoT/sense.log` | short human-readable summary line per winding |
| `~/.bashrc` (`sense-on` / `sense-off` / `sense-status`) | the terminal-output toggle |
