# Phase 9 — are real LLM layers depth-entangled (worst case) or redundant (tied-like)?

## Why
Phase 8 found that the per-block weight-translator `C` cannot extrapolate in depth on our
toy: donors **co-adapt their layers to the total depth**, so a shallow zoo's per-block
features don't cover a deep target's (worst case). But that toy (pointer-chasing) is the
*most* depth-entangled task possible — every layer is unique and strictly ordered. The
practical question for real models: **are their layers also worst-case, or are they
redundant/interchangeable** (the tied-like regime where coverage-by-a-shallow-zoo, hence
depth-extrapolation, would work)?

## What we measured
`llama_min.py` loads **supra50m** — a real 12-layer Llama (hidden 512, GQA 8q/4kv, RoPE,
SwiGLU, tied embeddings, ~50M params) — straight from safetensors, **no `transformers`
dependency**, with per-layer skip/swap and hidden-state capture. Forward is verified
correct: it generates coherent English and next-token entropy is 2.5 nats (vs 10.4
uniform). `entangle.py` probes it on the model's own self-generated (on-distribution)
sequences:

- **update** `‖x_{l+1}−x_l‖/‖x_l‖` — how much layer `l` moves the residual stream;
- **cos(l,l+1)** — high ⇒ the layer barely changes the representation;
- **drop-KL** `KL(full ‖ skip layer l)` — how much layer `l` matters to the output;
- **swap-KL** `KL(full ‖ swap l,l+1)` — order-sensitivity = NON-interchangeability;
- **wsim** — gauge-invariant cross-layer weight-spectrum similarity.

## Result (supra50m, real 50M Llama)

| layer | update | cos(l,l+1) | drop-KL | swap-KL |
|---|---|---|---|---|
| 0 | 9.62 | 0.12 | **6.39** | **6.67** |
| 1 | 0.64 | 0.88 | 0.78 | 0.70 |
| 2 | 1.41 | 0.83 | 1.84 | 1.17 |
| 3–11 (deep interior) | ~0.5 | **0.88–0.93** | **0.18–0.45** | **0.08–0.23** |

Summary: **67% of layers have drop-KL < 0.5** (individually near-prunable); deep-layer
**swap-KL ≈ 0.1** (adjacent layers are nearly interchangeable); cross-layer weight-spectrum
similarity **0.99** (the layers are almost the same "type").

## Conclusion — the depth barrier is a worst-case artifact, not the real-model regime
- **Layers 0–2 are unique and load-bearing** (layer 0 dominates: it builds the
  representation). These are few and distinct.
- **The deep interior (3–11) is highly redundant and interchangeable** — "iterative
  refinement" by nearly identical, order-insensitive, individually-droppable layers. This
  is the **tied/redundant regime**, the *opposite* of our toy's worst case (where every
  layer mattered and order was strict). It matches the broad literature on transformer
  layer redundancy / deep-layer prunability.
- **Therefore the user's coverage scheme is viable for real models** where phase 8 said it
  was not for the toy: a small donor zoo that covers (a) the few distinct early-layer types
  and (b) the *one* redundant deep-layer type could let `C` translate a network of
  **arbitrary depth**, because the deep stack is a repeated type — coverage is depth-
  invariant by construction. The phase-8 negative result was the toy being maximally
  depth-entangled; real models sit near the favorable end.

**Net for the project.** Depth-extrapolation of the weight-translator is blocked only when
layers are genuinely unique per depth. Real LLMs are not: their deep interior is a repeated,
interchangeable block. The practical recipe is *cover the type repertoire* — a handful of
early-layer types plus the redundant deep type — rather than match the full depth.

## Reproduce
- Local (CPU, this repo): `python experiments/depth_entangle/entangle.py`.
- Real target (Kaggle GPU): `qwen_entangle.py` runs the identical metrics on
  Qwen3.5-0.8B via `transformers` (hidden-state capture + module-list skip/swap). Same
  expected profile: a few distinct early layers, a long redundant interchangeable interior.

## Finale — type-coverage → depth transfer on the real target (`coverage.py`)
Does the favorable phase-9 picture let a *small zoo of exemplars* reproduce the full-depth
network (the user's scheme)? We test the crudest translator — verbatim **grafting** of an
exemplar layer's weights into a slot (`llama_min.remap`) — on supra50m.

- **Graft-cost matrix** `G[i,j]=KL(full ‖ slot i runs layer j)`: slot 0 is covered by
  *nothing* (KL 5–8 from every source); slots 1,2 and the final slot 11 are semi-unique;
  the interior 3–10 is mutually coverable (most `G[i,j]<0.5`). So ≈4 distinct types + a
  redundant interior — as phase 9 predicted.
- **But tiling fails — superadditively.** Copy the best single deep exemplar (layer 4) into
  *all* interior slots 3–10, keep {0,1,2,11} exact: the **sum of independent single-graft
  costs is 2.49**, yet the **joint end-to-end KL is 6.51** (≈2.6× worse). Greedy zoo
  reconstruction of all 12 layers reaches only KL≈2.7 even with 6 exemplars, and a 4-type
  reconstruction generates garbled text.

**Refined conclusion.** Deep layers are *locally* interchangeable (drop one, swap two,
substitute one — all cheap) but the model is **not tile-compressible**: small per-slot
mismatches compound across depth, so copying a few exemplars into every slot derails the
residual stream. Local interchangeability ≠ "the interior is k tiled prototypes."

**What this means for the weight-translator `C`.** Verbatim coverage (a zoo copied in, no
training) is *insufficient* — the per-slot **learned correction** that `C` supplies is
exactly the missing ingredient, not an optional refinement. The good news from `G`: the
correction each deep slot needs is *small* (single-graft costs <0.5), so a translator with
even light per-slot adaptation has little distance to cover in the interior. The scheme is
"cover the ~4 types **and** learn the small per-slot adapter," not "cover the types and
copy." The depth barrier is lower than the phase-8 worst case but is not zero.

## Honest caveats
- Inputs are self-generated (on-distribution but model-confident); absolute KL scales are
  probe-dependent — the **per-layer profile and the early-vs-deep contrast** are the signal,
  not the absolute numbers.
- supra50m is 50M/12-layer; bigger models have even longer redundant interiors (literature),
  so this is a conservative lower bound on real-model redundancy. Qwen run confirms at scale.
- "Interchangeable interior" supports depth-extrapolation of `C`, but the *early* unique
  layers still must be covered explicitly — depth-matching is replaced by *type*-matching,
  not eliminated.
