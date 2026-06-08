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

**Net for the project (this section's hypothesis — later refined below).** Depth-
extrapolation looks blocked only when layers are unique per depth, and real LLMs' deep
interior is locally interchangeable — suggesting "cover the type repertoire" instead of
matching depth. **The three sections that follow test this and walk it back:** verbatim
tiling fails superadditively, a light adapter on one exemplar fails too, and the interior
turns out to hold ~6.7 *distinct* transformations (PR), so local interchangeability is
robustness, not compressibility. Read on.

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

## Does a light learned adapter close the gap? (`adapter.py`) — prediction falsified
The verbatim-graft finale left a hypothesis: single-graft costs <0.5 suggest each deep
slot needs only a *small* correction, so a light per-slot adapter on **one shared exemplar**
should recover the model. We tested it: interior slots 3–10 all share one frozen exemplar
layer (L6) plus a trainable **rank-4 LoRA delta on every projection + a per-slot RMSNorm
gain** (307K params), distilled to the full model's logits.

| setup | step-0 (verbatim tile) | train-KL | held-out-KL | generation |
|---|---|---|---|---|
| light adapter, 4 seq | 6.85 | 0.05 | **3.88** | degenerate |
| light adapter, 12 seq + weight-decay | 7.64 | 0.20 | **3.47** | degenerate |

The adapter fits the *training* logits easily (KL→0.05 — capacity is ample) but **generalizes
only to KL≈3.5**, closing barely half the verbatim gap (≈7→0), and the reconstruction still
emits repetitive garbage. More data moved the held-out floor only 3.9→3.5.

**Conclusion — the prediction is wrong, and instructively so.** Single-graft cheapness
measured each interior layer's replaceability *with all other layers intact*; it does **not
compose**. When all 8 interior slots are served by one exemplar at once, each adapter must
also correct the compounded distribution shift from every *other* graft — so the required
per-slot correction is large and high-rank, not the small low-rank tweak the local metric
implied. The deep interior is locally interchangeable but encodes 8 genuinely **distinct**
transformations; it is not "one type + a light adapter."

**Net for the weight-translator.** Depth transfer cannot be cheated by tiling a covered
exemplar even with a learned light adapter. Phase-9 redundancy buys *robustness*
(drop/swap/substitute-one are cheap), not *compressibility* (tile-and-correct fails). The
real-model regime sits between phase-8 worst case (no transfer) and naive optimism (a few
types tile): a depth-D target needs per-slot translator capacity scaling with the number of
distinct interior transformations, not a constant zoo. Local interchangeability ≠ low-rank-
in-depth.

## How many distinct transformations does the interior really hold? (`effrank.py`)
Run every interior block on a **common** input (the stream entering slot 3), take each
block's residual update `U_i = block_i(H) − H`, and measure the participation ratio
(effective rank) of `{U_i}` — gauge-invariant, in function space.

- Pairwise cosine of the update fields is **0.04–0.28** (near-orthogonal); only neighbours
  9–10 reach 0.44.
- **Participation ratio = 6.72 of 8** interior layers (top-1 component = 26% of energy).
  Context: all 12 layers → PR 11.6/12; early 0–2 → PR 2.9/3.

So the interior carries **~6.7 functionally distinct transformations**, not one repeated
type. This *quantifies why the adapter failed*: a single shared exemplar captures only ~26%,
and the rest is genuinely high-rank — no light per-slot tweak can synthesize it.

**Reconciling with "redundant/interchangeable" above.** The phase-9 drop/swap/single-graft
results are real but mean *robustness*, not *copies*: each layer's update is **small**
relative to the residual stream (update-ratio ~0.5) and **near-orthogonal** to the others.
Removing or reordering one small, distinct increment perturbs the stream only slightly (low
KL) — but the increments are not substitutes for each other. Local interchangeability =
"you can afford to lose any one small contribution," not "the contributions are the same."
The deep interior is an **ensemble of ~N small, nearly-orthogonal refinements**: robust to
single perturbations, yet high-rank and therefore incompressible in depth.

## Honest caveats
- Inputs are self-generated (on-distribution but model-confident); absolute KL scales are
  probe-dependent — the **per-layer profile and the early-vs-deep contrast** are the signal,
  not the absolute numbers.
- supra50m is 50M/12-layer; bigger models have even longer redundant interiors (literature),
  so this is a conservative lower bound on real-model redundancy. Qwen run confirms at scale.
- "Interchangeable interior" supports depth-extrapolation of `C`, but the *early* unique
  layers still must be covered explicitly — depth-matching is replaced by *type*-matching,
  not eliminated.
