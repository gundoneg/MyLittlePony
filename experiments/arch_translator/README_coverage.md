# Phase 8 — coverage vs depth: when does "train shallow → port deep" work?

## The question
Phase 6 found the per-block translator `C` transfers near-perfectly when trained **at**
the target depth (exp3, L=3: 98.3%), but **shallow→deep** transfer failed (exp2, L=2→L=4:
near-chance). The user's objection: `PerLayerTranslator` translates each block as a
**local** function of that block's own weights (the σ-free positional signature
`per_block_feature = P·WqᵀWk·Pᵀ`, xlate.py:30), so the failure is not "depth is
unreachable" but "the shallow zoo did not **cover** the deep block types." Claim:
a shallow-but-**diverse** zoo that spans the deep target's per-block features should let
`C` translate at any depth.

Two facts established up front:
- **`per_block_feature` is invariant to the phase-5 data-align rotation** (`P·QᵀMQ·Pᵀ =
  P·M·Pᵀ` for orthogonal `Q`). So alignment is **not** the lever for the per-block
  translator — it cannot have been the missing ingredient in exp2.
- **Depth-needed ⟺ recursion (for attention).** Any task that genuinely needs depth is
  recursive (block `l` consumes block `l-1`'s output), so a "deep-layer type" cannot be
  exhibited by a too-shallow donor in isolation. Pure pointer-chasing is the worst case.

## Coverage metric (`coverage.py`)
On the exact vectors `C` consumes (`per_block_feature`, post-align, frame-invariant):
- `nn_reconstruction_coverage` — per deep block, nearest-neighbour cosine into the shallow
  block-feature bank; report **mean** and **worst-case (min)**. Worst-case is the
  diagnostic: transfer should fail when some deep block has **no** near neighbour.
- `subspace_overlap_coverage` — fraction of deep-feature energy in the shallow PCA span.

## Phase A — cross-depth transfer, realistic pipeline (`run_cover.py`)
Independent-init + data-align (the exp3 pipeline), pointer-chasing, donors at 100%. Train
`C` on a **narrow** shallow zoo (L=2, hops=2) → zero-shot apply to a **deep** target that
**genuinely uses its depth** (raw-donor layer-ablation confirms it), vs the native-depth
ceiling. (Ablation is measured on **raw** donors: an aligned state rotates only the
translator-relevant columns, so it is not a runnable transformer and ablates to a spurious
~0%.)

**Result (L2 → L3, feat+depthfrac):**

| | zero-shot | warm@15 | s90 |
|---|---|---|---|
| train L2, test **L2** (sanity) | 94.2% | 100% | 0 |
| **TRANSFER L2→L3** | **7.2%** (chance 6.2%) | 43.9% | 35 |
| native L3→L3 (ceiling) | **95.1%** | 100% | 0 |

- deep-target layer-ablation `[54%, 31%, 77%]` → all 3 layers genuinely used.
- **coverage L2→L3: worst-case NN cosine 0.058, subspace overlap 0.14 ≈ 0.**
- (L2→L4 gives the same picture: transfer 6.9% vs native 89.7%, coverage worst-case 0.042.)

**Conclusion.** Shallow→deep transfer fails (chance) even though the pipeline is realistic,
the target genuinely uses depth, and native translation at that depth is ~perfect. The
**coverage metric correctly predicts the failure** (near-zero). This confirms the user's
**lens** — it is a coverage problem — and pins the cause: an L=2 donor's blocks physically
cannot exhibit the hop-3 signature (the **exhibitability barrier**). For genuinely
recursive depth, a **narrow** shallow zoo cannot manufacture the coverage. Alignment was
not the missing ingredient, exactly as predicted by feature-invariance.

What Phase A does **not** test: whether a **diverse** shallow zoo can manufacture coverage,
and how that depends on how recursive (vs parallelisable) the task's depth is. That is
Phase B.

## Phase B — layer-interdependence knob (in progress)
A pointer-chase with a per-hop **continue** (recursive, adds required depth) vs **reset**
(parallelisable, computable from the layer-0 input) branch, mixed by `alpha = P(continue)`:
`alpha=1` = pure recursion (Phase-A regime), `alpha=0` = depth not required. Sweep `alpha`,
train `C` on **narrow** vs **diverse** shallow zoos, apply to a deep target, and correlate
zero-shot with the measured coverage to locate the boundary `alpha*` where shallow coverage
stops sufficing. Confirms the hypothesis if diverse-shallow reaches the native ceiling for
`alpha < alpha*` while narrow stays at chance, with coverage tracking the transition.
