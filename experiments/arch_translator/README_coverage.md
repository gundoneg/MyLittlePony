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

## Phase B — layer-interdependence sweep (`run_interdep.py`)
A pointer-chase with a per-hop **continue** (recursive, adds required depth) vs **reset**
(parallelisable, computable from the layer-0 input) branch, mixed by `alpha = P(continue)`.
Calibration confirms the knob: per-layer ablation rises with `alpha` (L=4 layer-0 30→64%,
layer-1 7→50%, layer-2 5→28%; the 4th layer stays idle at d=64, so the sweep uses deep=3).
Per `alpha`: train `C` on a **narrow** (uniform maxjump=2) vs a **diverse** (varied
maxjump) shallow L=2 zoo, zero-shot to a deep L=3 target; report measured coverage, the
native-depth ceiling, and the deep-target depth-usage (raw-donor ablation).

**Result (shallow L2 → deep L3, indep+align):**

| alpha | deep-ablation | cov narrow w/c | cov div w/c | zs narrow | zs diverse | zs native |
|---|---|---|---|---|---|---|
| 0.00 | `33,23,1` | 0.04 | 0.02 | 14.8% | 7.0% | **99.8%** |
| 0.25 | `32,35,3` | 0.03 | 0.02 | 12.5% | 6.4% | 98.2% |
| 0.50 | `53,25,4` | 0.05 | 0.03 | 6.3% | 5.4% | 97.8% |
| 0.75 | `53,17,22` | 0.03 | 0.03 | 7.2% | 6.0% | 95.0% |
| 1.00 | `54,31,77` | 0.06 | 0.03 | 7.2% | 5.2% | **95.1%** |

(chance 6.2%; "w/c" = worst-case NN cosine; depth-usage rises with alpha as designed.)

**Conclusion — the hypothesis is refuted, but the lens is right.**
- The **native-depth ceiling is 95–100% at every alpha**: the deep target is perfectly
  translatable when `C` is trained at its depth.
- **Shallow→deep transfer fails across the entire interdependence range**, and a
  varied-maxjump **diverse zoo does not help** (diverse ≈ narrow ≈ chance; its coverage is
  not higher). Worst-case coverage is near-zero everywhere and predicts the failure.
- **The mechanism is deeper than "unseen deep types".** Transfer fails even at `alpha=0`,
  where the deep target barely uses its depth (ablation `33,23,1`) and solves essentially
  the same near-shallow task — narrow still gives only 14.8% vs native 99.8%. So the cause
  is **depth-entanglement**: donors **co-adapt their layers to the total depth**, so the
  *same* task is decomposed across layers differently at L=2 vs L=3, and the per-block
  feature distributions structurally do not overlap across depths. A shallow zoo cannot
  manufacture coverage because the obstacle is not a missing task-type but the depth-
  dependent way computation is split across blocks.

**What this settles (the path to big models).** The user's coverage *lens* is correct and
the metric works — but for these tasks coverage cannot be created from shallow donors. This
robustly confirms phase 6's conclusion: **train `C` at (or near) the target depth**;
zero-shot depth extrapolation is not viable for the per-block translator. The remaining
escape hatch is targets with **tied/repeated layers** (universal-transformer style), where
the per-layer decomposition is depth-invariant by construction so coverage is automatic.

**Limitations.** "Diversity" here was only varied maxjump — a weak lever orthogonal to the
depth-decomposition obstacle. A stronger per-role diversity was not tried; but the `alpha=0`
same-task failure strongly suggests no shallow diversity can overcome depth-entanglement.
Toy width caps usable depth at ~3 (the 4th layer never engages at d=64); the mechanism is
portable, the absolute numbers are not.
