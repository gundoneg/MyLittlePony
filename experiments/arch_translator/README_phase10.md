# Phase 10 — re-analysis & E0: the phase-8 "depth extrapolation is impossible" verdict was a pipeline artifact

## Why
Phases 6/8 concluded that the per-block weight-translator `C` cannot transfer shallow→deep,
even at alpha=0 where the deep target barely uses its depth (`run_interdep`: narrow 14.8% vs
native 99.8%; "coverage ~0 at every alpha"). A code-level re-analysis found that conclusion
rests on **two confirmed pipeline bugs plus an OOD feature-scaling issue**, not on genuine
depth-entanglement.

## The bugs

**Bug 1 — gauge-corrupted features for every layer l≥1.** `align.py` rotated only
`blocks.0.mix.qkv` (`ROTATE_KEYS`). `per_block_feature` `S = P (Wqᵀ Wk) Pᵀ` is gauge-invariant
**only if both `pos` and `qkv` are rotated**. data_align rotated `pos` for all layers but `qkv`
for block 0 only, so for l≥1 the aligned feature carries a per-donor random `Q` (no longer
invariant). Confirmed by `e0_diag.py`:

| layer | RAW feature norm | ALIGNED (buggy) | ALIGNED (after fix) |
|---|---|---|---|
| 0 | 702 ± 62 | 702 ± 62 | 702 ± 62 |
| 1 | 796 ± 58 | **414 ± 156** | 796 ± 58 |
| 2 | 1185 ± 87 | **560 ± 223** | 1185 ± 87 |

Shallow→deep coverage (worst-case NN cosine): buggy **0.033** → fixed **0.147** (= raw). Fix:
`rotate_keys(state)` rotates every block's qkv (one shared residual basis ⇒ one Q applies).

**Bug 2 — cross-depth reference-frame mismatch.** `run_cover`/`run_interdep` aligned each depth's
zoo to *its own* donor 0, so `C`'s equivariant vocab maps `W_t/W_h/W_p` (learned in the shallow
frame) were applied to deep targets living in an unrelated deep frame ⇒ chance at every alpha.
Fix (`run_cover_fixed.common_frame`): strong per-depth `data_align` + one global full-residual
bridge rotating the deep reference onto the shallow one, so both depths share one frame.

**Issue C — OOD feature scaling.** Deep layers have systematically larger |feature| (702→1185),
so standardising with a shallow zoo's per-coordinate stats sends the deepest layer OOD. Fix
(`xlate.scale_free`): unit-direction + `log|feature|` scalar.

## Result (`run_cover_fixed.py`, L2→L3, chance 6.2%)

| | alpha=0 (deep layer idle, ablation [31,23,**1**]) | alpha=1 (recursive, ablation [52,27,**71**]) |
|---|---|---|
| OLD (buggy) transfer | 6.4% (chance) | 6.7% (chance) |
| FIXED transfer (run v1 / v2) | **69.3% / 37.6%** | **14.3% / 13.0%** |
| FIXED native ceiling (v1 / v2) | 71.6% / 57.8% | 60.8% / 85.4% |
| coverage worst-case | 0.256 | 0.139 |

**Robust across three runs (v1 single-ref frame, v2 bridged frame, tied):**
1. The OLD pipeline gives **chance (6.4%) at every alpha** — the original blanket failure reproduced.
2. The FIXED pipeline transfers **far above chance, strongest at alpha=0** (37–69%): the bugs, not
   depth-entanglement, caused the "fails even at alpha=0" result.
3. A **real residual gap remains only at alpha=1** (transfer ≈13–14% in every run, vs native
   58–85%): genuinely recursive depth (the hop-3 block has no exhibitor at depth 2) is not
   coverable from shallow donors — and the **coverage metric predicts it** (0.256 at alpha=0 vs
   0.139 at alpha=1).

## Corrected verdict
Phase-8's "zero-shot depth extrapolation is impossible / coverage is ~0 everywhere" is **false** —
that was gauge corruption + cross-depth frame mismatch + OOD scaling. The true picture is a
**graded coverage effect**: shallow→deep weight transfer works to the extent the deep target's
computation is **within the shallow repertoire** (near/above-native at alpha=0, where the deep
layer is idle), and breaks only for **genuinely recursive depth** (alpha=1). This vindicates the
user's "cover the layer-type repertoire" lens for the non-recursive regime — which, per phase 9,
is where real LLM deep layers largely sit (small near-orthogonal *refinements*, not strong
recursion), so real-model depth transfer should sit toward the favorable end.

## Honest caveats
- Toy scale (V=16, d=64, n=8 donors): absolute numbers are noisy (native ceiling 58–85% across
  seeds); the **alpha contrast and the OLD-vs-FIXED gap** are the portable signal, not the
  absolutes.
- The interior-feature-mismatch control was **regime-dependent** (features clearly used in the v1
  frame; ~no effect in the v2 frame), so the "constant-interior" confound (Finding 3) is **not
  fully excluded** — donors here vary mostly in σ (which rides the vocab maps), so the generated
  interior is only weakly donor-specific. A cleaner test needs per-donor interior diversity.
- Tied regime confirmed the alpha-contrast in *ratio* but with near-chance native ceilings (tied
  zero-shot is inherently low/noisy at toy scale; the data-align realistic pipeline is what
  produces high zero-shot, as in phase 5).
