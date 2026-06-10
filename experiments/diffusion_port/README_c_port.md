# C-only port (supra50m → Mercury-2 diffusion) — results log

Two notebooks, two methods:
- `supra_to_mercury.ipynb` — **BASELINE / ceiling**: the original Mercury/DiffuLLaMA recipe
  (gradual continued-training of the transformer itself). Not the project method.
- `c_mercury_port.ipynb` — **project method**: model B is produced *only* by the weight
  translator C. `B* = C(supra50m)` = donor weights + per-block low-rank deltas emitted by C
  from gauge-invariant block signatures + an emitted [MASK] embedding. B is never trained.

## Run 1 (c_mercury_port, no zoo↔target alignment) — NEGATIVE, diagnosed

C trains fine on the zoo (train-ELBO 10.0→3.15; a zoo donor's held masked-CE@0.5 6.56→4.02).
But zero-shot on supra50m fails:

| held masked-CE | t=0.3 | t=0.5 | t=0.7 | t=0.9 |
|---|---|---|---|---|
| floor (raw supra, bidirectional) | 5.58 | 6.39 | 7.45 | 9.49 |
| B* = C(supra) | 6.38 | 6.49 | 8.29 | 9.46 |

- B* does **not** beat the floor (equal/worse). Reconstruction of 25%-masked text: floor 32.5%
  vs **B* 14.1%**. Parallel generation degenerate.
- **Control:** shuffling the block signatures fed to C gives masked-CE 6.51 vs B* 6.59 —
  **equal** ⇒ C is *ignoring the signatures*.

**Diagnosis — no shared coordinate frame between the zoo and supra (the project's #1 lesson).**
C's *input* (SVD spectra) is gauge-invariant, but its *output* — a weight delta `U@Vᵀ` added
to a projection — is **basis-dependent**. The delta is learned in the zoo's frames and lands
in the wrong basis on supra, so it's noise; the signature contribution drowns in the frame
mismatch (hence the shuffled-signature control matching). This is the same Bug-2 class E0
fixed for the toy, not yet applied here. Consistent with phase 3 (independent-init zero-shot =
chance) and phase 5 (activation data-align is what recovers transfer).

**Fix (next run):** align the zoo and supra to one frame via phase-5 `data_align` (residual-
activation Procrustes on the shared self-generated probe) so the emitted delta lands in
supra's basis. C still never trains B; only an alignment (inference on the self-generated
probe — the allowed budget) is added. Then the signature can carry, and the delta can land.

Practical note: AR self-generation of the corpus took ~22 min (no KV cache) — add caching or
shrink the corpus next iteration.

## Run 2 (SVD-frame deltas + speedups) — to run on Kaggle

**Fix for the frame mismatch.** Instead of explicitly rotating the zoo into supra's frame
(which would break RMSNorm — the wrong symmetry group, phase 3c — and contaminate C with
rotation-correction behavior absent in supra), C now emits each block's correction **in that
matrix's own SVD frame**: `ΔW = Uᵣ · A(z) · Vᵣᵀ`, where `Uᵣ,Vᵣ` are the donor matrix's
intrinsic singular vectors (cached once) and `A(z)` is a small r×r correction C predicts from
the gauge-invariant signature. C's input is gauge-invariant and its **output is gauge-
covariant**, so the delta always lands in the model's current frame *by construction* — the
correct "one frame" for zoo and supra, with no model rotated. Norm corrections are per-layer
scalars (frame-free); the [MASK] embedding is a softmax combo of the donor's own rows.

**Speedups (no quality loss).** KV-cached self-generation (validated: cached == uncached
greedy, exact, on real supra) cuts corpus generation ~10×; SVD cached once; donor steps
1200→800 (donors overfit the tiny corpus regardless). The bare-BOS self-generated text is
repetitive (a 50M donor talking to itself) — that is the honest data budget; masked-CE is
measured against exactly that distribution.

Notebook: `c_mercury_port.ipynb` (regenerate with `build_c_mercury_nb.py`). Pending: the
Kaggle GPU run — does B*=C(supra) now beat the floor, and does the shuffled-signature control
finally separate (proving C reads each block)?
