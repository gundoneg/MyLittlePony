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

Notebook: `c_mercury_port.ipynb` (regenerate with `build_c_mercury_nb.py`).

### Run-2 result — NEGATIVE again, sharper diagnosis

KV-cached generation worked (corpus 1339s → 86s, ×15). C improved zoo donors
(masked-CE@0.5 7.09 → 4.91), but zero-shot on supra:

| held masked-CE | t=0.3 | t=0.5 | t=0.7 | t=0.9 |
|---|---|---|---|---|
| floor (raw bidir) | 5.44 | 6.04 | 7.05 | 9.39 |
| B* = C(supra) | 8.34 | 8.34 | 8.67 | **8.71** |

B* beats the floor **only at t=0.9** and is flat ~8.3–8.7 across t; reconstruction 11.6% vs
floor 22.9%; shuffled-signature control still matches (8.18 vs 8.34). Reading: C learned a
roughly global "smooth the predictions" delta — useful for weak zoo donors in the high-noise
regime, destructive for a sharp trained model at low noise.

**Diagnosis — donor-distribution gap.** The zoo (shallow, barely-trained, random-init
spectra) is statistically nothing like a well-trained 12-layer model: supra's block
signatures are OOD for C's encoder, and the corrections C learned for weak matrices are the
wrong ones for trained matrices. The frame fix (run 2) was necessary but not sufficient.

## Run 3 — self-zoo: the zoo IS supra's own sub-stacks — **POSITIVE**

"Align the zoo and supra into one frame" taken to its logical end: build the zoo from
**depth-truncated sub-stacks of supra itself** (layers 0..L−1 + final norm + tied head,
L ∈ {2,4,6,8,10} — valid AR models, AR CE 8.75/7.75/7.87/5.70/2.79). Zoo frames and weight
statistics are then *exactly* the target's; C learns the AR→denoiser per-block rule on real
supra blocks in shallow contexts and extrapolates to the full L=12 stack. Honest disclosure:
the zoo shares weights with the target — held-out is the full-depth composition and blocks
10–11 (never seen as interior). B is still never trained. Zoo pretraining disappears
(truncations are free) — the session is ~10 min.

### Result — B* = C(supra) beats the floor at every noise level, and the control proves C reads the weights

| held masked-CE (uniform=10.37, donor AR=1.51) | t=0.3 | t=0.5 | t=0.7 | t=0.9 |
|---|---|---|---|---|
| floor (raw supra, bidirectional) | 5.49 | 5.94 | 6.99 | 9.54 |
| **B\* = C(supra)** | **5.10** | **5.11** | **5.14** | **5.18** |

- B\* is **flat ~5.1 across all mask rates** — genuine denoiser behaviour — while the floor
  degrades with noise; decisive at t=0.9 (5.18 vs 9.54). **B was never trained**; this is
  entirely C's emitted per-block deltas + [MASK] embedding.
- **Signature control (decisive):** shuffling the block signatures fed to C gives masked-CE
  **8.44** vs B\* **4.97** — a huge gap (in runs 1–2 this control *matched* B\*). So **C
  genuinely reads each block's spectrum**; the per-block weight-translation mechanism is real,
  not a global smoothing trick.
- **Extrapolation probe:** applying C's deltas to blocks 0..9 only (seen in truncation
  contexts) and leaving 10–11 raw scores **4.34** — better than deltas on all 12 (4.97). The
  seen-context blocks transfer cleanly; the last two (never an interior at trunc≤10) are the
  weak point (fixable: include L=11/12 truncations or a separate last-block treatment).
- Reconstruction (25% mask): floor 24.4% vs B\* 25.6% (low-noise, both ok). From-scratch
  parallel generation is still degenerate — the hardest mode at this tiny budget.

### Verdict
The project's thesis is demonstrated on a real model: **knowledge moved across an
architectural change (AR → discrete diffusion) purely by the weight-translator C, with zero
training of model B**, using only data sampled from the donor — and a control confirms C
operates per-block on the actual weights. Limitations are quality/scale (B\* is a weak
denoiser, masked-CE ~5.1 ≫ donor AR 1.5; parallel generation not yet fluent) and the
self-zoo sharing weights with the target. The mechanism is proven; closing the
quality gap is the scale/budget axis (more truncation depths incl. 11–12, larger rank,
bigger corpus), or the BASELINE continued-training ceiling for comparison.

## Run 4 — the five decision-graph paths — quality jump, generation trap diagnosed

Corpus v2 (unigram-prompted) + L=11 truncation + 64×64 subspace + KD soft targets + 6000 steps:

| held masked-CE (donor AR = 1.60) | t=0.3 | t=0.5 | t=0.7 | t=0.9 |
|---|---|---|---|---|
| floor (raw bidir) | 6.01 | 6.84 | 7.62 | 9.30 |
| B\* run-3 (for reference) | ~5.1 flat | | | |
| **B\* run-4** | **3.16** | **3.66** | **4.09** | **4.82** |

- **Reconstruction 34.5% vs floor 14.9%** (run-3: 25.6 vs 24.4 — barely separated), and the
  recovered text is *readable* (near-semantic recovery of a weather-advice paragraph).
- **Probe closed:** deltas on 0..10 = 3.59 vs all-12 = 3.58 (run-3: 4.34 vs 4.97) — the L=11
  truncation fixed the unseen-blocks weakness exactly as the graph predicted.
- Control still cleanly separated: shuffled signatures 8.53 vs 3.58.

**Remaining failure — from-scratch generation collapses into the *filler-confidence trap*:**
in a sea of masks the model's most-confident tokens are whitespace/newline/`</s>`; raw
confidence ranking reveals them first, they become the anchors, and the cascade fills the
sequence with fillers ([B] = all newlines, [C] = all spaces; [A], the only sampler with
non-zero constant temperature, produced fragments of content). Reconstruction works because
real-text anchors exist. The run-4 prompted test also accidentally used a junk prompt row.

## Run 5 (queued) — sampler-only fixes for the filler trap
PMI-debiased ranking (`conf = log p(tok) − β·log prior(tok)`, prior = corpus unigram) so
reveals favor tokens confident *relative to their base rate*; temperature floor 0.3 until the
last 10% of steps (no premature argmax collapse); banned special ids in generation; prompts
selected by coherence (lowest per-seq donor AR CE). No change to C or training.

## Run 5 — WARM measured; generation trap persists in a new form

**WARM (the phases-3–8 metric, brought to the port):** fine-tune B by the diffusion loss
from two starts:

| N steps | from B\*=C(supra) | from floor |
|---|---|---|
| 0 | **3.61** | 6.91 |
| 100 | 3.14 | 3.31 |
| 200 | 3.09 | 3.03 |
| 800 | 2.75 | 2.56 |

**Translation value: C's emission ≈ a 100-step head start of full fine-tuning** (the floor
needs N=100 to match B\*'s zero-shot). The advantage is *transient* — curves converge by
N≈200 (and the floor ends slightly lower at N=800) — i.e. C provides initialization value,
not a better training trajectory; consistent with warm-start's role in the toy phases.

**Generation:** the trap moved but persisted: [B] produced "the the the…" — PMI debias only
reordered *reveals* while the *predictions* themselves were frequency-dominated; and the
prompt picker chose junk again (lowest AR CE = the most predictable = degenerate filler
rows — exactly backwards). Underlying profile: B\* is strong at dense-anchor infill
(CE 3.16 @ t=0.3) and weak sparse (4.82 @ t=0.9), so from-scratch parallel text is its
hardest regime.

## Run 6 (queued) — debias the prediction, not just the reveal order
Sampler: subtract `samp_beta·log(prior)` from the logits BEFORE softmax (frequency-penalized
prediction) + repetition penalty `rep_gamma·log1p(count)`; semi-AR blocks shrunk to 16
(denser anchors); prompts picked by max unique tokens (content-rich), not min CE. Plus the
practical-recipe demo: generation from the **B\*+100-steps** warm point — the model the warm
curve says matches the translation-value point — both prompted and unconditional.
