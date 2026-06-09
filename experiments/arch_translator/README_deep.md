# Phase 5 — recovering the shared basis from Q-A data, into depth

**Question (from the user).** Commercial models are 30+ layers. Phase 3c showed the
residual-stream basis cannot be read off static weights, and post-hoc *weight*
alignment of independently-trained nets does not restore zero-shot transfer; phase 4
showed transfer decays with depth. But phase 3c also found that **full activations** of
two same-behaviour nets align well (residual 0.65 vs 1.15 for weights). So: **if we hold
a small Q-A dataset of the model to port, can we extract its basis from that data — and
does it hold up as the net gets deep?**

## Setup (same-task, full transfer)

- `N` donor **transformers** all solve **one** shared task `tau=(sigma,k)` from
  **independent** random inits. Same behaviour, different residual-stream bases. Depth
  `L in {2,4,8}`.
- **"Q-A"** = a fixed batch of task inputs; we read each donor's activations on it.
- Put every donor into a reference donor's frame three ways, then train the
  cross-architecture translator `C` (transformer -> SSM) on train donors and test
  transfer (zero-shot + warm-start) on **held** donors:
  - `raw` — no alignment (control).
  - `weight` — orthogonal Procrustes on `tok+pos` weights (phase-3c method; only 48 rows
    -> underdetermined).
  - `data` — orthogonal Procrustes on **per-layer activations** over the Q-A probe (NEW).
- `tied` (all donors from one init) is the shared-frame **upper bound**.

**Why depth should help `data`, not hurt it.** The residual stream is a *single* `d`-dim
basis threaded through every block. Each extra layer just contributes more activation
vectors constraining the *same* alignment `Q`. `weight` stays stuck at 48 embedding rows
regardless of depth; `data` gets richer with depth.

`C` itself is unchanged across methods — we isolate the **basis** question. (With a
shared task the reference interior is already nearly right, so the hypernet's job is easy
and any transfer gap is attributable to the frame, not to generating deep interior.)

## Results

`V=16`, `d=64`, `ctx=32`, 12 train + 5 held donors, donor 200 steps, translator 300
steps. Chance = 6.25%. **zero-shot** task accuracy of `B*=C(A*)` on held donors (the
differentiator), and the frame residual achieved by each method's own `Q` (lower = the
two residual-stream frames are better aligned):

| L | raw | weight | **data** | tied (upper bd) | residual raw / weight / **data** |
|---|-----|--------|----------|-----------------|----------------------------------|
| 2 | 6.0% | 6.0% | **76.8%** | 100.0% | 1.42 / 1.31 / **0.64** |
| 4 | 6.2% | 7.5% | **94.4%** | 100.0% | 1.41 / 1.32 / **0.62** |
| 8 | 5.9% | 7.4% | **94.7%** | 100.0%* | 1.43 / 1.33 / **0.60** |

\* tied is the shared-frame upper bound; ~100% at every depth.

Warm-start saturates and does **not** separate the methods: with 15 fine-tune steps every
init (even `raw`) reaches ~99-100% — these tiny nets relearn the task from almost any
start. So **zero-shot** is the honest readout, and the gap there is stark.

## Takeaway

- **Static weights don't carry the basis; the Q-A *activations* do.** `weight` alignment
  (orthogonal Procrustes on 48 embedding rows) leaves zero-shot at chance — independent
  nets are not related by a `Q` you can read off the embeddings. `data` alignment, using
  activations on a tiny shared probe, recovers **77-95%** zero-shot.
- **Depth helps, exactly as predicted.** The residual stream is one `d`-dim basis shared
  across all layers, so deeper nets give *more* activation vectors to pin the same `Q`:
  `data` zero-shot rises 76.8% -> 94.4% -> 94.7% and its residual falls 0.64 -> 0.62 ->
  0.60 as `L` goes 2 -> 4 -> 8. `weight`/`raw` are flat at chance regardless of depth.
- So for the user's question — *can a small Q-A set recover the basis for deep nets?* —
  **yes, and it scales into depth** in this sandbox. The basis lives in the behaviour's
  activations, not the weights, and more depth means more evidence for it.

## Honest caveats

- `same-task` is the cleanest possible test of basis recovery. A real port also faces the
  **generation** problem (producing correct deep interior for the target architecture),
  which data alignment does **not** address — that is the next lever.
- `data` alignment is approximate (phase-3c residual ~0.65), so expect partial zero-shot
  plus warm-start finishing the job, not a free lunch.
- Toy scale (`V=16`, `d=64`). What is meant to port is the *mechanism* — activations on a
  shared behaviour expose a common basis, and depth aids the estimate — not the absolute
  numbers.

## Reproduce

```
python run_deep.py --depths 2,4,8 --n_train 12 --n_held 5 --donor_steps 200 \
                   --c_steps 300 --tps 6 --warm 15
```

Code: `align.py` (`layer_activations`, `data_align`, `method_Q`, `frame_residual`) and
`run_deep.py`.
