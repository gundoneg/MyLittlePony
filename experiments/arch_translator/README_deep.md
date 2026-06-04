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

<!-- RESULTS -->
_(filled in from `run_deep.log` once the sweep completes)_

## Takeaway

<!-- TAKEAWAY -->

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
