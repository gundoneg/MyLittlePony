# Phase 6 — scaling the weight-translator `C` with depth

**Goal.** The monolithic `Translator` (`xlate.Translator`) generates the target SSM's
deep interior from a single layer-0 code through one big matrix `M`, so its
parameters are sized to a fixed depth and its zero-shot transfer **decayed with
depth** (phase-4: 84% @L1 → 76% @L2 → 67% @L4). `PerLayerTranslator` instead emits
each block `l` from a per-block feature of donor block `l`, through a **shared**
encoder and a **shared** `M_block` reused at every depth — so its parameter count is
independent of depth, and it can in principle be trained shallow and applied deep.

This phase asks: **does the per-block weight-tied translator actually scale with
depth better than the monolith?**

## The load-bearing finding: the old testbed was degenerate

Before trusting any depth-scaling number we ran a **layer-ablation control**
(neutralise block `l` in a donor, measure the task-accuracy drop). On the original
`(σ,k)` task — predict `σ(x_{t−k})` — the result was damning:

```
(σ,k), L=4:  per-layer ablation drop  [1%, 1%, 0%, 0%]
```

`(σ,k)` is a **single gather**: one attention layer attends `k` steps back at any
lag, so deeper donors solve it in layer 0–1 and leave their deep blocks **idle**.
Idle blocks have arbitrary weights → there is nothing to translate → zero-shot
necessarily collapses to chance at depth. **The depth decay was a property of the
task, not a failure of the translator.** No translator design can be fairly judged
on a benchmark where the deep weights carry no computation.

## The fix: a depth-demanding task (pointer-chasing)

We added a backward **pointer-chasing** task (`task.sample_tasks_hops`): each
position holds a small backward jump `j_i = (x_i mod J)+1`; the target follows the
pointer chain `m = depth` times, `y_t = σ(x[p_m(t)])`. Following `m` pointers needs
`~m` sequential gathers, i.e. `~m` layers (one attention layer = one hop). `σ` is
still visible in the embedding/head, so the translator's vocab-equivariant part
stays meaningful.

Calibration (`calib_hops.py`, `maxjump=2`) confirms depth is now genuinely used:

```
(σ,k)  L=4          donor 100% | ablation [1%, 1%, 0%, 0%]   <- idle deep blocks
hops   L=2 hops=2   donor 100% | ablation [50%, 88%]         <- every layer works
hops   L=3 hops=3   donor 100% | ablation [52%, 35%, 83%]    <- every layer works
hops   L=4 hops=4   donor 100% | ablation [64%, 44%, 4%, 4%] <- caps at ~2 used layers
```

At `d_model=64`, donors fully exercise 2–3 layers; at L=4 they find a 2-layer
shortcut for 4 short hops, so the two deepest blocks go idle again. **The usable
depth window at this toy width is L≈2–3** — that is where a depth-scaling claim can
be tested cleanly.

## EXP 1 — isolation: mono vs perlayer on the fixed testbed

Tied-init donors, different `(σ,hops)` per donor, `n_train=10`, `n_held=4`,
`c_steps=350`. Metrics: zero-shot, warm@15 (translate + 15 fine-tune steps), and
steps-to-90% (fine-tune steps to reach 90% accuracy). Chance = 6.25%.

| hops | donor | **mono** zs / warm / s90 | **perlayer** zs / warm / s90 | ablation |
|---|---|---|---|---|
| L=2 | 100% | 14.4% / 86.6% / 20 | **23.3% / 99.7% / 8** | `[50,88]` depth used |
| L=3 | 100% | 13.6% / 81.4% / 18 | 9.0% / **96.2% / 12** | `[53,33,84]` depth used |
| L=4 | 100% | **48.6%** / 97.7% / 10 | 33.9% / **99.3%** / 10 | `[63,44,5,4]` deep idle |

**Reading it honestly:**

- **Where depth genuinely matters (L=2, L=3), the per-block translator wins on
  practice:** warm-start 96–99.7% vs 81–87%, and 1.5–2.5× fewer steps to 90%. Its
  depth-agnostic, weight-tied structure pays off exactly when there is real per-layer
  computation to reproduce.
- **Pure zero-shot is noisy** at this toy scale: perlayer takes L=2, mono edges L=3.
  The warm-start / convergence advantage is the robust signal; raw zero-shot on a
  16-token vocab with 10 donors is high-variance.
- **L=4 is a confound, not a mono win.** The donor solved 4 hops with ~2 layers
  (`ablation [63,44,5,4]`), so the task partially re-degenerated; zero-shot jumped for
  both and the monolith looks better only because the effective depth dropped back to
  ~2. It is the same idle-deep-block artifact, re-confirming why the control matters.

## EXP 2 — depth-transfer (the unique capability)

Train **one** `PerLayerTranslator` on shallow (L=2) donors, apply **zero-shot to
deeper** held donors. The monolith *cannot do this at all* — its `M` has shape
`interior(L) × d_z`, undefined off its training depth — so this is the per-block
design's unique payoff. `depth_frac` conditions each block on a normalised depth
fraction so middle layers of a deeper net interpolate between the training-depth
endpoints (absolute layer index would not generalise).

_(results appended below when the run completes)_

## Honest caveats

- **Toy width caps usable depth at L≈2–3.** The clean depth-scaling window is small;
  `L=4+` re-degenerates at `d_model=64`. The *mechanism* (a shared, depth-agnostic
  per-block translator) is what is portable, not these absolute numbers — and pushing
  past 3 working layers needs more width/capacity, not a different translator.
- **Per-layer correspondence assumption.** PerLayerTranslator assumes donor block `l`
  maps to target block `l`. The layer-ablation control shows the blocks do carry
  computation, but block-to-block alignment across architectures is still an
  assumption, not a proof.
- **Zero-shot variance.** With V=16 and ~10 donors, raw zero-shot swings; warm@15 and
  steps-to-90 are the stable comparisons.
- This is the bridge to real models: it tests whether a *weight* translator scales in
  depth on a task where depth is load-bearing — the prerequisite before applying `C`
  to a genuinely deep pretrained model.
