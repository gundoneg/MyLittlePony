# Is the cipher σ readable from a model's *static weights*? — a controlled titration

**Question.** We build a "zoo" of tiny donors that are identical *except* for a
secret vocabulary permutation (a cipher σ_i): each donor is a Transformer trained
on text whose 65 characters have been relabelled by its own σ_i. If a downstream
translator is ever going to map one donor's weights onto another's, it must be able
to read σ_i back out of the **static weights alone** — no activations, no data.

An earlier run concluded flatly *"cipher is NOT recoverable from static weights."*
That was true for the zoo as built, but the diagnosis was incomplete. This
experiment turns the question into a **controlled titration**: hold everything
fixed, dial divergence between donors up from zero, and map exactly where σ stops
being weights-readable — and why.

## TL;DR

- **At zero divergence, σ is read out of raw weights at 100%.** The thesis "weights
  carry σ" is *true* — when donors share a per-character init anchor, a donor is
  literally the canonical donor with its vocab rows permuted, and a cosine match of
  embedding rows recovers σ exactly.
- **The killer is the init anchor, not the data.** Whether each character starts
  from a *shared* base vector across donors is the single variable that decides
  readability. Batch/data-order divergence (`div`) is **a near-total non-factor**.
- **There is a sharp cliff.** Recovery stays at 100% until the (cipher-free)
  interior divergence reaches ≈0.84, then collapses to chance (1/65 ≈ 1.5%) by
  ≈0.96. No gauge-alignment or graph-matching method recovers σ past the cliff.
- The original "not recoverable" zoo used a **tied (non-anchored) init** — i.e. it
  sat *below* the cliff by construction. The negative result was real but
  regime-specific; this maps the whole regime.

## Setup

- **Donors.** `models.py` TinyTransformer A (d=96, 2 layers, 3 heads, ctx=64),
  vocab=65 (tinyshakespeare chars). Each donor i gets cipher σ_i = `randperm(65)`
  and trains on σ_i-relabelled text. Donor 0 is the canonical reference.
- **Recovery target.** For donor-i row j, the true relative map is
  `σ̂(j) = perm₀[ inv(perm_i)[j] ]` (the donor-0 row holding the same character).
  Accuracy = fraction of the 65 indices mapped correctly; chance = 1/65 = 1.5%.
- **Two divergence knobs (`zoo.py`).**
  - `align` ∈ [0,1] — the **init anchor**. `align=1`: char-aligned, so character
    *c* starts from the same base vector `base[c]` in every donor (placed at row
    σ_i(c)). `align=0`: tied init, where row *j* always holds `base[j]`, so
    character *c* starts from `base[σ_i(c)]` — a *different* vector per donor.
    Intermediate values blend the two (scale-preserved).
  - `div` ∈ [0,1] — fraction of training steps on which a donor draws a **private**
    batch order; on the rest, all donors draw from a **common** stream (same char
    indices, then relabelled by their own σ_i). `div=0` = identical data exposure.
- **The divergence ruler.** Interior matrices (`qkv`, `o`, `mlp.fc`, `mlp.proj`)
  have *no vocabulary axis*, so the cipher cannot touch them. The mean relative L2
  distance of donor i's interior to donor 0's is a clean, method-agnostic measure
  of "real" divergence (0 = identical, ≈1.41 = independent random). It is the
  x-axis of the cliff.

## Recovery methods (`recover.py`, all weights-only)

| | method | idea |
|---|---|---|
| **M0** | emb-cosine | optimal (Hungarian) match of `tok` rows by cosine. Baseline. |
| **M1** | gauge-fixed | first remove the hidden-channel gauge between donors (orthogonal Procrustes on the cipher-free interior matrices), then match `tok` rows. |
| **M2** | graph-QAP | objective-guarded Frank-Wolfe on the behaviour matrix `G = tok·headᵀ` (gauge-invariant), refining the M0 guess. Never degrades its warm start. |

Hungarian assignment is a pure-numpy solver (`hungarian.py`) — scipy is unavailable
in this environment.

## Results (`python titrate.py --n 8 --steps 250`)

### Table A — the limit curve: sweep the init anchor (data order shared)

```
       setting  int_div |  M0 emb  M1 gauge  M2 graph
    align=1.00    0.000 |  100.0%    100.0%    100.0%
    align=0.85    0.447 |  100.0%    100.0%    100.0%
    align=0.70    0.718 |  100.0%    100.0%    100.0%
    align=0.60    0.841 |  100.0%    100.0%    100.0%
    align=0.50    0.919 |   31.0%     50.1%     20.9%   <- the cliff
    align=0.30    0.964 |    1.3%      1.3%      1.3%
    align=0.00    0.972 |    1.3%      1.3%      1.3%
```

```
  M0 sigma-recovery  vs  interior-divergence
  int_div     M0  0%                                    100%
    0.000   100%  |########################################|
    0.447   100%  |########################################|
    0.718   100%  |########################################|
    0.841   100%  |########################################|
    0.919    31%  |############............................|
    0.964     1%  |#.......................................|
    0.972     1%  |#.......................................|
```

The transition is razor-sharp: **100% up to interior_div ≈ 0.84, chance by ≈ 0.96.**
Right at the edge (align=0.50) the gauge-fix M1 gives a genuine lift (50% vs M0 31%),
but it cannot carry you across — once interiors diverge past the cliff, the embedding
rows are no longer a clean permutation and the signal is gone.

### Table B — data order is a non-factor

```
       setting  int_div |  M0 emb  M1 gauge  M2 graph
align=1 div=0.0    0.000 |  100.0%    100.0%    100.0%
align=1 div=0.5    0.373 |  100.0%    100.0%    100.0%
align=1 div=1.0    0.480 |  100.0%    100.0%    100.0%
align=0 div=0.0    0.972 |    1.3%      1.3%      1.3%
align=0 div=0.5    0.980 |    1.3%      1.3%      1.3%
align=0 div=1.0    0.973 |    1.3%      1.3%      1.3%
```

Dialling the data order from fully shared to fully independent moves recovery
**not at all**, at either end. The anchor decides everything; the data order is
noise on top.

### Table C — the natural regimes (reproducing "NOT recoverable")

```
       setting  int_div |  M0 emb  M1 gauge  M2 graph
     tied init    0.973 |    1.3%      1.3%      1.3%
independent init  1.416 |    1.1%      2.6%      2.6%
```

Both natural inits (tied or fully independent) land *below the cliff* and σ is at
chance — exactly the earlier negative result, now explained: the original zoo used a
tied init, so the cipher offset every donor's embedding lookups from step 0, the
interior diverged to ≈1.0, and the embedding rows stopped being a permutation.

### Closing the loop — does a weights-read σ give a *working* model? (`translate_cipher.py`)

The whole point of recovering σ is to *use* it. The translator C is trivial **given**
σ: take the canonical distilled SSM target B₀ and relabel its vocab rows by the σ
read from A_i's weights — `B̂_i.tok[j] = B₀.tok[σ̂(j)]`, interior copied. We then
compare B̂_i to the **true** B_i (distilled from A_i directly) on σ_i-relabelled
validation text. `naive` = using B₀ with no relabel.

```
=== align=1.0  [char-aligned anchor] ===
pair sigma_acc |  B_hat CE  true CE  naive CE | agree(hat,true)  agree(naive)
MEAN    100.0% |     2.150    2.150     6.997 |           91.1%          0.5%

=== align=0.0  [tied / natural] ===
MEAN      1.5% |     6.889    2.150     6.889 |            0.8%          0.8%
```

At the anchor, the relabelled model **matches the true target** (CE 2.150 = the
ceiling; 91% top-1 agreement) while the naive copy is useless (CE 7.0, 0.5%). At the
tied init, σ is at chance, so the "translated" model is **byte-for-byte as good as
naive** (CE 6.889 = naive 6.889, 0.8% agreement). The translator works exactly where,
and only where, σ is weights-recoverable.

## Why the anchor, mechanistically

With a non-anchored init the cipher acts *before the first forward pass*: donor i
reads character *c* from embedding row σ_i(c), which (without char-alignment) holds
an unrelated vector. So from step 0 the interior of each donor consumes a different
stream of vectors → the cipher-free interior itself diverges to near-random
(int_div ≈ 0.97 even with **identical** data, Table B bottom). The permutation
structure the recovery methods look for never exists in the trained weights. The
char-aligned anchor is precisely what keeps "character *c*'s vector" donor-invariant,
so the only inter-donor difference is a row permutation — recoverable exactly.

## Phase 2 — squeezing the cliff

Phase 1 found a sharp cliff. Phase 2 asks the three follow-ups: **where** exactly is
it (and is it a genuine phase transition in divergence?), can we **move** it (smarter
init/regularisation, or a stronger reader?), and how does it depend on **network
size**? Drivers: `cliff.py`, `anchor_reg.py`, `scale.py` (n=8, steps=250 unless noted).

### A — where the cliff is, and is it mechanism-independent? (`cliff.py`)

Two *independent* ways to inject divergence on top of the cipher: weaken the init
anchor (`align`), or keep `align=1` but add donor-specific init `noise`. Reported
against the method-agnostic `interior_div` axis, with the interpolated M1 50%-crossing.

```
MECHANISM 1 — weaken the init anchor (align)        MECHANISM 2 — init noise (align=1)
 align  int_div    M0    M1    M3                     noise  int_div    M0    M1
  0.65    0.784  100%  100%  100%                       1.0    0.798  100%  100%
  0.60    0.841  100%  100%  100%                       2.0    0.810   37%   35%
  0.575   0.865  100%  100%  100%                       2.5    0.827   20%   21%
  0.55    0.887   97%  100%  100%                       3.0    0.848   12%   15%
  0.50    0.919   31%   50%   45%   <- crossing          4.0    0.881    6%    6%
  0.45    0.941    2%    8%    2%                         6.0    0.933    3%    3%
 M1 50%-crossing @ interior_div = 0.919              M1 50%-crossing @ 0.807
```

Both mechanisms show the **same razor-sharp cliff** (100% → chance within ~0.05 of
`interior_div`), in the same neighbourhood. But the coordinates differ by ≈0.11
(align 0.92 vs noise 0.81): **`interior_div` is a strong but imperfect predictor** —
the *kind* of divergence shifts the threshold somewhat (per-row init noise corrupts
the embedding-permutation structure a little faster than the coherent anchor blend).
So: a real phase-transition-like boundary, located at `interior_div ≈ 0.8–0.92`, not
a single universal constant.

### B — can we move the cliff? Two levers, both null

**B-1, inject the anchor during training** (`anchor_reg.py`): start from the tied
init (align=0) and add `lam·‖tok[σ(c)] − ref[c]‖²` pulling each char toward a shared
reference.

```
   lam   int_div  A_ce    M0    M1
   0.0    0.972   2.14   1.3%  1.3%
  10.0    0.989   2.16   1.3%  1.3%
 100.0    1.011   2.19   1.3%  1.3%
1000.0    1.031   2.26   1.3%  1.3%
```

Recovery stays at chance all the way to `lam=1000`, while CE *worsens* and divergence
*grows*. A weak injected anchor cannot undo the cipher's row-permutation within the
training budget; you would have to force the embeddings all the way back to the shared
anchor — i.e. just re-impose `align=1`. The cliff does not move.

**B-2, a stronger reader** (`recover.py` M3 = alternating gauge↔row refinement +
multi-donor consensus): M3 50%-crossing **0.916 vs M1 0.919** — identical, marginally
worse at the edge (consensus through bridge donors that are *themselves* at the cliff
compounds noise). **M1's single gauge-fix already marks the ceiling** of weights-only
reading; a cleverer reader buys nothing.

### C — the cliff vs network size (`scale.py`, n=6 steps=200)

```
 d_model  n_layer   cliff interior_div (M1 50%)
    48       2              0.851
    96       2              0.876
   192       2              0.916      width  ↑  => cliff moves RIGHT
    96       1              0.926
    96       2              0.876
    96       4              0.813      depth  ↑  => cliff moves LEFT
```

- **Width helps:** wider models (48→96→192) read σ at *higher* divergence — more
  dimensions make embedding rows more separable and the hidden gauge better determined.
- **Depth hurts:** more layers (1→2→4) push the cliff *lower*. This **refutes** the
  naive hypothesis that more interior gives the gauge-fix more to lock onto — deeper
  donors evidently diverge in richer, less gauge-like ways, breaking readability sooner.
- (Absolute numbers shift with the lighter n=6/steps=200 budget; the **trends** —
  width→right, depth→left — are the portable result.)

### Phase-2 takeaway

The cliff is **robust**: it is a sharp, phase-transition-like boundary in divergence
(`interior_div ≈ 0.8–0.92`), it is **not movable** by injected anchoring or by a
stronger reader, and it shifts predictably with architecture (wider = more tolerant,
deeper = less). Below it σ is exactly readable; above it the information is simply not
present in the static weights, full stop.

## Honest caveats

- `align=1` is an **idealisation**: it assumes donors were initialised from a shared
  per-character basis. Real independently-trained model zoos are the `align≈0`
  regime, where this experiment says σ is *not* recoverable from static weights by
  any of M0/M1/M2. The value here is the **slope**: it pinpoints what would have to
  be true (a shared init anchor) for weights to carry σ, and shows the transition is
  a cliff, not a gentle decay.
- M2 (graph-matching on `G`) never beats the embedding match — the behaviour matrix
  carries no *extra* recoverable signal beyond the embeddings, and degrades in
  lock-step with them. (Consistent with an oracle check: substituting the *true* σ
  into `G_i ≈ P·G₀·Pᵀ` already gives a large residual once divergence is real.)
- This is a 65-token, 2-layer sandbox. The cliff location (`interior_div ≈ 0.8–0.92`)
  is not a universal constant — it shifts with the divergence mechanism (Phase 2-A) and
  with width/depth (Phase 2-C). The portable results are the qualitative shapes: anchor
  decides, data doesn't, the transition is sharp, and it is not movable by reader or
  regulariser.

## Reproduce

```bash
cd experiments/arch_translator
python titrate.py --n 8 --steps 250      # Phase 1: Tables A/B/C + cliff curve
python translate_cipher.py --n 4         # Phase 1: close the translator loop
python cliff.py --n 8 --steps 250        # Phase 2-A: locate cliff + mechanism check + M3
python anchor_reg.py --n 8 --steps 250   # Phase 2-B1: anchor-reg lam sweep (null)
python scale.py --n 6 --steps 200        # Phase 2-C: cliff vs width/depth
python hungarian.py                      # solver self-test
```
