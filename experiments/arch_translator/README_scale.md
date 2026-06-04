# Phase 4 — scaling the cross-architecture translator

**Question.** Phase 3 showed zero-shot transfer works and warm-start (translate + a few
steps) reaches ~97%. How does this **scale** along donor count, width, depth, target
architecture, and number of tasks packed into one model?

**Setup.** All sweeps in the **tied** regime (shared frame — clean measure of the
method's capacity; the independent-frame penalty is characterised in phase 3b/3c). One
axis at a time from a baseline `Cfg(vocab=16, d_model=64, n_layer=1, n_head=4, ctx=32,
d_ff=256)`, n_train=20, n_held=6, donor 200 steps, translator 400 steps. Metrics:
zero-shot, warm@15, and **steps-to-90%** = fine-tune steps for B to reach 90% accuracy,
for the translated init vs a random init (cap 32). Chance = 6.2%. (`scale_study.py`)

## Results

```
count (number of donor models)
  n_train     donor  zeroshot  warm@15   st90>tr   st90>rnd
  8          100.0%     25.4%    89.3%        17        >32
  16         100.0%     73.0%   100.0%         5        >32
  32         100.0%     94.7%   100.0%         1        >32
  48         100.0%     99.6%   100.0%         0        >32

width (d_model)
  32         100.0%     55.2%   100.0%        11        >32
  64         100.0%     84.2%   100.0%         3        >32
  128        100.0%     97.1%   100.0%         0         25

depth (n_layer)
  1          100.0%     84.2%   100.0%         3        >32
  2          100.0%     76.0%   100.0%         4         31
  4          100.0%     66.5%   100.0%         5         25

arch (target architecture B)
  B=ssm      100.0%     84.2%   100.0%         3        >32
  B=xformer  100.0%     46.9%    52.9%        27        >32

mtask (tasks solved by ONE model, in-context selector)
  M=1        100.0%     82.2%   100.0%         4        >32
  M=2         99.8%     19.7%    51.6%        >32        >32
  M=4         97.3%      9.4%    33.2%        >32        >32
```

## Takeaways

- **More donors is a clean scaling law (the big win).** Zero-shot 25→73→95→**99.6%** and
  steps-to-90% **17→5→1→0** as the zoo grows 8→48. At 48 donors the translated B is
  already ≥90% with *no* fine-tuning. Random init never reaches 90% in 32 steps. The
  translator learns a genuine, sharpening rule from more examples.
- **Wider is better, monotonically.** d_model 32→128: zero-shot 55→**97%**, steps 11→0.
  More dimensions make the structure easier to read and write.
- **Deeper is harder (honest).** 1→4 layers: zero-shot 84→**67%**, steps 3→5. Translating
  more stacked computation degrades zero-shot (echoes the phase-2 depth finding). Warm@15
  still recovers to 100%.
- **Translating *into* an SSM is easier than into a transformer.** B=ssm 84% vs
  B=transformer 47% zero-shot (warm@15 100% vs 53%). Generating attention interior with
  the hypernet is harder than the SSM's simple diagonal recurrence — a target-architecture
  effect worth remembering.
- **Multi-task per model is the wall.** A single donor learns M tasks fine (97–100%), but
  the translator degrades sharply with M: zero-shot 82→20→9% and even warm@15 100→52→33%
  (never hitting 90% in 32 steps for M≥2). The in-context conditional logic (selector →
  which σ) is encoded in a way the equivariant vocab maps + simple hypernet cannot
  reproduce. This is the clearest frontier for future work.

Across every axis: donor accuracy is ~100% (the task is learnable) and a random init
essentially never reaches 90% within 32 steps — so the translation front-loads almost all
the learning wherever it works at all.

## Honest caveats

- Toy scale (V=16, d≤128, L≤4); the **trends** are the result, not the absolute numbers.
- Tied regime = upper bound (shared frame). Independent+aligned warm-start sits a little
  below (phase 3b); these scaling shapes should carry, shifted down.
- steps-to-90% is capped at 32 and means are over 6 held donors; ">32" = not reached.
- The multi-task selector mechanism is deliberately simple; a richer translator (e.g. one
  that reads conditional structure, not just a single vocab map) is the natural next step.

## Reproduce

```bash
cd experiments/arch_translator
for ax in count width depth arch mtask; do python scale_study.py --axis $ax; done
```
