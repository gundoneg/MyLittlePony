# Shakespeare port: Transformer → SSM

A small, readable end-to-end demo of the cross-architecture transfer idea from
`experiments/weight_translator/`, but on a **real (tiny) language model you can
actually talk to**. We train a GPT-style Transformer from scratch on
TinyShakespeare, then **port it into a state-space model (SSM)** and sample from
both.

Why from scratch and why torch: the container has no GPU and HuggingFace is
blocked, so there is no pretrained model to download — we train our own donor.
We use PyTorch (CPU) rather than the numpy autograd lab because a readable LM
needs more compute than numpy-on-CPU can give in reasonable time.

## The method (two reusable pieces)

1. **Gauge-free bridge.** Donor and target share vocabulary and `d_model`, so the
   token embedding, positional embedding, and `lm_head` (unembedding) transfer
   **verbatim** — and we *freeze* them in the target. This is the structured
   `emb`/`unemb` transfer validated in the numpy lab: the shared vocab
   representation is the architecture-agnostic bridge.
2. **Distillation through the bridge.** Only the SSM *interior* (mixers, FFNs,
   norms) is trained, to match the donor's next-token distribution
   (KL on logits + a little CE), reading and writing through the frozen shared
   embedding space. The SSM learns to reproduce the Transformer's function in
   the *same* representation.

The SSM mixer is the transfer-friendly design from iteration 12: a short causal
depthwise conv followed by a **diagonal LTI state-space model**. Because the SSM
is linear-time-invariant, its action is a causal convolution with a per-channel
impulse kernel `K[τ] = c·aᵀ·b` — so a sequence is processed with a single
`conv1d`, no Python time loop (fast on CPU). Verified exactly equal to the
explicit recurrence.

## Files

| file | role |
|------|------|
| `data.py` | downloads TinyShakespeare, trains a small (~2048) BPE, serves windowed batches |
| `donor.py` | GPT-style causal Transformer decoder |
| `target.py` | SSM (Mamba-lite, LTI) causal decoder, same emb/pos/head layout |
| `train_donor.py` | trains the donor from scratch → `donor.pt` |
| `transfer.py` | copies the gauge-free bridge + distills the SSM interior → `target.pt` |
| `chat.py` | sample from donor and/or target (one-shot `--prompt` or interactive REPL) |

## Run

```bash
pip install torch
python data.py                 # build tokenizer, check round-trip
python train_donor.py          # ~30-40 min CPU; saves donor.pt
python transfer.py             # distill SSM; saves target.pt
python chat.py --prompt "ROMEO:"   # compare both models
```

## Results

Both models are ~4–5M params, `d_model=256`, 4 layers, `ctx=128`, vocab 2048,
trained on CPU (4 threads).

| model | how it was made | val loss | val **perplexity** |
|-------|-----------------|---------:|-------------------:|
| donor (Transformer) | trained from scratch, 3500 steps (~35 min) | 3.078 | **21.7** |
| target (SSM)        | gauge-free bridge copied + interior distilled, 3000 steps | 3.067 | **21.5** |

The ported SSM **matches the Transformer it came from** (21.5 vs 21.7) — while its
token/positional embeddings and unembedding were *frozen copies* of the donor's
and only the SSM interior was trained, purely to imitate the donor's output
distribution. During distillation the SSM crossed donor-level perplexity by
~step 1000 and held there.

### Side-by-side samples (`temp 0.8, top_k 40`)

**Prompt `ROMEO:`**

```
donor (Transformer):           target (SSM):
ROMEO:                         ROMEO:
That thou hast thou canst not  And then a day for my wife's
say that thou fled;            good for her daughter:
But wilt thou shalt not know   I am I am a word, sir.
thou hast it that;
And I see, I am sure of love.  DUKE VINCENTIO:
Thou art a son thy head, ere   I pray you, sir, is not now.
thou shalt tell thee;
```

**Prompt `To be, or not to be`**

```
donor (Transformer):                    target (SSM):
To be, or not to be unconquant withal.  To be, or not to be a buckl of thy eye,
Show you may I do this contrary, we     'O boutray thy unfter blood of my son;
will To do what we do we and leave it   And that will I do it from my heart
down.                                   Hath ever so borne to make him offend.
MONTAGUE:                               GLOUCESTER:
I hear the king, that I do confess.     Why, then I am not the king of Richard's
```

Both produce recognizable (if not coherent) Shakespeare with character cues,
verse line breaks, and period diction — at near-identical perplexity.

## Honest caveats

- These are *tiny* from-scratch models (~4–5M params); output is Shakespeare-ish,
  not coherent prose.
- This port uses **teacher inference** (distillation) — a standard regime, and
  deliberately *different* from the zero-shot, no-inference weight translation in
  the numpy lab. What carries over is the gauge-free `emb`/`unemb` bridge and the
  conv-augmented SSM mixer; the interior is fit by distillation rather than a
  learned weight map.
