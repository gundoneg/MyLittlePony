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

(filled in after the runs — donor vs target validation perplexity and side-by-side
samples.)

## Honest caveats

- These are *tiny* from-scratch models (~4–5M params); output is Shakespeare-ish,
  not coherent prose.
- This port uses **teacher inference** (distillation) — a standard regime, and
  deliberately *different* from the zero-shot, no-inference weight translation in
  the numpy lab. What carries over is the gauge-free `emb`/`unemb` bridge and the
  conv-augmented SSM mixer; the interior is fit by distillation rather than a
  learned weight map.
