# AR → Diffusion port: adapt Qwen3.5-0.8B into a Mercury-like diffusion LM

This experiment ports an **autoregressive (AR) decoder-only transformer** into a
**discrete-diffusion language model** in the style of Inception Labs' *Mercury* —
the same Transformer backbone, but used as a **denoiser**: generation is
iterative **parallel denoising**, not left-to-right token-by-token decoding.

This is a *proven* path, and it is exactly our case: **Dream 7B** was initialized
from Qwen2.5-7B and adapted into a diffusion LM; **DiffuLLaMA / DiffuGPT**
(ICLR'25) convert an existing AR LLM into a diffusion LM with ≤200B tokens. So
the "translation" is **not** a weight remap — it is continued pre-training:
*same tensors*, we change (1) the attention mask, (2) the training objective, and
(3) how we sample.

## The recipe (three changes)

1. **Bidirectional attention with causal→full annealing.** AR uses a causal
   mask; a diffusion LM needs full (bidirectional) attention. We anneal from
   causal to full so early training preserves the pretrained behavior
   (DiffuLLaMA §3). Implemented here as a per-batch Bernoulli mixture with
   probability `ρ(step)` of staying causal.

2. **Absorbing-state masked-diffusion objective (LLaDA).** Add a `[MASK]`
   token. Forward process: sample `t ~ U(0,1]`, mask each token i.i.d. with
   probability `t`. The model predicts the originals at masked positions from
   bidirectional context. Loss is the LLaDA ELBO:

   ```
   L = E_t[ (1/t) · Σ_i 1[masked_i] · −log p(x0_i | x_t) / L ]
   ```

   The **1/t** weight (not present in BERT-style MLM) is what makes this an
   ELBO bound — see `diffusion.diffusion_loss`.

3. **Confidence-based parallel denoising sampler.** Start from an all-`[MASK]`
   sequence; over `T` steps, predict every masked token and **keep the
   highest-confidence** ≈`L/T` of them, re-masking the rest (LLaDA/Dream). The
   `[MASK]` logit is forced to −∞ so it is never emitted. Prompt conditioning =
   pin the prompt span, only the continuation is ever masked.

## Files

| File | Role | Runs where |
|---|---|---|
| `diffusion.py` | **Model-agnostic core**: `sample_mask_rate`, `forward_mask`, `diffusion_loss`, `diffusion_generate`, `set_bidirectional`, `anneal_causal_prob`. torch only. | CPU + Kaggle |
| `test_diffusion.py` | CPU unit tests proving the core (6 tests). | CPU (here) |
| `toy_adapt.py` | CPU driver: adapt the toy `LM` on char-Shakespeare; the **local verification** of the same recipe. | CPU (here) |
| `qwen_adapt.py` | Kaggle driver: Qwen3.5-0.8B + transformers/peft. | Kaggle |
| `qwen_kaggle.ipynb` | Notebook wrapping `qwen_adapt.py`. | Kaggle |

The Kaggle driver imports the **same** `diffusion.py` that the CPU tests
exercise — the recipe core is verified here; only the HF wiring is Kaggle-only.

## Verified on this CPU box

`python experiments/diffusion_port/test_diffusion.py` — all 6 pass:

- `forward_mask`: masked positions hold `[MASK]`, others unchanged, ≥1 mask/row,
  empirical mask fraction ≈ t.
- `bidirectional_toggle`: causal attention zeros the future; after
  `set_bidirectional(causal=False)` attention to future positions is non-zero —
  proving the one-line `models.py` edit takes effect.
- `loss_weighting`: halving `t` doubles the loss on a fixed mask (the 1/t weight).
- `sampler_validity`: output has the right shape, **no `[MASK]` remains**, ids in
  range, and a prompt prefix is preserved verbatim.
- `anneal_schedule`: `ρ(0)=1`, `ρ(end)=0`, linear in between.
- `loss_decreases`: a tiny LM trained with the diffusion loss on a structured
  (tiled-motif) task drops masked-CE ~94% and then denoises the motif back.

`python experiments/diffusion_port/toy_adapt.py` adapts the toy transformer on
char-level Shakespeare with the full recipe (annealing + masked-diffusion + a
clean held-out eval at fixed t) and prints denoised samples.

## Running on Kaggle (the real port)

On a Kaggle GPU notebook (T4×2 or P100, 16GB, network on):

```bash
pip install -q "transformers==5.8.*" "peft>=0.18" accelerate datasets safetensors
python qwen_adapt.py --model Qwen/Qwen3.5-0.8B --steps 2000 --bs 4 --seq_len 512
```

`qwen_adapt.py`:
- loads Qwen3.5-0.8B in **bf16** with `attn_implementation="eager"`,
- adds `<|mask|>`, `resize_token_embeddings`, mean-inits the new row,
- **forces bidirectional** attention by monkeypatching
  `transformers.masking_utils.create_causal_mask` and then **verifies** it
  (position-0 must become sensitive to a later token),
- attaches **LoRA** with `modules_to_save=["embed_tokens","lm_head"]` so the cold
  `[MASK]` embedding can learn; grad-checkpointing + `enable_input_require_grads`
  before `get_peft_model`,
- trains with the LLaDA loss + annealing, then denoises a prompt.

**Memory budget (single T4 16GB):** weights 1.6GB + LoRA/optimizer/embed ~1GB +
activations w/ checkpointing ~3GB ≈ **6–7GB**.

**Cheap smoke test before downloading Qwen:** the repo ships a real 50M
`LlamaForCausalLM` at `experiments/shakespeare_port/models/supra50m` (same
RMSNorm/RoPE/GQA/SwiGLU family). Point the driver at it to validate the whole
pipeline first:

```bash
python qwen_adapt.py --model ../shakespeare_port/models/supra50m --steps 200
```

## Honest caveats

- **Bidirectional mask can silently fail with fused kernels.** FlashAttention-2
  honors only an `is_causal` bool and ignores arbitrary 4D masks, so a
  "bidirectional" mask has no effect under FA2. We force `eager`/`sdpa` and add a
  runtime `verify_bidirectional` check before training.
- **transformers mask API drifts.** v5 builds masks via
  `masking_utils.create_causal_mask`; the old `AttentionMaskConverter` is
  deprecated. We pin `transformers==5.8.*` and keep the monkeypatch defensive.
- **Cold-start `[MASK]` embedding.** The new row is random; if frozen by LoRA it
  never learns. We mean-init it and put `embed_tokens` in `modules_to_save`.
- **Tiny token budget.** Dream used ~580B tokens. A few thousand LoRA steps on a
  small corpus yields a *weak* diffusion LM — expect locally-coherent, globally
  loose samples. Evaluation here is **qualitative**.
- **Annealing is simplified** (per-sequence Bernoulli vs DiffuLLaMA's
  per-position right-context sampling) — same end state, different transient.
- **Qwen3.5 may be a multimodal wrapper.** If so, the text decoder is nested
  (`model.model.language_model`); target it for the mask override + LoRA, or fall
  back to `supra50m` for the smoke test.
- **The Qwen path is not runnable on the dev box** (CPU-only, no transformers, HF
  blocked). Only the model-agnostic core is verified here; the transformers
  wiring is verified on Kaggle.

## References

- LLaDA — Nie et al., [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
- DiffuLLaMA / scaling diffusion LMs from AR — Gong et al., [arXiv:2410.17891](https://arxiv.org/abs/2410.17891)
- Dream 7B — [arXiv:2508.15487](https://arxiv.org/abs/2508.15487), [DreamLM/Dream](https://github.com/DreamLM/Dream)
- nanoLLaDA (CE/t = ELBO reference) — [Lukas-Xue/nanoLLaDA](https://github.com/Lukas-Xue/nanoLLaDA)
