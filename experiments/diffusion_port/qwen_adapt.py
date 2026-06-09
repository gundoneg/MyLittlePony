"""Kaggle driver: adapt Qwen3.5-0.8B (autoregressive) into a Mercury-like
discrete-diffusion LM, reusing the model-agnostic core in `diffusion.py`.

This file is meant to run on **Kaggle** (GPU + network + pip), NOT on the
CPU/offline box where the core was unit-tested. The recipe is identical to
`toy_adapt.py`; only the model wiring differs:

  1. load Qwen3.5-0.8B in bf16 with attn_implementation="eager",
  2. add a [MASK] token and resize embeddings (new row trains),
  3. force bidirectional attention (monkeypatch the causal-mask builder) and
     VERIFY it took effect before training,
  4. attach LoRA (with embed_tokens / lm_head in modules_to_save so the cold
     [MASK] embedding can learn), grad-checkpointing,
  5. train with the LLaDA 1/t masked-diffusion objective + causal->full anneal,
  6. generate with the confidence-based parallel denoiser.

Pins (match the in-repo supra50m checkpoint): transformers==5.8.*, peft>=0.18.

Smoke test before downloading Qwen: pass --model <path-to>/shakespeare_port/
models/supra50m (a real 50M LlamaForCausalLM, same RMSNorm/RoPE/GQA/SwiGLU
family) to validate the whole pipeline cheaply.

Usage on Kaggle:
    pip install -q "transformers==5.8.*" "peft>=0.18" accelerate datasets safetensors
    python qwen_adapt.py --model Qwen/Qwen3.5-0.8B --steps 2000
"""
import argparse
import sys

import torch

# diffusion.py lives next to this file; on Kaggle, add its dir to sys.path or
# copy it alongside. The import is intentionally bare so it works in a notebook.
try:
    import diffusion as D
except ImportError:  # running from repo root
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import diffusion as D


# --------------------------------------------------------------------------
# Bidirectional attention
# --------------------------------------------------------------------------
def force_bidirectional(model, *, anneal_state=None):
    """Monkeypatch the transformers causal-mask builder so the decoder attends
    in BOTH directions (still respecting padding). Returns a restore() fn.

    Why a monkeypatch and not just a flag: fused attention kernels
    (FlashAttention-2) ignore arbitrary 4D masks and only honor an `is_causal`
    bool, so a "bidirectional" 4D mask silently has no effect with FA2. We
    therefore (a) require eager/sdpa, and (b) replace the mask builder itself,
    which both backends consult.

    `anneal_state`, if given, is a dict with key "causal_prob" in [0,1]; on each
    call we keep the original causal mask with that probability (DiffuLLaMA-style
    annealing) and otherwise return the bidirectional (padding-only) mask.
    """
    import transformers.masking_utils as mu

    orig = mu.create_causal_mask

    def bidir(*args, **kwargs):
        # Annealing: sometimes fall back to the genuine causal mask.
        if anneal_state is not None:
            import random
            if random.random() < anneal_state.get("causal_prob", 0.0):
                return orig(*args, **kwargs)
        # Recover (config, input_embeds, attention_mask) robustly across the
        # positional/keyword call conventions transformers uses internally.
        attention_mask = kwargs.get("attention_mask")
        embeds = kwargs.get("input_embeds", kwargs.get("inputs_embeds"))
        if embeds is None and len(args) >= 2:
            embeds = args[1]
        if attention_mask is None and len(args) >= 3:
            attention_mask = args[2]
        if attention_mask is None:
            return None                       # full attention, no padding
        dtype = embeds.dtype if embeds is not None else torch.float32
        m = attention_mask[:, None, None, :].to(dtype)        # (B,1,1,L)
        return (1.0 - m) * torch.finfo(dtype).min             # 0 keep / -inf pad

    mu.create_causal_mask = bidir
    # Some model classes also cache a bound reference; patch the module attr is
    # enough for v5 decoders that call masking_utils.create_causal_mask.
    return lambda: setattr(mu, "create_causal_mask", orig)


@torch.no_grad()
def verify_bidirectional(model, tok, device):
    """Sanity check that attention truly flows from a later token to an earlier
    one. Compares the hidden state at position 0 with and without changing a
    LATER token; under causal attention pos-0 is invariant, under bidirectional
    it must change."""
    ids = tok("the quick brown fox", return_tensors="pt").input_ids.to(device)
    if ids.shape[1] < 3:
        print("  [verify] sequence too short, skipping")
        return
    h0 = model(ids, output_hidden_states=True).hidden_states[-1][0, 0]
    ids2 = ids.clone()
    ids2[0, -1] = (ids2[0, -1] + 1) % model.config.vocab_size   # perturb LAST token
    h1 = model(ids2, output_hidden_states=True).hidden_states[-1][0, 0]
    delta = (h0 - h1).abs().max().item()
    print(f"  [verify] pos-0 sensitivity to last token: {delta:.4e}  "
          f"({'BIDIRECTIONAL ok' if delta > 1e-4 else 'STILL CAUSAL -- check backend!'})")


# --------------------------------------------------------------------------
# Model setup
# --------------------------------------------------------------------------
def build_model(model_name, device, *, lora_r=16):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="eager")

    # [MASK] token: add a dedicated special token; new embedding row is the
    # absorbing state. mean-init it to speed the cold start.
    tok.add_special_tokens({"additional_special_tokens": ["<|mask|>"]})
    mask_id = tok.convert_tokens_to_ids("<|mask|>")
    model.resize_token_embeddings(len(tok))
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        emb[mask_id] = emb[:mask_id].mean(0)

    # Grad-checkpointing + input grads MUST be enabled before get_peft_model,
    # otherwise no gradients flow through the frozen, checkpointed base.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        # The cold [MASK] embedding (and tied lm_head) must be fully trainable.
        modules_to_save=["embed_tokens", "lm_head"])
    model = get_peft_model(model, lora)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_train > 0, "no trainable params -- check enable_input_require_grads order"
    print(f"  trainable params: {n_train:,}")
    return tok, model.to(device), mask_id


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def text_batches(tok, seq_len, batch_size, device, *, dataset="wikitext"):
    """Yield packed (B, seq_len) id tensors forever. Uses a small HF dataset by
    default; falls back to a tiny built-in corpus if datasets is unavailable."""
    try:
        from datasets import load_dataset
        if dataset == "wikitext":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = (r for r in ds["text"] if r.strip())
        else:
            ds = load_dataset(dataset, split="train", streaming=True)
            texts = (r["text"] for r in ds if r.get("text", "").strip())
    except Exception as e:  # offline / no datasets -> tiny fallback
        print(f"  [data] dataset load failed ({e}); using built-in corpus")
        texts = iter(["To be, or not to be, that is the question. "] * 100000)

    buf = []
    while True:
        for txt in texts:
            buf.extend(tok(txt, add_special_tokens=False).input_ids)
            while len(buf) >= batch_size * seq_len:
                chunk = buf[: batch_size * seq_len]
                buf = buf[batch_size * seq_len:]
                yield torch.tensor(chunk, device=device).view(batch_size, seq_len)
        texts = iter([])  # exhausted; outer while reuses buffer / fallback


# --------------------------------------------------------------------------
# Train + sample
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--anneal_steps", type=int, default=400)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--dataset", default="wikitext")
    ap.add_argument("--gen_len", type=int, default=64)
    ap.add_argument("--gen_steps", type=int, default=64)
    ap.add_argument("--save", default="/kaggle/working/diffusion_adapter")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, model={args.model}")

    tok, model, mask_id = build_model(args.model, device, lora_r=args.lora_r)

    anneal_state = {"causal_prob": 1.0}
    restore = force_bidirectional(model, anneal_state=anneal_state)
    anneal_state["causal_prob"] = 0.0          # verify in pure-bidirectional mode
    model.eval(); verify_bidirectional(model, tok, device); model.train()

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr)
    data = text_batches(tok, args.seq_len, args.bs, device, dataset=args.dataset)

    print("-" * 56)
    opt.zero_grad()
    for step in range(1, args.steps + 1):
        anneal_state["causal_prob"] = D.anneal_causal_prob(step, args.anneal_steps)
        x0 = next(data)
        t = D.sample_mask_rate(x0.shape[0], device=device)
        x_t, m = D.forward_mask(x0, t, mask_id)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(input_ids=x_t).logits
            loss = D.diffusion_loss(logits, x0, m, t) / args.grad_accum
        loss.backward()
        if step % args.grad_accum == 0:
            opt.step(); opt.zero_grad()
        if step % args.log_every == 0:
            print(f"  step {step:>5} | causal_p {anneal_state['causal_prob']:.2f} "
                  f"| loss {loss.item() * args.grad_accum:.3f}")
        if step % args.save_every == 0:
            model.save_pretrained(args.save)

    # ---- generation (fully bidirectional) ----
    anneal_state["causal_prob"] = 0.0
    model.eval()

    def fwd(ids):
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            return model(input_ids=ids).logits.float()

    print("-" * 56)
    prompt = tok("The meaning of life is", return_tensors="pt").input_ids.to(device)
    out = D.diffusion_generate(fwd, length=prompt.shape[1] + args.gen_len,
                               mask_id=mask_id, steps=args.gen_steps,
                               prompt_ids=prompt, device=device)
    print("denoised continuation:")
    print(" ", tok.decode(out[0], skip_special_tokens=True))

    model.save_pretrained(args.save)
    restore()
    print(f"saved adapter to {args.save}")


if __name__ == "__main__":
    main()
