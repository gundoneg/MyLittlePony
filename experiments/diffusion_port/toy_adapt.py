"""CPU driver: adapt the toy transformer `LM` into a discrete-diffusion LM on
char-level Shakespeare, using the model-agnostic core in `diffusion.py`.

This is the LOCAL verification of the same recipe the Kaggle notebook runs on
Qwen3.5-0.8B: bidirectional attention with causal->full annealing, the LLaDA
1/t masked-diffusion objective, and confidence-based parallel denoising. It runs
in seconds-to-minutes on CPU and prints a text table of {step, mask_rate,
masked_CE} plus qualitative denoised samples.

Run:  python experiments/diffusion_port/toy_adapt.py --steps 600
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "arch_translator"))
sys.path.insert(0, os.path.dirname(__file__))

from models import LM, Cfg                                   # noqa: E402
from data import CharData                                    # noqa: E402
import diffusion as D                                        # noqa: E402


def get_x0(data, split, bs, ctx, generator):
    """Grab a contiguous block of ids (ignore CharData's shifted y -- diffusion
    re-derives targets from the un-noised x)."""
    x, _ = data.batch(split, bs, ctx, generator=generator)
    return x


@torch.no_grad()
def eval_loss(model, data, mask_id, ctx, *, t_fixed=0.5, n_batch=8, bs=64):
    """Clean held-out masked-CE at a FIXED mask rate (removes the t-sampling
    variance that makes the train loss look flat). Lower is better; unigram
    char entropy is ln(vocab)."""
    g = torch.Generator().manual_seed(1234)
    was = model.training
    model.eval()
    D.set_bidirectional(model, causal=False)
    tot = 0.0
    for _ in range(n_batch):
        x0 = get_x0(data, "val", bs, ctx, g)
        t = torch.full((x0.shape[0],), t_fixed)
        x_t, m = D.forward_mask(x0, t, mask_id, generator=g)
        logits = model(x_t)[0]
        tot += D.diffusion_loss(logits, x0, m, t).item()
    if was:
        model.train()
    return tot / n_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--anneal_steps", type=int, default=400,
                    help="causal->bidirectional annealing horizon (0 = always bidir)")
    ap.add_argument("--gen_len", type=int, default=64)
    ap.add_argument("--gen_steps", type=int, default=64)
    ap.add_argument("--gen_temp", type=float, default=0.3,
                    help="sampling temperature (>0 breaks the cold-start collapse "
                         "to the most-frequent token)")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    data = CharData(device="cpu")
    mask_id = data.vocab                      # reserve one extra row for [MASK]
    cfg = Cfg(vocab=data.vocab + 1, d_model=args.d_model, n_layer=args.n_layer,
              n_head=args.n_head, ctx=args.ctx, bidirectional=True)
    model = LM(cfg, "transformer")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"toy AR->diffusion adaptation on char-Shakespeare")
    print(f"  vocab={data.vocab} (+1 mask), d_model={cfg.d_model}, "
          f"n_layer={cfg.n_layer}, ctx={args.ctx}, anneal_steps={args.anneal_steps}")
    print(f"  params={sum(p.numel() for p in model.parameters()):,}")
    import math as _math
    print(f"  unigram char entropy = ln(vocab) = {_math.log(data.vocab):.3f} nats "
          f"(eval below should fall well under this)")
    print("-" * 64)
    print(f"{'step':>5} | {'causal_p':>8} | {'train_CE':>9} | {'val_CE@t=.5':>11}")
    print("-" * 64)

    for step in range(1, args.steps + 1):
        x0 = get_x0(data, "train", args.bs, args.ctx, g)
        t = D.sample_mask_rate(args.bs, generator=g)
        x_t, m = D.forward_mask(x0, t, mask_id, generator=g)

        # Mask annealing: with prob rho keep this batch causal (AR-like) early,
        # ramping to fully bidirectional. Implemented via the instance flag.
        rho = D.anneal_causal_prob(step, args.anneal_steps)
        causal = torch.rand(1, generator=g).item() < rho
        D.set_bidirectional(model, causal=causal)

        logits = model(x_t)[0]
        loss = D.diffusion_loss(logits, x0, m, t)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % args.eval_every == 0 or step == 1:
            vce = eval_loss(model, data, mask_id, args.ctx)
            print(f"{step:>5} | {rho:>8.2f} | {loss.item():>9.3f} | {vce:>11.3f}")

    # ---- qualitative denoising (bidirectional inference) ----
    D.set_bidirectional(model, causal=False)
    model.eval()
    print("-" * 56)
    print("denoised samples (start = all [MASK], confidence unmasking):")
    out = D.diffusion_generate(lambda ids: model(ids)[0], length=args.gen_len,
                               mask_id=mask_id, steps=args.gen_steps,
                               batch_size=3, temperature=args.gen_temp,
                               alg_temp=0.5, generator=g, device="cpu")
    for i in range(out.shape[0]):
        print(f"  [{i}] {data.decode(out[i].tolist())!r}")

    # prompt-conditioned continuation
    prompt = data.encode("ROMEO:")[None]
    cont = D.diffusion_generate(lambda ids: model(ids)[0], length=args.gen_len,
                                mask_id=mask_id, steps=args.gen_steps,
                                prompt_ids=prompt, temperature=args.gen_temp,
                                alg_temp=0.5, generator=g, device="cpu")
    print("prompt-conditioned ('ROMEO:' + masked continuation):")
    print(f"  {data.decode(cont[0].tolist())!r}")


if __name__ == "__main__":
    main()
