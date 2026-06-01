"""Train the GPT donor from scratch on TinyShakespeare. Saves donor.pt."""
import argparse, math, os, time

import torch

from data import get_tokenizer, load_ids, Batcher
from donor import GPT, Config

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def eval_loss(model, batcher, iters, bs):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = batcher.batch("val", bs, torch)
        _, loss = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--out", default=os.path.join(HERE, "donor.pt"))
    args = ap.parse_args()

    torch.manual_seed(1337)
    bpe = get_tokenizer()
    ids = load_ids(bpe)
    batcher = Batcher(ids, args.ctx)

    cfg = Config(vocab_size=len(bpe.vocab), ctx=args.ctx, d_model=args.d_model,
                 n_layer=args.n_layer, n_head=args.n_head, d_ff=args.d_ff)
    model = GPT(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"donor GPT: {n/1e6:.2f}M params, vocab {cfg.vocab_size}, ctx {cfg.ctx}, "
          f"{cfg.n_layer}L d{cfg.d_model}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.1)

    t0 = time.time()
    best = float("inf")
    for step in range(1, args.steps + 1):
        x, y = batcher.batch("train", args.bs, torch)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == 1:
            vl = eval_loss(model, batcher, 20, args.bs)
            dt = time.time() - t0
            print(f"step {step:5d}/{args.steps} | train {loss.item():.3f} | "
                  f"val {vl:.3f} ppl {math.exp(vl):.1f} | {dt:.0f}s")
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                            "val_loss": vl}, args.out)

    # final sample
    prompt = torch.tensor([bpe.encode("ROMEO:\n")], dtype=torch.long)
    out = model.generate(prompt, 120, temperature=0.8, top_k=40)
    print("\n--- donor sample ---")
    print(bpe.decode(out[0].tolist()))
    print(f"\nbest val loss {best:.3f} (ppl {math.exp(best):.1f}) saved -> {args.out}")


if __name__ == "__main__":
    main()
