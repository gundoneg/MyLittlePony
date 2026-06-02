"""Port the GPT donor into an SSM target.

The reusable method, two pieces:
  1. GAUGE-FREE BRIDGE: copy emb / pos / lm_head from donor -> target and FREEZE
     them. Donor and target share vocab and d_model, so these tensors transfer
     verbatim. This is the structured emb/unemb transfer validated in the
     numpy weight-translator lab -- the shared vocab representation is the
     gauge-free bridge between the two architectures.
  2. DISTILLATION: train ONLY the SSM interior (mixers, FFNs, norms) to match
     the donor's next-token distribution (KL on logits + a little CE), reading
     and writing through the frozen shared embedding space. The SSM thus learns
     to reproduce the Transformer's function in the same representation.

Saves target.pt.
"""
import argparse, math, os, time

import torch
import torch.nn.functional as F

from data import get_tokenizer, load_ids, Batcher
from donor import GPT, Config
from target import SSMLM

HERE = os.path.dirname(os.path.abspath(__file__))


def transfer_gauge_free(donor, target):
    """Copy the gauge-free vocab/positional tensors and freeze them in target."""
    with torch.no_grad():
        target.tok.weight.copy_(donor.tok.weight)
        target.pos.weight.copy_(donor.pos.weight)
        target.head.weight.copy_(donor.head.weight)
        # final norm is also a readout-side gauge piece; copy as a good init
        target.nf.g.copy_(donor.nf.g)
    for p in (target.tok.weight, target.pos.weight, target.head.weight):
        p.requires_grad_(False)
    return target


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
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--temp", type=float, default=2.0, help="distillation temperature")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on hard CE vs KL")
    ap.add_argument("--donor", default=os.path.join(HERE, "donor.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "target.pt"))
    ap.add_argument("--eval_every", type=int, default=250)
    args = ap.parse_args()

    torch.manual_seed(7)
    bpe = get_tokenizer()
    ids = load_ids(bpe)

    ckpt = torch.load(args.donor, weights_only=False)
    cfg = Config(**ckpt["cfg"])
    batcher = Batcher(ids, cfg.ctx)

    donor = GPT(cfg)
    donor.load_state_dict(ckpt["model"])
    donor.eval()
    for p in donor.parameters():
        p.requires_grad_(False)
    print(f"loaded donor (val loss {ckpt.get('val_loss', float('nan')):.3f})")

    target = SSMLM(cfg)
    transfer_gauge_free(donor, target)
    trainable = [p for p in target.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in target.parameters())
    print(f"target SSM: {n_all/1e6:.2f}M params, {n_tr/1e6:.2f}M trainable "
          f"(emb/pos/head frozen = gauge-free bridge)")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.1)
    T = args.temp
    t0 = time.time()
    best = float("inf")
    for step in range(1, args.steps + 1):
        x, y = batcher.batch("train", args.bs, torch)
        with torch.no_grad():
            tlogits, _ = donor(x)
        slogits, _ = target(x)
        # KL(teacher || student) at temperature T
        kl = F.kl_div(
            F.log_softmax(slogits / T, dim=-1),
            F.softmax(tlogits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
        ce = F.cross_entropy(slogits.view(-1, slogits.size(-1)), y.view(-1))
        loss = (1 - args.alpha) * kl + args.alpha * ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == 1:
            vl = eval_loss(target, batcher, 20, args.bs)
            dt = time.time() - t0
            print(f"step {step:5d}/{args.steps} | kl {kl.item():.3f} ce {ce.item():.3f} | "
                  f"val {vl:.3f} ppl {math.exp(vl):.1f} | {dt:.0f}s")
            if vl < best:
                best = vl
                torch.save({"model": target.state_dict(), "cfg": cfg.__dict__,
                            "val_loss": vl}, args.out)

    prompt = torch.tensor([bpe.encode("ROMEO:\n")], dtype=torch.long)
    out = target.generate(prompt, 120, temperature=0.8, top_k=40)
    print("\n--- target (SSM) sample ---")
    print(bpe.decode(out[0].tolist()))
    print(f"\nbest target val loss {best:.3f} (ppl {math.exp(best):.1f}) saved -> {args.out}")
    print(f"donor val loss {ckpt.get('val_loss', float('nan')):.3f} "
          f"(ppl {math.exp(ckpt['val_loss']):.1f}) for comparison")


if __name__ == "__main__":
    main()
