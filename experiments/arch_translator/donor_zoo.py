"""Phase 3 — train a zoo of transformer donors A_i, one per task tau_i.

Two init regimes (the anchor ablation that links back to phase 1-2):
  * tied        : every donor starts from the SAME weights (shared coordinate
                  frame) -> the translator can align them.
  * independent : every donor gets its own random init (no shared frame).
"""
import argparse
import time
import torch

from models import Cfg, LM
from task import (sample_tasks, make_batch, make_multitask_batch,
                  masked_ce, task_accuracy)

TIED_SEED = 1234


def batch_for(task, bs, ctx, vocab, g):
    """Dispatch: a single task (dict) or a multi-task bundle (list of dicts)."""
    if isinstance(task, list):
        return make_multitask_batch(task, bs, ctx, vocab, g)
    return make_batch(task, bs, ctx, vocab, g)


def train_donor(cfg, task, steps, init_seed, bs=64, lr=3e-3):
    torch.manual_seed(init_seed)
    A = LM(cfg, "transformer")
    opt = torch.optim.AdamW(A.parameters(), lr=lr)
    g = torch.Generator().manual_seed(7000 + init_seed)
    A.train()
    for _ in range(steps):
        x, y = batch_for(task, bs, cfg.ctx, cfg.vocab, g)
        logits, _ = A(x)
        loss = masked_ce(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return A


@torch.no_grad()
def donor_acc(A, cfg, task, iters=8, bs=128):
    A.eval()
    g = torch.Generator().manual_seed(99)
    acc = 0.0
    for _ in range(iters):
        x, y = batch_for(task, bs, cfg.ctx, cfg.vocab, g)
        acc += task_accuracy(A(x)[0], y)
    return acc / iters


def build_donor_zoo(cfg, tasks, steps=300, tied=True, verbose=True):
    zoo = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        init_seed = TIED_SEED if tied else (1000 + i)
        A = train_donor(cfg, task, steps, init_seed)
        acc = donor_acc(A, cfg, task)
        zoo.append(dict(task=task, A=A.state_dict(), acc=acc))
        if verbose:
            print(f"  [{i:2d}] k={task['k']} donor-acc {acc:.2%}  ({time.time()-t0:.0f}s)")
    return zoo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=24)
    ap.add_argument("--n_held", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--tied", action="store_true", default=True)
    ap.add_argument("--independent", dest="tied", action="store_false")
    ap.add_argument("--out", default="donors_tied.pt")
    args = ap.parse_args()

    cfg = Cfg(vocab=16, d_model=64, n_layer=1, n_head=4, ctx=32, d_ff=256)
    tasks = sample_tasks(args.n_train + args.n_held, cfg.vocab, seed=0)
    print(f"donor zoo | V={cfg.vocab} ctx={cfg.ctx} d={cfg.d_model} | "
          f"{args.n_train}+{args.n_held} tasks | tied={args.tied} | chance={1/cfg.vocab:.1%}")
    zoo = build_donor_zoo(cfg, tasks, args.steps, args.tied)
    meta = dict(n_train=args.n_train, n_held=args.n_held, tied=args.tied)
    torch.save(dict(cfg=cfg.__dict__, zoo=zoo, meta=meta), args.out)
    mean = sum(z["acc"] for z in zoo) / len(zoo)
    print(f"saved {args.out} | mean donor-acc {mean:.2%}")


if __name__ == "__main__":
    main()
