"""Phase 3 orchestrator — does cross-arch zero-shot translation work, and does it
need the shared coordinate frame (the anchor lesson from phases 1-2)?

Builds donor zoos in BOTH init regimes (tied / independent), trains the translator
C on the train donors, and reports zero-shot held-out task accuracy plus controls:
  * zero-shot   : B* = C(A*) on unseen tasks
  * mismatched-A: feed the wrong donor's weights (should collapse to chance)
  * donor-acc   : the transformer donors' own accuracy (axis is learnable)
"""
import argparse
import torch

from models import Cfg
from task import sample_tasks
from donor_zoo import build_donor_zoo
from xlate import train_translator, eval_translator


def run_regime(cfg, tasks, n_train, tied, donor_steps, c_steps):
    zoo = build_donor_zoo(cfg, tasks, donor_steps, tied=tied, verbose=False)
    train, held = zoo[:n_train], zoo[n_train:]
    donor = sum(z["acc"] for z in zoo) / len(zoo)
    C, tmpl = train_translator(cfg, train, steps=c_steps)
    zs = eval_translator(C, tmpl, cfg, held, mismatch=False)
    mm = eval_translator(C, tmpl, cfg, held, mismatch=True)
    return dict(donor=donor, zero_shot=zs, mismatch=mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=24)
    ap.add_argument("--n_held", type=int, default=8)
    ap.add_argument("--donor_steps", type=int, default=300)
    ap.add_argument("--c_steps", type=int, default=600)
    args = ap.parse_args()

    cfg = Cfg(vocab=16, d_model=64, n_layer=1, n_head=4, ctx=32, d_ff=256)
    tasks = sample_tasks(args.n_train + args.n_held, cfg.vocab, seed=0)
    chance = 1 / cfg.vocab
    print(f"PHASE 3 — cross-arch zero-shot translation (tau = sigma + lag k)")
    print(f"  V={cfg.vocab} ctx={cfg.ctx} d={cfg.d_model} L={cfg.n_layer} | "
          f"{args.n_train} train + {args.n_held} held tasks | chance={chance:.1%}")

    print("\n  regime        donor-acc  zero-shot  mismatched-A")
    for tied in (True, False):
        r = run_regime(cfg, tasks, args.n_train, tied, args.donor_steps, args.c_steps)
        name = "tied (anchor)" if tied else "independent"
        print(f"  {name:13s}  {r['donor']:>8.1%}  {r['zero_shot']:>8.1%}  {r['mismatch']:>10.1%}")

    print(f"\n  read-out: if TIED zero-shot >> chance ({chance:.1%}) and >> mismatched-A,")
    print("  cross-arch transfer works; if INDEPENDENT collapses to chance, it is the")
    print("  shared coordinate frame (phases 1-2) that makes the dream switch on.")


if __name__ == "__main__":
    main()
