"""Phase 4 — scaling study for the cross-arch translator.

Sweeps one axis at a time (from a baseline) in the TIED regime and reports, per
point: zero-shot, warm@15, and steps-to-90% (fine-tune steps for B to reach 90%)
for the translated init vs a random init.

  python scale_study.py --axis count|width|depth|arch|mtask
"""
import argparse
import torch

from models import Cfg
from task import sample_tasks, sample_task_bundles
from donor_zoo import build_donor_zoo
from xlate import (train_translator, eval_translator, warm_start_acc,
                   warm_curve, steps_to_threshold, random_B_params)

N_HELD = 6
DONOR_STEPS = 200
C_STEPS = 400
MAX_STEPS = 32
BASE = dict(vocab=16, d_model=64, n_layer=1, n_head=4, ctx=32, d_ff=256)


def measure(cfg, task_list, n_train, b_kind="ssm", warm_steps=15):
    zoo = build_donor_zoo(cfg, task_list, DONOR_STEPS, tied=True, verbose=False)
    train, held = zoo[:n_train], zoo[n_train:]
    C, tmpl = train_translator(cfg, train, steps=C_STEPS, b_kind=b_kind)
    zs = eval_translator(C, tmpl, cfg, held)
    w15 = sum(warm_start_acc(tmpl, C.emit(z["A"]), cfg, z["task"], steps=warm_steps)
              for z in held) / len(held)

    def s90(init_fn):
        tot = 0.0
        for i, z in enumerate(held):
            cur = warm_curve(tmpl, init_fn(i, z), cfg, z["task"], max_steps=MAX_STEPS, every=4)
            tot += steps_to_threshold(cur, 0.9, cap=MAX_STEPS)
        return tot / len(held)

    s_tr = s90(lambda i, z: C.emit(z["A"]))
    s_rd = s90(lambda i, z: random_B_params(cfg, seed=i, b_kind=b_kind))
    donor = sum(z["acc"] for z in zoo) / len(zoo)
    return donor, zs, w15, s_tr, s_rd


def row(label, r):
    donor, zs, w15, s_tr, s_rd = r
    st = f">{MAX_STEPS}" if s_tr >= MAX_STEPS else f"{s_tr:.0f}"
    sr = f">{MAX_STEPS}" if s_rd >= MAX_STEPS else f"{s_rd:.0f}"
    print(f"  {label:14s}  {donor:>7.1%}  {zs:>8.1%}  {w15:>7.1%}  {st:>9}  {sr:>9}")


def header(axis):
    print(f"PHASE 4 scaling | axis={axis} | tied | chance={1/BASE['vocab']:.1%} "
          f"| n_held={N_HELD} donor_steps={DONOR_STEPS} c_steps={C_STEPS}")
    print(f"  {'point':14s}  {'donor':>7}  {'zeroshot':>8}  {'warm@15':>7}  "
          f"{'st90>tr':>9}  {'st90>rnd':>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True,
                    choices=["count", "width", "depth", "arch", "mtask"])
    args = ap.parse_args()
    header(args.axis)
    nt0 = 20

    if args.axis == "count":
        for nt in (8, 16, 32, 48):
            cfg = Cfg(**BASE)
            tasks = sample_tasks(nt + N_HELD, cfg.vocab, seed=0)
            row(f"n_train={nt}", measure(cfg, tasks, nt))

    elif args.axis == "width":
        for d in (32, 64, 128):
            cfg = Cfg(**{**BASE, "d_model": d})
            tasks = sample_tasks(nt0 + N_HELD, cfg.vocab, seed=0)
            row(f"d_model={d}", measure(cfg, tasks, nt0))

    elif args.axis == "depth":
        for L in (1, 2, 4):
            cfg = Cfg(**{**BASE, "n_layer": L})
            tasks = sample_tasks(nt0 + N_HELD, cfg.vocab, seed=0)
            row(f"n_layer={L}", measure(cfg, tasks, nt0))

    elif args.axis == "arch":
        for bk in ("ssm", "transformer"):
            cfg = Cfg(**BASE)
            tasks = sample_tasks(nt0 + N_HELD, cfg.vocab, seed=0)
            row(f"B={bk}", measure(cfg, tasks, nt0, b_kind=bk))

    elif args.axis == "mtask":
        for M in (1, 2, 4):
            cfg = Cfg(**BASE)
            bundles = sample_task_bundles(nt0 + N_HELD, M, cfg.vocab, seed=0)
            row(f"M={M}", measure(cfg, bundles, nt0))


if __name__ == "__main__":
    main()
