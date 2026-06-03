"""Phase 3b orchestrator — does POST-HOC alignment recover cross-arch transfer?

Compares three regimes on the SAME task set:
  * tied               : shared frame by construction (phase-3 baseline)
  * independent (raw)   : own random frame each -> transfer dies
  * independent (aligned): each donor rotated onto a reference donor's frame
                           (align.py), with NO retraining of the donors

Also reports a warm-start finisher: a few fine-tune steps on B from the translated
init vs from a random init, to show the practical "clean-up" step.
"""
import argparse
import torch

from models import Cfg
from task import sample_tasks
from donor_zoo import build_donor_zoo
from align import align_zoo
from xlate import (train_translator, eval_translator, warm_start_acc,
                   random_B_params)


def zero_shot(cfg, train, held, c_steps):
    C, tmpl = train_translator(cfg, train, steps=c_steps)
    zs = eval_translator(C, tmpl, cfg, held, mismatch=False)
    mm = eval_translator(C, tmpl, cfg, held, mismatch=True)
    return C, tmpl, zs, mm


def warm(C, tmpl, cfg, held, steps=15):
    trans = sum(warm_start_acc(tmpl, C.emit(z["A"]), cfg, z["task"], steps=steps)
                for z in held) / len(held)
    rand = sum(warm_start_acc(tmpl, random_B_params(cfg, seed=s), cfg, z["task"], steps=steps)
               for s, z in enumerate(held)) / len(held)
    return trans, rand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20)
    ap.add_argument("--n_held", type=int, default=8)
    ap.add_argument("--donor_steps", type=int, default=250)
    ap.add_argument("--c_steps", type=int, default=500)
    ap.add_argument("--warm_steps", type=int, default=15)
    args = ap.parse_args()

    cfg = Cfg(vocab=16, d_model=64, n_layer=1, n_head=4, ctx=32, d_ff=256)
    tasks = sample_tasks(args.n_train + args.n_held, cfg.vocab, seed=0)
    nt, chance = args.n_train, 1 / cfg.vocab
    print(f"PHASE 3b — post-hoc alignment recovers transfer? | V={cfg.vocab} d={cfg.d_model} "
          f"| {args.n_train}+{args.n_held} tasks | chance={chance:.1%}")

    tied = build_donor_zoo(cfg, tasks, args.donor_steps, tied=True, verbose=False)
    indep = build_donor_zoo(cfg, tasks, args.donor_steps, tied=False, verbose=False)
    ref = indep[0]["A"]                                  # reference frame = donor 0
    indep_al = align_zoo(indep, ref)

    rows = []
    # tied baseline
    _, _, zs, mm = zero_shot(cfg, tied[:nt], tied[nt:], args.c_steps)
    rows.append(("tied (anchor)", zs, mm, None, None))
    # independent raw
    _, _, zs, mm = zero_shot(cfg, indep[:nt], indep[nt:], args.c_steps)
    rows.append(("independent (raw)", zs, mm, None, None))
    # independent aligned (+ warm-start finisher)
    C, tmpl, zs, mm = zero_shot(cfg, indep_al[:nt], indep_al[nt:], args.c_steps)
    wt, wr = warm(C, tmpl, cfg, indep_al[nt:], steps=args.warm_steps)
    rows.append(("independent (aligned)", zs, mm, wt, wr))

    print(f"\n  regime                  zero-shot  mismatched-A  warm@{args.warm_steps}  random@{args.warm_steps}")
    for name, zs, mm, wt, wr in rows:
        w = f"{wt:>7.1%}" if wt is not None else "      -"
        r = f"{wr:>8.1%}" if wr is not None else "       -"
        print(f"  {name:22s}  {zs:>8.1%}  {mm:>11.1%}  {w}  {r}")

    print(f"\n  read-out: if 'independent (aligned)' zero-shot jumps from ~chance back up,")
    print("  a shared frame can be RECOVERED post hoc for already-trained models; the gap")
    print("  to 'tied' is what an imperfect (orthogonal) alignment costs, and warm-start")
    print("  shows the few-step clean-up beating a random init.")


if __name__ == "__main__":
    main()
