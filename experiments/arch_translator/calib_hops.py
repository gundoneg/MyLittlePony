"""Calibrate pointer-chasing learnability vs depth so donors actually master the
task (translating an under-trained donor is meaningless). Sweeps donor_steps and
maxjump for hops=depth; reports donor-acc + per-layer ablation drops."""
import torch
from models import Cfg, LM
from task import sample_tasks_hops, make_batch, task_accuracy
from donor_zoo import build_donor_zoo
from sanity_hops import ablation


def run(L, steps, maxjump, n=4):
    cfg = Cfg(vocab=16, d_model=64, n_layer=L, n_head=4, ctx=32, d_ff=256)
    tasks = sample_tasks_hops(n, cfg.vocab, hops=L, maxjump=maxjump, seed=100 + L)
    zoo = build_donor_zoo(cfg, tasks, steps=steps, tied=True, verbose=False)
    dm = sum(z["acc"] for z in zoo) / n
    dr = ablation(zoo, cfg)
    print(f"  L={L} hops={L} steps={steps} maxjump={maxjump} | donor-acc {dm:.0%} "
          f"| ablation {[f'{d:.0%}' for d in dr]}")


if __name__ == "__main__":
    print("calibrating hops=depth learnability:")
    run(3, 800, 2)
    run(4, 1200, 2)
    run(4, 1200, 4)
    run(6, 1500, 2)
