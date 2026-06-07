"""Calibrate the layer-interdependence knob (phase 8). For a fixed depth, sweep alpha
and report donor-acc + per-layer ablation. The knob is valid if donors learn it AND the
ablation drop (depth-usage) RISES with alpha: alpha~0 => deep blocks idle (depth not
needed), alpha~1 => every block used (pure recursion)."""
import torch
from models import Cfg
from task import sample_tasks_interdep
from donor_zoo import build_donor_zoo
from sanity_hops import ablation


def run(L, alpha, steps, maxjump=2, n=4):
    cfg = Cfg(vocab=16, d_model=64, n_layer=L, n_head=4, ctx=32, d_ff=256)
    tasks = sample_tasks_interdep(n, cfg.vocab, depth=L, alpha=alpha, maxjump=maxjump, seed=100 + L)
    zoo = build_donor_zoo(cfg, tasks, steps=steps, tied=True, verbose=False)
    dm = sum(z["acc"] for z in zoo) / n
    dr = ablation(zoo, cfg)
    print(f"  L={L} alpha={alpha:.2f} steps={steps} | donor-acc {dm:.0%} "
          f"| ablation {[f'{d:.0%}' for d in dr]}")


if __name__ == "__main__":
    print("interdependence knob: ablation should rise with alpha:")
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        run(4, a, 1200)
