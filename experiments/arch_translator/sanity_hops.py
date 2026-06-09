"""Gating check for the depth-demanding pointer-chasing task.

The whole point of phase-6 (scaling the translator C with depth) is degenerate if
the donors don't actually USE their depth: the (sigma,k) task is one gather, so
deep blocks sit idle (layer-ablation ~0% drop) and their weights are arbitrary /
un-translatable. This script checks whether pointer-chasing (hops = depth) fixes
that: depth-L donors should (a) learn it, and (b) show per-layer ablation drops at
EVERY layer, proving the deep blocks carry computation.

Run:  python sanity_hops.py
"""
import torch
from models import Cfg, LM
from task import sample_tasks, sample_tasks_hops, make_batch, task_accuracy
from donor_zoo import build_donor_zoo


def neutralise_block(state, l):
    out = {k: v.clone() for k, v in state.items()}
    for k in (f"blocks.{l}.mix.o.weight", f"blocks.{l}.mix.o.bias",
              f"blocks.{l}.mlp.proj.weight", f"blocks.{l}.mlp.proj.bias"):
        if k in out:
            out[k] = torch.zeros_like(out[k])
    return out


@torch.no_grad()
def _acc(m, cfg, task, iters, bs, seed):
    g = torch.Generator().manual_seed(seed)
    a = 0.0
    for _ in range(iters):
        x, y = make_batch(task, bs, cfg.ctx, cfg.vocab, g)
        a += task_accuracy(m(x)[0], y)
    return a / iters


def ablation(zoo, cfg):
    drops = [0.0] * cfg.n_layer
    for z in zoo:
        m = LM(cfg, "transformer"); m.load_state_dict(z["A"]); m.eval()
        base = _acc(m, cfg, z["task"], 6, 128, 99)
        for l in range(cfg.n_layer):
            mn = LM(cfg, "transformer"); mn.load_state_dict(neutralise_block(z["A"], l)); mn.eval()
            drops[l] += base - _acc(mn, cfg, z["task"], 6, 128, 99)
    return [d / len(zoo) for d in drops]


def run(label, tasks_fn, depths, n=6, steps=400):
    print(f"\n=== {label} ===")
    for L in depths:
        cfg = Cfg(vocab=16, d_model=64, n_layer=L, n_head=4, ctx=32, d_ff=256)
        tasks = tasks_fn(n, cfg, L)
        zoo = build_donor_zoo(cfg, tasks, steps=steps, tied=True, verbose=False)
        dm = sum(z["acc"] for z in zoo) / n
        dr = ablation(zoo, cfg)
        print(f"  L={L} donor-acc {dm:.0%} | per-layer ablation drop "
              f"{[f'{d:.0%}' for d in dr]}")


if __name__ == "__main__":
    depths = [2, 4]
    run("(sigma,k) CONTROL -- expect idle deep blocks",
        lambda n, cfg, L: sample_tasks(n, cfg.vocab, seed=100 + L), depths)
    run("pointer-hops (hops=depth) -- expect drops at every layer",
        lambda n, cfg, L: sample_tasks_hops(n, cfg.vocab, hops=L, seed=100 + L), depths)
