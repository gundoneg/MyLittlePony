"""Обучение «зоопарка» доноров: по донору на каждую задачу. Веса -> в память/файл.

Донор учится обычным CE; затем сохраняем его параметры (numpy). Переводчик потом
читает ТОЛЬКО веса (без инференса донора).
"""
from __future__ import annotations
import numpy as np
from autograd import cross_entropy
from nn import to_tensors, Adam
import donor, data


def train_one_donor(task, seed, iters=200, lr=5e-3, batch=64, cfg=donor.CFG,
                    init_seed=None):
    rng = np.random.default_rng(seed)
    p = donor.init_params(np.random.default_rng(init_seed) if init_seed is not None
                          else rng, cfg)
    opt = Adam(p, lr=lr)
    for it in range(iters):
        x, y = data.sample_batch(task, rng, batch=batch, V=cfg["V"], L=cfg["L"])
        pt = to_tensors(p, requires_grad=True)
        lg = donor.forward(pt, x, cfg)
        V = lg.shape[-1]
        loss = cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        loss.backward()
        opt.step(p, {k: pt[k].grad for k in p})
    return p


def eval_params(mod, p, task, rng, n=256, cfg=donor.CFG):
    x, y = data.sample_batch(task, rng, batch=n, V=cfg["V"], L=cfg["L"])
    lg = mod.forward(to_tensors(p, False), x, cfg)
    V = lg.shape[-1]
    pred = lg.reshape(-1, V).data.argmax(-1)
    return float((pred == y.reshape(-1)).mean())


def build_zoo(tasks, iters=200, base_seed=1000, cfg=donor.CFG, verbose=True,
              init_seed=None):
    """init_seed=None -> independent init (как раньше); int -> tied-init (общий базис)."""
    zoo = []
    rng = np.random.default_rng(7)
    for i, task in enumerate(tasks):
        p = train_one_donor(task, base_seed + i, iters=iters, cfg=cfg,
                            init_seed=init_seed)
        acc = eval_params(donor, p, task, rng, cfg=cfg)
        zoo.append({"task": task, "params": p, "acc": acc})
        if verbose:
            print(f"  donor {i:2d}: acc={acc:.3f} (lag={task['k']})")
    return zoo
