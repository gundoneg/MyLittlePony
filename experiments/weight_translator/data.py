"""Синтетическое семейство задач для «зоопарка доноров».

Задача tau = (sigma, k): перестановка алфавита sigma и лаг k.
Цель: y_t = sigma(x_{t-k}); для t<k цель = sigma(PAD) (фиксированный токен 0).

Нужны И поточечное отображение (sigma), И смешивание по времени (лаг k) =>
задача честно требует sequence-mixing, различая внимание и SSM.
"""
from __future__ import annotations
import numpy as np

# дефолтные размеры (крошечные — под телефон)
V = 12      # размер алфавита
L = 12      # длина последовательности


def make_task(rng):
    sigma = rng.permutation(V)
    k = int(rng.integers(1, 4))   # лаг 1..3
    return {"sigma": sigma, "k": k}


def sample_batch(task, rng, batch=64, V=V, L=L):
    x = rng.integers(0, V, size=(batch, L))
    sigma, k = task["sigma"], task["k"]
    y = np.zeros((batch, L), dtype=np.int64)
    for t in range(L):
        if t < k:
            y[:, t] = sigma[0]            # PAD-токен = 0 -> sigma(0)
        else:
            y[:, t] = sigma[x[:, t - k]]
    return x.astype(np.int64), y


def make_task_split(seed=0, n_train=32, n_held=8):
    """Непересекающиеся train/held-out наборы задач (по (sigma,k))."""
    rng = np.random.default_rng(seed)
    seen, tasks = set(), []
    while len(tasks) < n_train + n_held:
        t = make_task(rng)
        key = (tuple(t["sigma"].tolist()), t["k"])
        if key in seen:
            continue
        seen.add(key)
        tasks.append(t)
    return tasks[:n_train], tasks[n_train:n_train + n_held]
