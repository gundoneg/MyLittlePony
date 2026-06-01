"""Утилиты: инициализация параметров, Adam, упаковка dict<->вектор.

Модели функциональные: params — это dict[str, np.ndarray]; forward(params_t, x)
принимает dict[str, Tensor]. Так переводчику удобно ВЫДАВАТЬ параметры цели и
сразу гонять её forward (градиент течёт в переводчик).
"""
from __future__ import annotations
import numpy as np
from autograd import Tensor


def glorot(rng, *shape, scale=1.0):
    fan = shape[0] + (shape[1] if len(shape) > 1 else 0)
    s = scale * np.sqrt(2.0 / max(fan, 1))
    return rng.standard_normal(shape) * s


def to_tensors(params: dict, requires_grad=True) -> dict:
    return {k: Tensor(v, requires_grad=requires_grad) for k, v in params.items()}


def flatten(params: dict):
    """dict -> (vector, spec) для упаковки в/из переводчика."""
    keys = sorted(params.keys())
    vecs, spec = [], []
    for k in keys:
        a = params[k]
        spec.append((k, a.shape))
        vecs.append(a.reshape(-1))
    return np.concatenate(vecs), spec


def total_size(spec):
    return int(sum(int(np.prod(sh)) for _, sh in spec))


def unflatten_tensor(vec: Tensor, spec) -> dict:
    """Tensor-вектор (P,) -> dict[str, Tensor] по spec (с сохранением графа)."""
    out, off = {}, 0
    for k, sh in spec:
        n = int(np.prod(sh))
        out[k] = vec.reshape(-1) if False else None
        sub = _slice(vec, off, off + n)
        out[k] = sub.reshape(*sh) if len(sh) > 0 else sub.reshape(())
        off += n
    return out


def _slice(vec: Tensor, a, b):
    """Срез вектора-Tensor [a:b] с поддержкой backward."""
    data = vec.data[a:b]
    out = Tensor(data, requires_grad=vec.requires_grad, _children=(vec,))
    if out.requires_grad:
        def bw():
            g = np.zeros_like(vec.data)
            g[a:b] = out.grad
            Tensor._acc(vec, g)
        out._backward = bw
    return out


class Adam:
    def __init__(self, params: dict, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k in params:
            g = grads[k]
            if g is None:
                continue
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
