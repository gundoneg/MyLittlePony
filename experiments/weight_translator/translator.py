"""Переводчик H — гиперсеть: theta_B = theta0 + M @ z,  z = MLP(features(A)).

Вход — признаки донора (inv/raw/qr). Выход — ВСЕ параметры цели B (плоский
вектор P), распаковываемый в dict и подаваемый в target.forward. Обучается
поведенческим лоссом (CE задачи на выходе B); донор не запускается.

theta0 — общий «базовый» B (как random-init); M·z — низкоранговая (rank d_z)
модуляция под конкретную задачу, прочитанную из весов донора.
"""
from __future__ import annotations
import numpy as np
from autograd import Tensor
from nn import glorot, flatten, unflatten_tensor
import target


def b_spec(cfg=target.CFG):
    p0 = target.init_params(np.random.default_rng(0), cfg)
    vec, spec = flatten(p0)
    return spec, vec.shape[0]


def init_translator(rng, F, P, d_z=8, h=32, theta0=None):
    tp = {
        "W1": glorot(rng, h, F, scale=0.5),
        "b1": np.zeros(h),
        "W2": glorot(rng, d_z, h, scale=0.5),
        "b2": np.zeros(d_z),
        "theta0": theta0.copy() if theta0 is not None else glorot(rng, P) * 0.0,
        "M": glorot(rng, P, d_z, scale=0.1),
    }
    return tp


def translate(tt: dict, feat: np.ndarray, spec):
    """tt: dict[str,Tensor]; feat: (F,) np -> dict[str,Tensor] параметров B."""
    f = Tensor(feat)
    z = (tt["W1"] @ f + tt["b1"]).silu()
    z = tt["W2"] @ z + tt["b2"]               # (d_z,)
    theta = tt["theta0"] + (tt["M"] @ z)      # (P,)
    return unflatten_tensor(theta, spec)
