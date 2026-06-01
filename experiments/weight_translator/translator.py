"""Переводчик H — варианты гиперсети, выдающие параметры цели B.

Варианты:
  - 'linear'     : θ_B = θ0 + M·z,  z = MLP1(features(A)).            [бейзлайн]
  - 'mlp'        : то же, но z = MLP2 (глубже, больше d_z).          [капасити]
  - 'structured' : словарные матрицы B (emb/unemb/pos) предсказываются
                   аффинно ИЗ словарных матриц донора (несут σ, gauge-свободны и
                   токен-выровнены между архитектурами), а «нутро» B
                   (микшер/FFN/нормы) = θ0_rest + M_rest·z(инварианты).

Обучается поведенческим лоссом (CE задачи на выходе B); донор не запускается
(его веса лишь читаются как константы).
"""
from __future__ import annotations
import numpy as np
from autograd import Tensor
from nn import glorot, flatten, unflatten_tensor
import target

VOCAB_KEYS = ("emb", "unemb", "pos")


def b_spec(cfg=target.CFG):
    p0 = target.init_params(np.random.default_rng(0), cfg)
    vec, spec = flatten(p0)
    return spec, vec.shape[0]


def split_spec(spec, keys=VOCAB_KEYS):
    vocab = [(k, sh) for (k, sh) in spec if k in keys]
    rest = [(k, sh) for (k, sh) in spec if k not in keys]
    return vocab, rest


def _rest_size(spec, keys=VOCAB_KEYS):
    _, rest = split_spec(spec, keys)
    return int(sum(int(np.prod(sh)) for _, sh in rest))


def _codebook(cfg):
    """Фиксированный общий кодбук U0 (d,V) и его псевдообратный U+ (V,d)."""
    d, V = cfg["d_model"], cfg["V"]
    r = np.random.default_rng(999)
    U0 = r.standard_normal((d, V))
    U0 /= np.linalg.norm(U0, axis=0, keepdims=True)
    return U0, np.linalg.pinv(U0)


def init_translator(rng, F, spec, variant="linear", d_z=8, h=32, theta0=None, cfg=target.CFG):
    P = int(sum(int(np.prod(sh)) for _, sh in spec))
    tp = {}
    if variant in ("linear", "mlp"):
        tp["W1"] = glorot(rng, h, F, scale=0.5)
        tp["b1"] = np.zeros(h)
        if variant == "mlp":
            tp["Wh"] = glorot(rng, h, h, scale=0.5)
            tp["bh"] = np.zeros(h)
        tp["W2"] = glorot(rng, d_z, h, scale=0.5)
        tp["b2"] = np.zeros(d_z)
        tp["theta0"] = theta0.copy() if theta0 is not None else glorot(rng, P) * 0.0
        tp["M"] = glorot(rng, P, d_z, scale=0.1)
    elif variant == "structured":
        d = cfg["d_model"]
        Prest = _rest_size(spec)
        # словарные аффинные карты (общие; per-task вход = веса донора)
        tp["Wee"] = np.eye(d) + glorot(rng, d, d, scale=0.05)
        tp["Wuu"] = np.eye(d) + glorot(rng, d, d, scale=0.05)
        tp["Wpp"] = np.eye(d) + glorot(rng, d, d, scale=0.05)
        # гиперсеть для «нутра»
        tp["W1"] = glorot(rng, h, F, scale=0.5)
        tp["b1"] = np.zeros(h)
        tp["W2"] = glorot(rng, d_z, h, scale=0.5)
        tp["b2"] = np.zeros(d_z)
        theta0_rest = (flatten({k: v for k, v in
                                target.init_params(np.random.default_rng(123), cfg).items()
                                if k not in VOCAB_KEYS})[0])
        tp["theta0"] = theta0_rest
        tp["M"] = glorot(rng, Prest, d_z, scale=0.1)
    elif variant == "struct2":
        # emb/unemb конструируются детерминированно из G донора (кодбук U0).
        keys = ("emb", "unemb")
        Prest = _rest_size(spec, keys)
        tp["W1"] = glorot(rng, h, F, scale=0.5)
        tp["b1"] = np.zeros(h)
        tp["W2"] = glorot(rng, d_z, h, scale=0.5)
        tp["b2"] = np.zeros(d_z)
        tp["theta0"] = flatten({k: v for k, v in
                                target.init_params(np.random.default_rng(123), cfg).items()
                                if k not in keys})[0]
        tp["M"] = glorot(rng, Prest, d_z, scale=0.1)
    else:
        raise ValueError(variant)
    return tp


def _z_from_feat(tt, feat, variant):
    f = Tensor(feat)
    z = (tt["W1"] @ f + tt["b1"]).silu()
    if variant == "mlp":
        z = (tt["Wh"] @ z + tt["bh"]).silu()
    return tt["W2"] @ z + tt["b2"]


def translate(tt, A_params, feat, spec, variant="linear", cfg=target.CFG):
    """tt: dict[str,Tensor]; A_params: dict[np] донора; feat: (F,) -> dict[str,Tensor] B."""
    if variant in ("linear", "mlp"):
        z = _z_from_feat(tt, feat, variant)
        theta = tt["theta0"] + (tt["M"] @ z)
        return unflatten_tensor(theta, spec)
    if variant == "structured":
        _, rest_spec = split_spec(spec)
        z = _z_from_feat(tt, feat, "linear")
        theta_rest = tt["theta0"] + (tt["M"] @ z)
        bp = unflatten_tensor(theta_rest, rest_spec)
        bp["emb"] = Tensor(A_params["emb"]) @ tt["Wee"]       # (V,d)@(d,d)
        bp["unemb"] = tt["Wuu"] @ Tensor(A_params["unemb"])   # (d,d)@(d,V)
        bp["pos"] = Tensor(A_params["pos"]) @ tt["Wpp"]       # (L,d)@(d,d)
        return bp

    # struct2: emb/unemb из G донора через общий кодбук U0
    keys = ("emb", "unemb")
    _, rest_spec = split_spec(spec, keys)
    z = _z_from_feat(tt, feat, "linear")
    theta_rest = tt["theta0"] + (tt["M"] @ z)
    bp = unflatten_tensor(theta_rest, rest_spec)
    U0, Uplus = _codebook(cfg)
    G = A_params["emb"] @ A_params["unemb"]               # (V,V) базис-инвариант
    bp["emb"] = Tensor(G @ Uplus)                         # (V,V)@(V,d)=(V,d)
    bp["unemb"] = Tensor(U0)                              # (d,V)
    return bp
