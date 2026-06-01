"""Цель B — Mamba-lite: диагональный SSM-микшер (рекуррентность) + SwiGLU + RMSNorm.

Иной оператор смешивания по времени (скан), чем у донора (внимание). Тот же
словарь/интерфейс. Функциональный стиль, как у донора.
"""
from __future__ import annotations
import numpy as np
from autograd import Tensor, rmsnorm, stack
from nn import glorot

CFG = dict(V=12, L=12, d_model=24, d_ff=32, conv_w=4)


def init_params(rng, cfg=CFG):
    V, L, d, dff = cfg["V"], cfg["L"], cfg["d_model"], cfg["d_ff"]
    p = {}
    p["emb"] = glorot(rng, V, d)
    p["pos"] = glorot(rng, L, d) * 0.5
    p["l0_ln1"] = np.ones(d)
    p["l0_Win"] = glorot(rng, d, d)
    W = cfg.get("conv_w", 4)               # короткая причинная depthwise-свёртка (как в Mamba)
    p["l0_conv"] = np.zeros((d, W)); p["l0_conv"][:, 0] = 1.0   # старт = тождество (без задержки)
    p["l0_a_log"] = np.zeros(d)            # a = sigmoid(a_log) ~ 0.5 старт
    p["l0_b"] = np.ones(d) * 0.5
    p["l0_c"] = np.ones(d) * 0.5
    p["l0_d"] = np.zeros(d)                # skip
    p["l0_Wg"] = glorot(rng, d, d)
    p["l0_Wout"] = glorot(rng, d, d)
    p["l0_ln2"] = np.ones(d)
    p["l0_ffn_gate"] = glorot(rng, d, dff)
    p["l0_ffn_up"] = glorot(rng, d, dff)
    p["l0_ffn_down"] = glorot(rng, dff, d)
    p["lnf"] = np.ones(d)
    p["unemb"] = glorot(rng, d, V)
    return p


def _time_slice(t: Tensor, idx):
    """t:(B,L,d) -> (B,d) при времени idx, с backward."""
    data = t.data[:, idx, :]
    out = Tensor(data, requires_grad=t.requires_grad, _children=(t,))
    if out.requires_grad:
        def bw():
            g = np.zeros_like(t.data)
            g[:, idx, :] = out.grad
            Tensor._acc(t, g)
        out._backward = bw
    return out


def _col(t2d: Tensor, j):
    """столбец j двумерного (d,W) Tensor -> (d,), с backward."""
    data = t2d.data[:, j]
    out = Tensor(data, requires_grad=t2d.requires_grad, _children=(t2d,))
    if out.requires_grad:
        def bw():
            g = np.zeros_like(t2d.data)
            g[:, j] = out.grad
            Tensor._acc(t2d, g)
        out._backward = bw
    return out


def forward(pt, x_idx, cfg=CFG):
    V, L, d, dff = cfg["V"], cfg["L"], cfg["d_model"], cfg["d_ff"]
    B, Lx = x_idx.shape
    h = pt["emb"].gather_rows(x_idx) + pt["pos"].reshape(1, L, d)
    # --- diagonal SSM mixer ---
    res = h
    hn = rmsnorm(h, pt["l0_ln1"])
    xin = hn @ pt["l0_Win"]                         # (B,L,d)
    # короткая причинная depthwise-свёртка: xc_t = Σ_j conv[:,j] ⊙ xin_{t-j}
    Wc = pt["l0_conv"].shape[1]
    xc = []
    for t in range(Lx):
        acc = None
        for j in range(Wc):
            if t - j < 0:
                continue
            term = _time_slice(xin, t - j) * _col(pt["l0_conv"], j).reshape(1, d)
            acc = term if acc is None else acc + term
        xc.append(acc)
    xin = stack(xc, axis=1)                          # (B,L,d), с задержкой
    gate = (hn @ pt["l0_Wg"]).silu()                # (B,L,d)
    a = pt["l0_a_log"].sigmoid().reshape(1, d)      # (1,d)
    bcoef = pt["l0_b"].reshape(1, d)
    ccoef = pt["l0_c"].reshape(1, d)
    dskip = pt["l0_d"].reshape(1, d)
    h_state = Tensor(np.zeros((B, d)))
    ys = []
    for t in range(Lx):
        xt = _time_slice(xin, t)                    # (B,d)
        h_state = a * h_state + bcoef * xt
        yt = ccoef * h_state + dskip * xt
        ys.append(yt)
    y = stack(ys, axis=1)                           # (B,L,d)
    y = y * gate
    h = res + (y @ pt["l0_Wout"])
    # --- SwiGLU FFN ---
    res = h
    hn = rmsnorm(h, pt["l0_ln2"])
    g = hn @ pt["l0_ffn_gate"]
    u = hn @ pt["l0_ffn_up"]
    h = res + ((g.silu() * u) @ pt["l0_ffn_down"])
    hf = rmsnorm(h, pt["lnf"])
    return hf @ pt["unemb"]
