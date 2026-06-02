"""Донор A — «Kimi-подобный» микро-декодер: causal MHA + soft-MoE(SwiGLU) + RMSNorm.

Функциональный стиль: init_params -> dict[np], forward(dict[Tensor], x) -> logits.
Позиционные эмбеддинги — обучаемые абсолютные (вместо RoPE) ради чистоты QK-инвариантов.
MoE — мягкий гейтинг (дифференцируемая релаксация top-k), эквивариантный к
перестановке экспертов.
"""
from __future__ import annotations
import numpy as np
from autograd import Tensor, rmsnorm
from nn import glorot

CFG = dict(V=12, L=12, d_model=24, n_head=2, d_head=12, d_ff=32, n_exp=4)


def init_params(rng, cfg=CFG):
    V, L, d, H, dh, dff, E = (cfg["V"], cfg["L"], cfg["d_model"], cfg["n_head"],
                              cfg["d_head"], cfg["d_ff"], cfg["n_exp"])
    assert H * dh == d
    p = {}
    p["emb"] = glorot(rng, V, d)
    p["pos"] = glorot(rng, L, d) * 0.5
    p["l0_ln1"] = np.ones(d)
    for nm in ("Wq", "Wk", "Wv", "Wo"):
        p[f"l0_{nm}"] = glorot(rng, d, d)
    p["l0_ln2"] = np.ones(d)
    p["l0_router"] = glorot(rng, d, E)
    for e in range(E):
        p[f"l0_e{e}_gate"] = glorot(rng, d, dff)
        p[f"l0_e{e}_up"] = glorot(rng, d, dff)
        p[f"l0_e{e}_down"] = glorot(rng, dff, d)
    p["lnf"] = np.ones(d)
    p["unemb"] = glorot(rng, d, V)
    return p


def _causal_mask(L):
    m = np.triu(np.ones((L, L)), k=1) * (-1e9)
    return m.reshape(1, 1, L, L)


def forward(pt, x_idx, cfg=CFG):
    """pt: dict[str,Tensor]; x_idx: (B,L) int. -> logits Tensor (B,L,V)."""
    V, L, d, H, dh, E = (cfg["V"], cfg["L"], cfg["d_model"], cfg["n_head"],
                         cfg["d_head"], cfg["n_exp"])
    B = x_idx.shape[0]
    Lx = x_idx.shape[1]
    h = pt["emb"].gather_rows(x_idx)               # (B,L,d)
    h = h + pt["pos"].reshape(1, L, d)
    # --- attention ---
    res = h
    hn = rmsnorm(h, pt["l0_ln1"])
    q = (hn @ pt["l0_Wq"]).reshape(B, Lx, H, dh).transpose((0, 2, 1, 3))  # (B,H,L,dh)
    k = (hn @ pt["l0_Wk"]).reshape(B, Lx, H, dh).transpose((0, 2, 1, 3))
    v = (hn @ pt["l0_Wv"]).reshape(B, Lx, H, dh).transpose((0, 2, 1, 3))
    scores = (q @ k.transpose((0, 1, 3, 2))) * (1.0 / np.sqrt(dh))         # (B,H,L,L)
    scores = scores + Tensor(_causal_mask(Lx))
    att = scores.softmax(axis=-1)
    ctx = (att @ v).transpose((0, 2, 1, 3)).reshape(B, Lx, d)              # (B,L,d)
    h = res + (ctx @ pt["l0_Wo"])
    # --- soft MoE ---
    res = h
    hn = rmsnorm(h, pt["l0_ln2"])
    gate_logits = hn @ pt["l0_router"]            # (B,L,E)
    gate = gate_logits.softmax(axis=-1)
    moe = None
    for e in range(E):
        g = hn @ pt[f"l0_e{e}_gate"]
        u = hn @ pt[f"l0_e{e}_up"]
        out_e = (g.silu() * u) @ pt[f"l0_e{e}_down"]   # (B,L,d)
        w = _slice_last(gate, e).reshape(B, Lx, 1)     # (B,L,1)
        term = out_e * w
        moe = term if moe is None else moe + term
    h = res + moe
    hf = rmsnorm(h, pt["lnf"])
    logits = hf @ pt["unemb"]                      # (B,L,V)
    return logits


def _slice_last(t: Tensor, e):
    """t[...,e] как Tensor (с backward). t: (...,E)."""
    data = t.data[..., e]
    out = Tensor(data, requires_grad=t.requires_grad, _children=(t,))
    if out.requires_grad:
        def bw():
            g = np.zeros_like(t.data)
            g[..., e] = out.grad
            Tensor._acc(t, g)
        out._backward = bw
    return out
