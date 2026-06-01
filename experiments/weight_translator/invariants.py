"""Кодировщики весов донора в признаки + gauge-преобразования.

Три режима признаков для ablation:
  - 'inv'  : gauge-инварианты (M_QK, M_OV, Грам-матрицы экспертов, router-Грам)
             + gauge-свободные emb/pos/unemb. Точно инвариантны к нашим gauge-ops.
  - 'raw'  : все веса донора как есть (меняются под gauge).
  - 'qr'   : QR-«канонизация» голов (имитирует gauge-fixing; численно неустойчива).

gauge-ops, точно сохраняющие функцию донора:
  перестановка голов; per-head QK-вращение (Wq->WqR, Wk->WkR^{-T});
  per-head VO-вращение (Wv->WvS, Wo->S^{-1}Wo); перестановка экспертов;
  перестановка скрытых нейронов эксперта.
"""
from __future__ import annotations
import numpy as np
from donor import CFG


def _blocks(d, H, dh):
    return [slice(h * dh, (h + 1) * dh) for h in range(H)]


# ---------------------------------------------------------------- признаки
def featurize(p, mode="inv", cfg=CFG):
    d, H, dh, E, V, L = (cfg["d_model"], cfg["n_head"], cfg["d_head"],
                         cfg["n_exp"], cfg["V"], cfg["L"])
    blk = _blocks(d, H, dh)
    if mode == "raw":
        keys = sorted(p.keys())
        return np.concatenate([p[k].reshape(-1) for k in keys])

    if mode == "qr":
        # «канонизация»: ортонормировать head-блоки Wq,Wk через QR (gauge-fix R).
        Wq, Wk = p["l0_Wq"].copy(), p["l0_Wk"].copy()
        feat = []
        for s in blk:
            Q, R = np.linalg.qr(Wq[:, s])
            feat.append(Q.reshape(-1))
            feat.append((Wk[:, s] @ R.T).reshape(-1))
        base = [p[k].reshape(-1) for k in sorted(p.keys())
                if k not in ("l0_Wq", "l0_Wk")]
        return np.concatenate(feat + base)

    if mode == "kpos":
        # σ-РАЗВЯЗАННЫЙ признак: только позиционные логиты (несут лаг k), без G/σ.
        MQK = np.zeros((d, d)); MOV = np.zeros((d, d))
        for s in blk:
            MQK += p["l0_Wq"][:, s] @ p["l0_Wk"][:, s].T
            MOV += p["l0_Wv"][:, s] @ p["l0_Wo"][s, :]
        Spos = p["pos"] @ MQK @ p["pos"].T              # (L,L) — лаг k
        Sov = p["pos"] @ MOV @ p["pos"].T               # (L,L)
        return np.concatenate([Spos.reshape(-1), Sov.reshape(-1)])

    if mode == "inv2":
        # базис-инвариантные признаки задачи (инвар. к глоб. сопряжению M->P^T M P)
        MQK = np.zeros((d, d)); MOV = np.zeros((d, d))
        for s in blk:
            MQK += p["l0_Wq"][:, s] @ p["l0_Wk"][:, s].T
            MOV += p["l0_Wv"][:, s] @ p["l0_Wo"][s, :]
        G = p["emb"] @ p["unemb"]                       # (V,V) — несёт sigma
        Spos = p["pos"] @ MQK @ p["pos"].T              # (L,L) — несёт лаг k
        Sov = p["pos"] @ MOV @ p["pos"].T               # (L,L)
        sv_qk = np.linalg.svd(MQK, compute_uv=False)    # инвар. к ортогон. сопряжению
        sv_ov = np.linalg.svd(MOV, compute_uv=False)
        ge = np.zeros((d, d)); ue = np.zeros((d, d)); de = np.zeros((d, d))
        for e in range(E):
            g = p[f"l0_e{e}_gate"]; u = p[f"l0_e{e}_up"]; dn = p[f"l0_e{e}_down"]
            ge += g @ g.T; ue += u @ u.T; de += dn.T @ dn
        sv_e = np.concatenate([np.linalg.svd(m, compute_uv=False)
                               for m in (ge, ue, de)])
        return np.concatenate([G.reshape(-1), Spos.reshape(-1), Sov.reshape(-1),
                               sv_qk, sv_ov, sv_e])

    # mode == 'inv'
    feat = []
    MQK = np.zeros((d, d)); MOV = np.zeros((d, d))
    for s in blk:
        MQK += p["l0_Wq"][:, s] @ p["l0_Wk"][:, s].T        # инвар. к QK-вращению
        MOV += p["l0_Wv"][:, s] @ p["l0_Wo"][s, :]          # инвар. к VO-вращению
    feat += [MQK.reshape(-1), MOV.reshape(-1)]
    gg = np.zeros((d, d)); uu = np.zeros((d, d)); dd = np.zeros((d, d))
    for e in range(E):
        g = p[f"l0_e{e}_gate"]; u = p[f"l0_e{e}_up"]; dn = p[f"l0_e{e}_down"]
        gg += g @ g.T; uu += u @ u.T; dd += dn.T @ dn            # инвар. к hidden-perm
    feat += [gg.reshape(-1), uu.reshape(-1), dd.reshape(-1)]
    feat.append((p["l0_router"] @ p["l0_router"].T).reshape(-1))  # инвар. к expert-perm
    # gauge-свободные (несут task-identity sigma)
    feat += [p["emb"].reshape(-1), p["unemb"].reshape(-1), p["pos"].reshape(-1)]
    return np.concatenate(feat)


def feat_dim(mode="inv", cfg=CFG):
    import numpy as np
    dummy = _zeros_params(cfg)
    return featurize(dummy, mode, cfg).shape[0]


def _zeros_params(cfg):
    from donor import init_params
    return init_params(np.random.default_rng(0), cfg)


# ------------------------------------------------------------ gauge-ops
def _rand_orth(rng, n):
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))           # детерминируем знак
    return Q


def gauge_transform(p, rng, cfg=CFG):
    d, H, dh, E, dff = (cfg["d_model"], cfg["n_head"], cfg["d_head"],
                        cfg["n_exp"], cfg["d_ff"])
    blk = _blocks(d, H, dh)
    q = {k: v.copy() for k, v in p.items()}

    # per-head QK/VO вращения
    for s in blk:
        R = _rand_orth(rng, dh); S = _rand_orth(rng, dh)
        q["l0_Wq"][:, s] = p["l0_Wq"][:, s] @ R
        q["l0_Wk"][:, s] = p["l0_Wk"][:, s] @ np.linalg.inv(R).T
        q["l0_Wv"][:, s] = p["l0_Wv"][:, s] @ S
        q["l0_Wo"][s, :] = np.linalg.inv(S) @ p["l0_Wo"][s, :]

    # перестановка голов (блоки колонок Wq,Wk,Wv; блоки строк Wo)
    hp = rng.permutation(H)
    Wq, Wk, Wv, Wo = (q["l0_Wq"].copy(), q["l0_Wk"].copy(),
                      q["l0_Wv"].copy(), q["l0_Wo"].copy())
    for newh, oldh in enumerate(hp):
        ns, os = blk[newh], blk[oldh]
        q["l0_Wq"][:, ns] = Wq[:, os]; q["l0_Wk"][:, ns] = Wk[:, os]
        q["l0_Wv"][:, ns] = Wv[:, os]; q["l0_Wo"][ns, :] = Wo[os, :]

    # перестановка экспертов (+ колонок router) и hidden-перестановки
    ep = rng.permutation(E)
    router = q["l0_router"].copy()
    new_experts = {}
    for newe, olde in enumerate(ep):
        hp2 = rng.permutation(dff)
        new_experts[f"l0_e{newe}_gate"] = p[f"l0_e{olde}_gate"][:, hp2]
        new_experts[f"l0_e{newe}_up"] = p[f"l0_e{olde}_up"][:, hp2]
        new_experts[f"l0_e{newe}_down"] = p[f"l0_e{olde}_down"][hp2, :]
        q["l0_router"][:, newe] = router[:, olde]
    q.update(new_experts)
    return q


# ------------------------------------------------------------ self-check
def verify(seed=0):
    import numpy as np
    from nn import to_tensors
    import donor, data
    rng = np.random.default_rng(seed)
    p = donor.init_params(rng)
    x, _ = data.sample_batch(data.make_task(rng), rng, batch=8)
    lg0 = donor.forward(to_tensors(p, False), x).data
    q = gauge_transform(p, rng)
    lg1 = donor.forward(to_tensors(q, False), x).data
    fdiff = np.max(np.abs(lg0 - lg1))
    inv0 = featurize(p, "inv"); inv1 = featurize(q, "inv")
    raw0 = featurize(p, "raw"); raw1 = featurize(q, "raw")
    idiff = np.max(np.abs(inv0 - inv1))
    rdiff = np.max(np.abs(raw0 - raw1))
    print(f"gauge preserves donor logits : max|Δ|={fdiff:.2e}  (должно ~0)")
    print(f"invariant features Δ          : {idiff:.2e}  (должно ~0)")
    print(f"raw features Δ                : {rdiff:.2e}  (должно >> 0)")
    assert fdiff < 1e-8, "gauge-op НЕ сохраняет функцию!"
    assert idiff < 1e-8, "инварианты НЕ инвариантны!"
    assert rdiff > 1e-2, "raw неожиданно инвариантны"
    print("OK: gauge-ops — симметрии; инварианты инвариантны; raw — нет.")


if __name__ == "__main__":
    verify()
