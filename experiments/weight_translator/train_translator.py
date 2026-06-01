"""Обучение переводчика H поведенческим лоссом. Донор НЕ запускается.

Для задачи tau_i: feat = standardize(featurize(A_i, mode)); theta_B = H(feat);
гоняем target.forward(theta_B) на данных tau_i; CE. Сумма по подвыборке задач.
gauge-аугментация: на входе к донору применяем случайную симметрию (для всех
режимов одинаково — честное сравнение).
"""
from __future__ import annotations
import numpy as np
from autograd import cross_entropy, Tensor
from nn import to_tensors, Adam, flatten
import target, data, translator
from invariants import featurize, gauge_transform


def compute_standardizer(zoo, mode):
    F = np.stack([featurize(z["params"], mode) for z in zoo])
    mu = F.mean(0); sd = F.std(0) + 1e-6
    return mu, sd


def make_features(params, mode, mu, sd, rng=None):
    p = gauge_transform(params, rng) if rng is not None else params
    return (featurize(p, mode) - mu) / sd


def train_translator(zoo, mode, cfg=target.CFG, iters=400, lr=3e-3, d_z=8,
                     tasks_per_step=8, augment=True, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    spec, P = translator.b_spec(cfg)
    mu, sd = compute_standardizer(zoo, mode)
    F = (featurize(zoo[0]["params"], mode)).shape[0]
    theta0 = flatten(target.init_params(np.random.default_rng(123), cfg))[0]
    tp = translator.init_translator(rng, F, P, d_z=d_z, theta0=theta0)
    opt = Adam(tp, lr=lr)

    n = len(zoo)
    for it in range(iters):
        idx = rng.choice(n, size=min(tasks_per_step, n), replace=False)
        tt = to_tensors(tp, requires_grad=True)
        loss = None
        for j in idx:
            z = zoo[j]
            feat = make_features(z["params"], mode, mu, sd,
                                 rng=rng if augment else None)
            bp = translator.translate(tt, feat, spec)
            x, y = data.sample_batch(z["task"], rng, batch=48,
                                     V=cfg["V"], L=cfg["L"])
            lg = target.forward(bp, x, cfg)
            V = lg.shape[-1]
            l = cross_entropy(lg.reshape(-1, V), y.reshape(-1))
            loss = l if loss is None else loss + l
        loss = loss * (1.0 / len(idx))
        loss.backward()
        opt.step(tp, {k: tt[k].grad for k in tp})
        if verbose and (it + 1) % max(1, iters // 5) == 0:
            print(f"    [{mode}] it={it+1:4d} loss={float(loss.data):.3f}")
    return {"tp": tp, "spec": spec, "mu": mu, "sd": sd, "mode": mode, "cfg": cfg}
