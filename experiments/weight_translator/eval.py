"""Бенчмарк: zero-shot перенос, warm-start vs random, gauge-робастность."""
from __future__ import annotations
import numpy as np
from autograd import cross_entropy
from nn import to_tensors, Adam
import target, data, translator
from invariants import featurize
from train_translator import make_features


def predict_B(H, params, augment_rng=None):
    feat = make_features(params, H["mode"], H["mu"], H["sd"], rng=augment_rng)
    tt = to_tensors(H["tp"], requires_grad=False)
    bp = translator.translate(tt, feat, H["spec"])
    return {k: v.data.copy() for k, v in bp.items()}


def task_acc(bp_np, task, rng, cfg=target.CFG, n=256):
    x, y = data.sample_batch(task, rng, batch=n, V=cfg["V"], L=cfg["L"])
    lg = target.forward(to_tensors(bp_np, False), x, cfg)
    V = lg.shape[-1]
    pred = lg.reshape(-1, V).data.argmax(-1)
    return float((pred == y.reshape(-1)).mean())


def finetune(bp_np, task, rng, steps, lr=5e-3, cfg=target.CFG, batch=48):
    p = {k: v.copy() for k, v in bp_np.items()}
    opt = Adam(p, lr=lr)
    for _ in range(steps):
        x, y = data.sample_batch(task, rng, batch=batch, V=cfg["V"], L=cfg["L"])
        pt = to_tensors(p, True)
        lg = target.forward(pt, x, cfg)
        V = lg.shape[-1]
        loss = cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        loss.backward()
        opt.step(p, {k: pt[k].grad for k in p})
    return task_acc(p, task, rng, cfg)


def transfer_benchmark(H, held_zoo, ft_steps=20, cfg=target.CFG, seed=0):
    rng = np.random.default_rng(seed)
    zs, ws, rnd = [], [], []
    for z in held_zoo:
        bp = predict_B(H, z["params"])
        zs.append(task_acc(bp, z["task"], rng, cfg))
        ws.append(finetune(bp, z["task"], rng, ft_steps, cfg=cfg))
        r0 = target.init_params(np.random.default_rng(int(rng.integers(1e9))), cfg)
        rnd.append(finetune(r0, z["task"], rng, ft_steps, cfg=cfg))
    return {"zero_shot": float(np.mean(zs)),
            "warm_start_ft": float(np.mean(ws)),
            "random_ft": float(np.mean(rnd))}


def gauge_robustness(H, held_zoo, N=12, cfg=target.CFG, seed=0):
    """Под N случайными симметриями донора: разброс признаков и качества B'."""
    rng = np.random.default_rng(seed)
    feat_stds, loss_stds = [], []
    for z in held_zoo:
        feats, accs = [], []
        for _ in range(N):
            r = np.random.default_rng(int(rng.integers(1e9)))
            feats.append(make_features(z["params"], H["mode"], H["mu"], H["sd"], rng=r))
            bp = predict_B(H, z["params"], augment_rng=np.random.default_rng(
                int(rng.integers(1e9))))
            accs.append(task_acc(bp, z["task"], rng, cfg, n=128))
        feats = np.stack(feats)
        feat_stds.append(float(feats.std(0).mean()))   # средний разброс призн.
        loss_stds.append(float(np.std(accs)))          # разброс качества B'
    return {"feat_std": float(np.mean(feat_stds)),
            "acc_std_under_gauge": float(np.mean(loss_stds))}
