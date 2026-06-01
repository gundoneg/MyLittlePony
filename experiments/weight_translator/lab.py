"""Harness для итераций: кешируем зоопарк доноров, гоняем варианты H на нём.

    python lab.py --build --n_train 24 --n_held 8 --donor_iters 180
    python lab.py --run                      # прогнать дефолтный список вариантов
    python lab.py --run --configs structured:inv,linear:inv

Кеш зоопарка в zoo_cache.pkl (донороы — самое дорогое, обучаем один раз).
"""
from __future__ import annotations
import argparse, pickle, os, time, json
import numpy as np
import data, donor, target
from train_zoo import build_zoo
from train_translator import train_translator
from eval import transfer_benchmark, gauge_robustness

CACHE = "zoo_cache.pkl"


def build_cache(n_train, n_held, donor_iters):
    train_tasks, held_tasks = data.make_task_split(0, n_train, n_held)
    print(f"[build] train={n_train} held={n_held} donor_iters={donor_iters}")
    tz = build_zoo(train_tasks, iters=donor_iters, base_seed=1000, verbose=False)
    hz = build_zoo(held_tasks, iters=donor_iters, base_seed=5000, verbose=False)
    meta = dict(n_train=n_train, n_held=n_held, donor_iters=donor_iters)
    with open(CACHE, "wb") as f:
        pickle.dump({"train": tz, "held": hz, "meta": meta}, f)
    print(f"[build] donor acc train={np.mean([z['acc'] for z in tz]):.3f} "
          f"held={np.mean([z['acc'] for z in hz]):.3f} -> {CACHE}")


def load_cache():
    with open(CACHE, "rb") as f:
        d = pickle.load(f)
    return d["train"], d["held"], d["meta"]


def run_config(tz, hz, variant, mode, htrain=500, ft=25, N=12, d_z=8, seed=0, wd=0.0):
    from eval import predict_B, task_acc
    H = train_translator(tz, mode, iters=htrain, tasks_per_step=min(8, len(tz)),
                         variant=variant, d_z=d_z, seed=seed, wd=wd, verbose=False)
    tb = transfer_benchmark(H, hz, ft_steps=ft)
    gr = gauge_robustness(H, hz, N=N)
    rng = np.random.default_rng(11)
    ztr = float(np.mean([task_acc(predict_B(H, z["params"]), z["task"], rng)
                         for z in tz[:min(8, len(tz))]]))
    return {**tb, **gr, "zero_train": ztr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n_train", type=int, default=24)
    ap.add_argument("--n_held", type=int, default=8)
    ap.add_argument("--donor_iters", type=int, default=180)
    ap.add_argument("--htrain", type=int, default=500)
    ap.add_argument("--ft", type=int, default=25)
    ap.add_argument("--d_z", type=int, default=8)
    ap.add_argument("--configs", default="linear:inv,mlp:inv,structured:inv")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.build or not os.path.exists(CACHE):
        build_cache(args.n_train, args.n_held, args.donor_iters)
    if not args.run:
        return

    tz, hz, meta = load_cache()
    print(f"[run] zoo {meta} | random-init базовый общий")
    configs = [c.split(":") for c in args.configs.split(",")]
    rows = {}
    t0 = time.time()
    for variant, mode in configs:
        r = run_config(tz, hz, variant, mode, htrain=args.htrain,
                       ft=args.ft, d_z=args.d_z, wd=args.wd)
        rows[f"{variant}:{mode}"] = r
        print(f"  {variant:11}:{mode:4}  zeroHELD={r['zero_shot']:.3f} "
              f"zeroTRAIN={r['zero_train']:.3f} warm={r['warm_start_ft']:.3f} "
              f"rnd={r['random_ft']:.3f} feat_std={r['feat_std']:.1e}")

    print("\n=========== СВОДКА ===========")
    print(f"{'config':18}{'zero':>7}{'warm':>7}{'random':>8}{'feat_std':>11}{'acc_std':>9}")
    for k, r in rows.items():
        print(f"{k:18}{r['zero_shot']:>7.3f}{r['warm_start_ft']:>7.3f}"
              f"{r['random_ft']:>8.3f}{r['feat_std']:>11.1e}{r['acc_std_under_gauge']:>9.3f}")
    print(f"\n[done] {time.time()-t0:.0f}s")
    if args.out:
        json.dump({"meta": meta, "rows": rows}, open(args.out, "w"),
                  indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
