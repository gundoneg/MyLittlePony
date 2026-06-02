"""Оркестратор: зоопарк -> переводчики (inv/raw/qr) -> бенчмарк. Таблицы + results.json.

    python run.py --smoke     # быстро (десятки секунд)
    python run.py --full      # полный прогон (~минуты)
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import data, donor, target
from train_zoo import build_zoo
from train_translator import train_translator
from eval import transfer_benchmark, gauge_robustness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--modes", default="inv,raw,qr")
    args = ap.parse_args()

    if args.smoke or not args.full:
        cfg_n = dict(n_train=6, n_held=4, donor_iters=120, htrain=150, ft=15, N=8)
        tag = "SMOKE"
    else:
        cfg_n = dict(n_train=32, n_held=8, donor_iters=220, htrain=500, ft=25, N=12)
        tag = "FULL"
    modes = args.modes.split(",")

    print(f"=== weight-translator [{tag}] ===")
    t0 = time.time()
    train_tasks, held_tasks = data.make_task_split(
        seed=0, n_train=cfg_n["n_train"], n_held=cfg_n["n_held"])

    print(f"[1] обучаю зоопарк доноров: train={len(train_tasks)} held={len(held_tasks)}")
    train_zoo = build_zoo(train_tasks, iters=cfg_n["donor_iters"], base_seed=1000)
    held_zoo = build_zoo(held_tasks, iters=cfg_n["donor_iters"], base_seed=5000)
    print(f"    donor acc: train={np.mean([z['acc'] for z in train_zoo]):.3f} "
          f"held={np.mean([z['acc'] for z in held_zoo]):.3f}")

    results = {"tag": tag, "modes": {}}
    for mode in modes:
        print(f"[2] переводчик H (mode={mode})")
        H = train_translator(train_zoo, mode, iters=cfg_n["htrain"],
                             tasks_per_step=min(8, len(train_zoo)))
        tb = transfer_benchmark(H, held_zoo, ft_steps=cfg_n["ft"])
        gr = gauge_robustness(H, held_zoo, N=cfg_n["N"])
        results["modes"][mode] = {**tb, **gr}
        print(f"    zero-shot={tb['zero_shot']:.3f}  warm={tb['warm_start_ft']:.3f}  "
              f"random={tb['random_ft']:.3f}  feat_std={gr['feat_std']:.2e}  "
              f"acc_std_gauge={gr['acc_std_under_gauge']:.3f}")

    # таблицы
    print("\n================  РЕЗУЛЬТАТЫ  ================")
    print("\nA) Перенос на held-out (точность задачи, 0..1):")
    print(f"  {'mode':6} {'zero-shot':>10} {'warm-start':>11} {'random-init':>12}")
    for m in modes:
        r = results["modes"][m]
        print(f"  {m:6} {r['zero_shot']:>10.3f} {r['warm_start_ft']:>11.3f} "
              f"{r['random_ft']:>12.3f}")
    print("\nB) Gauge-робастность (меньше = стабильнее):")
    print(f"  {'mode':6} {'feat_std':>12} {'acc_std_gauge':>15}")
    for m in modes:
        r = results["modes"][m]
        print(f"  {m:6} {r['feat_std']:>12.2e} {r['acc_std_under_gauge']:>15.3f}")
    print("\nИнтерпретация: inv должен иметь feat_std~0 (точная инвариантность),")
    print("raw/qr — заметно выше; warm-start(inv) >= random-init = полезная фора.")

    results["seconds"] = round(time.time() - t0, 1)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[done] {results['seconds']}s -> results.json")


if __name__ == "__main__":
    main()
