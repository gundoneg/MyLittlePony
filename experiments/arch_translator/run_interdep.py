"""Phase 8 / Phase B — the layer-interdependence sweep. Does a DIVERSE shallow zoo
manufacture coverage of a DEEP target, and how does that depend on how recursive the
task is (alpha)? Per alpha: train C on a NARROW shallow zoo vs a DIVERSE shallow zoo,
zero-shot to a deep target; report measured coverage, the native-depth ceiling, and the
deep-target depth-usage (raw-donor ablation). Locate the boundary alpha* where shallow
coverage stops sufficing.

Run: python run_interdep.py --alpha 0,0.25,0.5,0.75,1.0
"""
import argparse, time
import torch

from models import Cfg, LM
from task import sample_tasks_interdep
from donor_zoo import build_donor_zoo
from align import align_zoo
from xlate import PerLayerTranslator, train_translator
from run_gen import evaluate, probe_inputs, layer_ablation
from coverage import nn_reconstruction_coverage, subspace_overlap_coverage

CHANCE = 1 / 16


def cfg_at(L, d_model):
    return Cfg(vocab=16, d_model=d_model, n_layer=L, n_head=4, ctx=32, d_ff=256)


def steps_for(L, override=0):
    return override if override else 400 * L


def build(cfg, tasks, realistic, donor_steps):
    raw = build_donor_zoo(cfg, tasks, steps=steps_for(cfg.n_layer, donor_steps),
                          tied=not realistic, verbose=False)
    if realistic:
        probe = probe_inputs(tasks[0], cfg, 64)
        return raw, align_zoo(raw, raw[0]["A"], "data", cfg=cfg, probe_x=probe)
    return raw, raw


def narrow_tasks(n, cfg, L, alpha, seed):
    """Homogeneous shallow zoo: every donor same structure (maxjump=2)."""
    return sample_tasks_interdep(n, cfg.vocab, depth=L, alpha=alpha, maxjump=2, seed=seed)


def diverse_tasks(n, cfg, L, alpha, seed):
    """Diverse shallow zoo: vary maxjump across donors to broaden the per-block feature
    spread (the user's 'cover the type repertoire with small models')."""
    tasks = []
    for j in range(n):
        mj = 1 + (j % 3)                                   # maxjump in {1,2,3}
        tasks += sample_tasks_interdep(1, cfg.vocab, depth=L, alpha=alpha,
                                       maxjump=mj, seed=seed + 1000 * j)
    return tasks


def train_eval(cfg_tr, train_zoo, cfg_d, target, warm, c_steps, frac=True):
    C, _ = train_translator(cfg_tr, train_zoo, steps=c_steps, tasks_per_step=6,
                            translator_cls=PerLayerTranslator, depth_frac=frac)
    return evaluate(C, LM(cfg_d, "ssm"), cfg_d, target, warm_at=warm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--shallow", type=int, default=2)
    ap.add_argument("--deep", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_train", type=int, default=10)
    ap.add_argument("--n_held", type=int, default=4)
    ap.add_argument("--donor_steps", type=int, default=0)
    ap.add_argument("--c_steps", type=int, default=350)
    ap.add_argument("--warm", type=int, default=15)
    ap.add_argument("--realistic", action="store_true", default=True)
    ap.add_argument("--tied", dest="realistic", action="store_false")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alpha.split(",")]
    cfg_s = cfg_at(args.shallow, args.d_model)
    cfg_d = cfg_at(args.deep, args.d_model)
    t0 = time.time()

    print("=" * 100)
    print(f"PHASE-B interdependence sweep  shallow L{args.shallow} -> deep L{args.deep}  "
          f"d_model={args.d_model}  ({'indep+align' if args.realistic else 'tied'})  chance {CHANCE:.1%}")
    print(f"{'alpha':>6} | {'deep-ablation':>22} | {'cov(narrow/div) mean,worst':>30} | "
          f"{'zs narrow':>10} {'zs diverse':>10} {'zs native':>10}")
    print("=" * 100)

    for a in alphas:
        narrow = build(cfg_s, narrow_tasks(args.n_train, cfg_s, args.shallow, a, 7), args.realistic, args.donor_steps)[1]
        diverse = build(cfg_s, diverse_tasks(args.n_train, cfg_s, args.shallow, a, 11), args.realistic, args.donor_steps)[1]
        deep_raw, deep = build(cfg_d, sample_tasks_interdep(args.n_train + args.n_held, cfg_d.vocab,
                               depth=args.deep, alpha=a, maxjump=2, seed=200), args.realistic, args.donor_steps)
        d_native, d_target = deep[:args.n_train], deep[args.n_train:]
        drops = layer_ablation(deep_raw[args.n_train:], cfg_d)

        cnm, cnw = nn_reconstruction_coverage(narrow, d_target, cfg_d)
        cdm, cdw = nn_reconstruction_coverage(diverse, d_target, cfg_d)
        zs_n = train_eval(cfg_s, narrow, cfg_d, d_target, args.warm, args.c_steps)[0]
        zs_d = train_eval(cfg_s, diverse, cfg_d, d_target, args.warm, args.c_steps)[0]
        zs_nat = train_eval(cfg_d, d_native, cfg_d, d_target, args.warm, args.c_steps)[0]

        abl = ",".join(f"{d:.0%}" for d in drops)
        print(f"{a:>6.2f} | {abl:>22} | n {cnm:.2f}/{cnw:.2f}  d {cdm:.2f}/{cdw:.2f} | "
              f"{zs_n:>9.1%} {zs_d:>9.1%} {zs_nat:>9.1%}")
    print(f"\ntotal {time.time()-t0:.0f}s  (chance {CHANCE:.1%})")


if __name__ == "__main__":
    main()
