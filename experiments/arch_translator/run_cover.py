"""Phase 8 / Fase A — the cheap decisive test: cross-depth transfer of the per-block
translator on the pointer-chasing task, in the FAVORABLE (independent-init + data-align)
regime, with a measured coverage gap.

exp2 (phase 6) trained C on L=2 and applied to L=4 -> near-chance. The user argues the
per-block translator is local, so failure = the shallow zoo did not COVER the deep
layer types. Here we (a) reproduce cross-depth transfer cleanly, (b) measure how well
the shallow per-block features cover the deep ones, and (c) compare to the native-depth
ceiling. For the pure pointer-chase, the prediction is a coverage GAP: an L=2 donor's
blocks only ever do hop-1/hop-2; hop-3/hop-4 are 'deep-only' types absent at depth 2.

Run: python run_cover.py
"""
import argparse, time
import torch

from models import LM
from task import sample_tasks_hops
from donor_zoo import build_donor_zoo
from align import align_zoo
from xlate import Translator, PerLayerTranslator, train_translator
from run_gen import cfg_at, evaluate, probe_inputs, layer_ablation
from coverage import nn_reconstruction_coverage, subspace_overlap_coverage

CHANCE = 1 / 16


def steps_for(L, override=0):
    return override if override else 400 * L


def make_zoo(L, n, seed, realistic, donor_steps=0, maxjump=2):
    cfg = cfg_at(L)
    tasks = sample_tasks_hops(n, cfg.vocab, hops=L, maxjump=maxjump, seed=seed)
    raw = build_donor_zoo(cfg, tasks, steps=steps_for(L, donor_steps), tied=not realistic, verbose=False)
    if realistic:                                  # data-align to donor 0 (phase 5)
        probe = probe_inputs(tasks[0], cfg, 64)
        aligned = align_zoo(raw, raw[0]["A"], "data", cfg=cfg, probe_x=probe)
    else:
        aligned = raw
    return cfg, raw, aligned                        # raw kept for runnable layer-ablation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shallow", type=int, default=2)
    ap.add_argument("--deep", type=int, default=4)
    ap.add_argument("--n_train", type=int, default=10)
    ap.add_argument("--n_held", type=int, default=4)
    ap.add_argument("--donor_steps", type=int, default=0, help="0 => 400*L")
    ap.add_argument("--c_steps", type=int, default=350)
    ap.add_argument("--tps", type=int, default=6)
    ap.add_argument("--warm", type=int, default=15)
    ap.add_argument("--realistic", action="store_true", default=True,
                    help="independent-init + data-align (phase-5 pipeline)")
    ap.add_argument("--tied", dest="realistic", action="store_false")
    args = ap.parse_args()
    t0 = time.time()

    print("=" * 72)
    print(f"PHASE-A cross-depth transfer  L{args.shallow} -> L{args.deep}   "
          f"({'realistic indep+align' if args.realistic else 'tied'})   chance {CHANCE:.1%}")
    print("=" * 72)

    cfg_s, _, shallow = make_zoo(args.shallow, args.n_train + args.n_held, seed=7,
                                 realistic=args.realistic, donor_steps=args.donor_steps)
    s_train, s_held = shallow[:args.n_train], shallow[args.n_train:]
    cfg_d, deep_raw, deep = make_zoo(args.deep, args.n_train + args.n_held, seed=200,
                                     realistic=args.realistic, donor_steps=args.donor_steps)
    d_native, d_target = deep[:args.n_train], deep[args.n_train:]
    print(f"  donor-acc: shallow L{args.shallow} {sum(z['acc'] for z in shallow)/len(shallow):.0%} | "
          f"deep L{args.deep} {sum(z['acc'] for z in deep)/len(deep):.0%}")

    # depth genuinely used in the deep target? (measure on RAW, runnable donors --
    # data-aligned states are not a valid runnable transformer, so ablation must use raw)
    drops = layer_ablation(deep_raw[args.n_train:], cfg_d)
    print(f"  deep-target layer-ablation drop/blk {[f'{d:.0%}' for d in drops]} (on raw donors)")

    # coverage of deep-target per-block features by the shallow TRAIN zoo
    cov_mean, cov_min = nn_reconstruction_coverage(s_train, d_target, cfg_d)
    cov_sub = subspace_overlap_coverage(s_train, d_target, cfg_d)
    print(f"  COVERAGE shallow->deep: NN cosine mean {cov_mean:.3f}  worst-case {cov_min:.3f}  "
          f"| subspace-overlap {cov_sub:.3f}")

    tmpl_s, tmpl_d = LM(cfg_s, "ssm"), LM(cfg_d, "ssm")
    for frac in (False, True):
        tag = "feat+depthfrac" if frac else "feat-only"
        C_shallow, _ = train_translator(cfg_s, s_train, steps=args.c_steps, tasks_per_step=args.tps,
                                        translator_cls=PerLayerTranslator, depth_frac=frac)
        zs_s = evaluate(C_shallow, tmpl_s, cfg_s, s_held, warm_at=args.warm)
        zs_t = evaluate(C_shallow, tmpl_d, cfg_d, d_target, warm_at=args.warm)
        C_native, _ = train_translator(cfg_d, d_native, steps=args.c_steps, tasks_per_step=args.tps,
                                       translator_cls=PerLayerTranslator, depth_frac=frac)
        zs_n = evaluate(C_native, tmpl_d, cfg_d, d_target, warm_at=args.warm)
        print(f"  [{tag}]")
        print(f"    train-depth  L{args.shallow} (sanity)   zs {zs_s[0]:.1%} | warm {zs_s[1]:.1%} | s90 {zs_s[2]:.0f}")
        print(f"    TRANSFER     L{args.shallow}->L{args.deep}      zs {zs_t[0]:.1%} | warm {zs_t[1]:.1%} | s90 {zs_t[2]:.0f}")
        print(f"    native ceil  L{args.deep}->L{args.deep}      zs {zs_n[0]:.1%} | warm {zs_n[1]:.1%} | s90 {zs_n[2]:.0f}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
