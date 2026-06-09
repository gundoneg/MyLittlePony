"""Phase 10 / E0 step 6 — re-run cross-depth transfer with the pipeline bugs fixed, on the
SAME raw donors, to decide whether phase-8's negative verdict was an artifact.

Fixes vs run_cover/run_interdep:
  A (align.py)  rotate EVERY block's qkv -> per_block_feature is gauge-invariant for l>=1.
  B (here)      align ALL depths to ONE common reference (shallow donor 0) via full-residual
                Procrustes (depth-independent, spans all d) instead of a per-depth reference,
                so the equivariant vocab maps W_t/W_h/W_p share one frame across depths.
  C (xlate)     scale_free features: unit-direction + log|feature| scalar, so the deepest
                layer's larger feature norm is not an OOD shift for the shallow-fit encoder.

Decisive case alpha=0: the deep target barely uses depth (its computation is within the
shallow repertoire), so a correct pipeline MUST transfer. Old result there was 14.8% vs
native 99.8% -- a glaring failure that points at the pipeline, not depth-entanglement.

Control (Finding 3): interior-feature mismatch -- emit a target with its OWN vocab rows but
ANOTHER donor's qkv (hence another per-block feature). If accuracy is unchanged, C ignores
the features and 'coverage' is causally irrelevant.

Run: python run_cover_fixed.py --alpha 0,1
"""
import argparse, time
import torch

from models import LM
from task import sample_tasks_interdep
from donor_zoo import build_donor_zoo
from align import procrustes, residual_activations, _apply_cols, align_zoo
from xlate import (PerLayerTranslator, train_translator, eval_translator, per_block_feature,
                   _donor_n_layer)
from run_interdep import cfg_at
from run_gen import evaluate, probe_inputs, layer_ablation
from coverage import nn_reconstruction_coverage

CHANCE = 1 / 16
QKV = lambda k: k.startswith("blocks.") and k.endswith(".mix.qkv.weight")


def align_all_to_ref(raw_zoo, cfg, ref_resid, probe):
    """Fix B: each donor -> common reference frame via full-residual Procrustes."""
    out = []
    for z in raw_zoo:
        Q = procrustes(residual_activations(z["A"], probe, cfg), ref_resid)
        out.append(dict(task=z["task"], A=_apply_cols(z["A"], lambda W: W @ Q), acc=z["acc"]))
    return out


def feat_mismatch_zoo(zoo):
    """Each donor keeps its vocab rows (tok/pos/head) but takes the NEXT donor's qkv,
    so per_block_feature changes while the vocab path does not."""
    out = []
    for j, z in enumerate(zoo):
        donor = dict(z["A"])
        src = zoo[(j + 1) % len(zoo)]["A"]
        for k in donor:
            if QKV(k):
                donor[k] = src[k]
        out.append(dict(task=z["task"], A=donor, acc=z["acc"]))
    return out


def zs(C, cfg_d, target):
    return eval_translator(C, LM(cfg_d, "ssm"), cfg_d, target)


def run_alpha(a, args):
    cfg_s, cfg_d = cfg_at(args.shallow, 64), cfg_at(args.deep, 64)
    n = args.n_train + args.n_held
    s_tasks = sample_tasks_interdep(n, cfg_s.vocab, depth=args.shallow, alpha=a, maxjump=2, seed=7)
    d_tasks = sample_tasks_interdep(n, cfg_d.vocab, depth=args.deep, alpha=a, maxjump=2, seed=200)
    tied = args.tied
    s_raw = build_donor_zoo(cfg_s, s_tasks, steps=400 * args.shallow, tied=tied, verbose=False)
    d_raw = build_donor_zoo(cfg_d, d_tasks, steps=400 * args.deep, tied=tied, verbose=False)
    drops = layer_ablation(d_raw[args.n_train:], cfg_d)
    print(f"\nalpha={a}  ({'TIED (shared frame, no align)' if tied else 'indep + common-frame align'})"
          f"  donor-acc s {sum(z['acc'] for z in s_raw)/n:.0%} d {sum(z['acc'] for z in d_raw)/n:.0%}"
          f"  deep-ablation {[f'{x:.0%}' for x in drops]}")
    tkw = dict(translator_cls=PerLayerTranslator, depth_frac=True, scale_free=True)

    if tied:                              # shared frame by construction -> no alignment
        s_fix, d_fix = s_raw, d_raw
        old_t, cov_old = None, (0.0, 0.0)
    else:                                 # independent init: align everything to ONE frame
        probe_s = probe_inputs(s_tasks[0], cfg_s, 64)
        probe_d = probe_inputs(d_tasks[0], cfg_d, 64)
        s_old = align_zoo(s_raw, s_raw[0]["A"], "data", cfg=cfg_s, probe_x=probe_s)
        d_old = align_zoo(d_raw, d_raw[0]["A"], "data", cfg=cfg_d, probe_x=probe_d)
        C_old, _ = train_translator(cfg_s, s_old[:args.n_train], steps=args.c_steps,
                                    tasks_per_step=6, translator_cls=PerLayerTranslator, depth_frac=True)
        old_t = zs(C_old, cfg_d, d_old[args.n_train:])
        cov_old = nn_reconstruction_coverage(s_old[:args.n_train], d_old[args.n_train:], cfg_d)
        ref_resid = residual_activations(s_raw[0]["A"], probe_s, cfg_s)
        s_fix = align_all_to_ref(s_raw, cfg_s, ref_resid, probe_s)
        d_fix = align_all_to_ref(d_raw, cfg_d, ref_resid, probe_s)

    cov_fix = nn_reconstruction_coverage(s_fix[:args.n_train], d_fix[args.n_train:], cfg_d)
    C_fix, _ = train_translator(cfg_s, s_fix[:args.n_train], steps=args.c_steps, tasks_per_step=6, **tkw)
    fix_s = zs(C_fix, cfg_s, s_fix[args.n_train:])
    fix_t = zs(C_fix, cfg_d, d_fix[args.n_train:])
    C_nat, _ = train_translator(cfg_d, d_fix[:args.n_train], steps=args.c_steps, tasks_per_step=6, **tkw)
    fix_n = zs(C_nat, cfg_d, d_fix[args.n_train:])
    fix_n_mm = zs(C_nat, cfg_d, feat_mismatch_zoo(d_fix[args.n_train:]))

    print(f"  coverage(worst-case) {cov_fix[1]:.3f}")
    if old_t is not None:
        print(f"  OLD(per-depth align)  transfer L{args.shallow}->L{args.deep}  zs {old_t:.1%}")
    print(f"  FIXED  sanity  L{args.shallow}->L{args.shallow}  zs {fix_s:.1%}")
    print(f"  FIXED  transfer L{args.shallow}->L{args.deep}  zs {fix_t:.1%}   (native ceil {fix_n:.1%}, "
          f"transfer/native {fix_t/max(fix_n,1e-9):.0%})")
    print(f"  CONTROL native interior-feature-mismatch zs {fix_n_mm:.1%} "
          f"(if ~= native {fix_n:.1%}, features are NOT used)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", default="0,1")
    ap.add_argument("--shallow", type=int, default=2)
    ap.add_argument("--deep", type=int, default=3)
    ap.add_argument("--n_train", type=int, default=8)
    ap.add_argument("--n_held", type=int, default=4)
    ap.add_argument("--c_steps", type=int, default=300)
    ap.add_argument("--tied", action="store_true",
                    help="shared-init donors (common frame by construction, no alignment)")
    args = ap.parse_args()
    t0 = time.time()
    print("=" * 78)
    print(f"E0 fixed cross-depth transfer  L{args.shallow}->L{args.deep}  chance {CHANCE:.1%}")
    print("=" * 78)
    for a in [float(x) for x in args.alpha.split(",")]:
        run_alpha(a, args)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
