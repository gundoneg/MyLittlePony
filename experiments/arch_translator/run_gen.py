"""Phase 6 — solve the GENERATION of deep target interior with a weight-TIED per-block
hypernet (one shared 'block translator' reused at every depth).

The monolithic Translator (xlate.Translator) emits all L blocks' SSM interior from a
single layer-0 code through one big matrix M -> zero-shot transfer DECAYS with depth
(README_scale: 84% @L1 -> 76% @L2 -> 67% @L4). PerLayerTranslator instead generates
each block l from a per-block feature of donor A's block l, through a SHARED encoder
and a SHARED M_block. Its parameter count is independent of depth.

Three experiments (user picked BOTH isolation regimes + the depth-transfer demo):

  exp1  isolation : tied-init donors, DIFFERENT (sigma,k) per donor; sweep L; compare
                    mono vs perlayer on zero-shot / warm@15 / steps-to-90. Does the
                    per-block design FLATTEN the depth decay?
  exp2  transfer  : train ONE perlayer C on shallow (L=2) donors, apply zero-shot to
                    DEEP held donors (L=4,6,8). The monolith cannot do this at all
                    (its M is sized to a fixed depth). This is the headline payoff.
  exp3  realistic : independent-init + data_align (phase 5) THEN perlayer generation,
                    swept over depth -- the basis-extraction and the generation fix
                    working together on the hard (no shared frame) regime.

Control (exp1): layer-ablation -- neutralise block l in a donor and measure the task
accuracy drop, to check deep blocks actually do work (so a perlayer 'win' at depth is
not just 'deep blocks are idle').

Toy scale (V=16, d=64); what is portable is the MECHANISM, not the absolute numbers.
"""
import argparse
import time
import torch

from models import Cfg, LM
from task import sample_tasks, sample_tasks_hops, make_batch, task_accuracy
from donor_zoo import build_donor_zoo
from align import align_zoo
from xlate import (Translator, PerLayerTranslator, train_translator, eval_translator,
                   warm_curve, steps_to_threshold, run_B)

CHANCE = 1 / 16


def cfg_at(L):
    return Cfg(vocab=16, d_model=64, n_layer=L, n_head=4, ctx=32, d_ff=256)


def make_tasks(N, cfg, L, args, seed):
    """Sample N tasks for depth L. The 'hops' task is depth-demanding (chain length
    = depth), so deep blocks are exercised; (sigma,k) is the original single-gather."""
    if getattr(args, "task", "sigmak") == "hops":
        return sample_tasks_hops(N, cfg.vocab, hops=L, maxjump=args.maxjump, seed=seed)
    return sample_tasks(N, cfg.vocab, seed=seed)


def donor_steps_for(L, args):
    """Pointer chains of length L need more training the deeper they get."""
    if getattr(args, "task", "sigmak") == "hops":
        return max(args.donor_steps, 400 * L)
    return args.donor_steps


def probe_inputs(task, cfg, n, seed=12345):
    g = torch.Generator().manual_seed(seed)
    x, _ = make_batch(task, n, cfg.ctx, cfg.vocab, g)
    return x


def mean_warm(C, template, cfg, held, max_steps=30, every=5, warm_at=15):
    """Mean over held donors of {warm@warm_at accuracy, steps-to-90}."""
    warm, s90 = [], []
    for z in held:
        params = C.emit(z["A"])
        curve = warm_curve(template, params, cfg, z["task"], max_steps=max_steps, every=every)
        # nearest recorded step >= warm_at
        wkey = min((s for s in curve if s >= warm_at), default=max_steps)
        warm.append(curve[wkey])
        s90.append(steps_to_threshold(curve, 0.9, cap=max_steps + every))
    return sum(warm) / len(warm), sum(s90) / len(s90)


def evaluate(C, template, cfg, held, warm_at=15):
    zs = eval_translator(C, template, cfg, held)
    wm, s90 = mean_warm(C, template, cfg, held, warm_at=warm_at)
    return zs, wm, s90


# --------------------------- exp1: isolation, mono vs perlayer ---------------------------
def neutralise_block(state, l):
    """Make block l a near-identity (residual pass-through): zero its two output
    projections. Used to measure how much work block l does in the donor."""
    out = {k: v.clone() for k, v in state.items()}
    for k in (f"blocks.{l}.mix.o.weight", f"blocks.{l}.mix.o.bias",
              f"blocks.{l}.mlp.proj.weight", f"blocks.{l}.mlp.proj.bias"):
        out[k] = torch.zeros_like(out[k])
    return out


@torch.no_grad()
def donor_task_acc(state, cfg, task, iters=6, bs=128, seed=99):
    m = LM(cfg, "transformer")
    m.load_state_dict(state)
    m.eval()
    g = torch.Generator().manual_seed(seed)
    a = 0.0
    for _ in range(iters):
        x, y = make_batch(task, bs, cfg.ctx, cfg.vocab, g)
        a += task_accuracy(m(x)[0], y)
    return a / iters


def layer_ablation(zoo, cfg):
    """Per-layer contribution: mean donor-acc drop when each block is neutralised."""
    drops = [0.0] * cfg.n_layer
    for z in zoo:
        base = donor_task_acc(z["A"], cfg, z["task"])
        for l in range(cfg.n_layer):
            ab = donor_task_acc(neutralise_block(z["A"], l), cfg, z["task"])
            drops[l] += base - ab
    return [d / len(zoo) for d in drops]


def exp1_isolation(depths, args):
    print("\n" + "=" * 72)
    print("EXP 1  ISOLATION (tied init, different (sigma,k) per donor): mono vs perlayer")
    print("=" * 72)
    results = {}
    for L in depths:
        cfg = cfg_at(L)
        N = args.n_train + args.n_held
        tasks = make_tasks(N, cfg, L, args, seed=100 + L)
        zoo = build_donor_zoo(cfg, tasks, steps=donor_steps_for(L, args), tied=True, verbose=False)
        dm = sum(z["acc"] for z in zoo) / N
        train, held = zoo[:args.n_train], zoo[args.n_train:]
        row = {"donor": dm}
        for name, cls in (("mono", Translator), ("perlayer", PerLayerTranslator)):
            C, tmpl = train_translator(cfg, train, steps=args.c_steps,
                                       tasks_per_step=args.tps, translator_cls=cls)
            row[name] = evaluate(C, tmpl, cfg, held, warm_at=args.warm)
        drops = layer_ablation(zoo, cfg) if args.ablate else None
        results[L] = (row, drops)
        print(f"  L={L} donor {dm:.0%} | "
              f"mono zs {row['mono'][0]:.1%} warm {row['mono'][1]:.1%} s90 {row['mono'][2]:.0f} | "
              f"perlayer zs {row['perlayer'][0]:.1%} warm {row['perlayer'][1]:.1%} "
              f"s90 {row['perlayer'][2]:.0f}"
              + (f" | ablation-drop/blk {[f'{d:.0%}' for d in drops]}" if drops else ""))
    return results


# --------------------------- exp2: depth-transfer (headline) ---------------------------
def exp2_depth_transfer(args):
    print("\n" + "=" * 72)
    print(f"EXP 2  DEPTH-TRANSFER: train perlayer C on L={args.train_depth}, "
          f"apply zero-shot to DEEP donors (the monolith cannot do this at all)")
    print("=" * 72)
    cfg_s = cfg_at(args.train_depth)
    N = args.n_train + args.n_held
    train_zoo = build_donor_zoo(cfg_s, make_tasks(args.n_train, cfg_s, args.train_depth, args, seed=7),
                                steps=donor_steps_for(args.train_depth, args), tied=True, verbose=False)
    print(f"  trained one perlayer C on {args.n_train} donors at L={args.train_depth} "
          f"(donor-acc {sum(z['acc'] for z in train_zoo)/len(train_zoo):.0%})")
    # depth_frac=True: condition on a NORMALISED depth fraction so middle layers of deep
    # nets interpolate between the L=2 endpoints (absolute index would not generalise).
    for use_frac in ([False, True] if args.both_frac else [args.depth_frac]):
        tag = "feat+depthfrac" if use_frac else "feat-only"
        C, _ = train_translator(cfg_s, train_zoo, steps=args.c_steps,
                                tasks_per_step=args.tps,
                                translator_cls=PerLayerTranslator, depth_frac=use_frac)
        print(f"  [{tag}]")
        for L in [args.train_depth] + [d for d in args.deep if d != args.train_depth]:
            cfg_d = cfg_at(L)
            held = build_donor_zoo(cfg_d, make_tasks(args.n_held, cfg_d, L, args, seed=200 + L),
                                   steps=donor_steps_for(L, args), tied=True, verbose=False)
            tmpl_d = LM(cfg_d, "ssm")
            zs, wm, s90 = evaluate(C, tmpl_d, cfg_d, held, warm_at=args.warm)
            # native ceiling: a perlayer C trained AT this depth on its own donors
            ntag = ""
            if args.native_ceiling and L != args.train_depth:
                ntrain = build_donor_zoo(cfg_d, make_tasks(args.n_train, cfg_d, L, args, seed=300 + L),
                                         steps=donor_steps_for(L, args), tied=True, verbose=False)
                Cn, tn = train_translator(cfg_d, ntrain, steps=args.c_steps,
                                          tasks_per_step=args.tps,
                                          translator_cls=PerLayerTranslator, depth_frac=use_frac)
                nzs, _, _ = evaluate(Cn, tn, cfg_d, held, warm_at=args.warm)
                ntag = f"  (native-ceiling zs {nzs:.1%})"
            kind = "TRAIN depth" if L == args.train_depth else "transfer"
            print(f"    L={L:>1} [{kind:11s}] zero-shot {zs:.1%} | warm@{args.warm} {wm:.1%} "
                  f"| s90 {s90:.0f}{ntag}")
    print(f"  (chance {CHANCE:.1%}.  mono Translator's M has shape interior(L) x d_z, "
          f"so it is undefined off its training depth.)")


# --------------------------- exp3: realistic pipeline ---------------------------
def exp3_realistic(depths, args):
    print("\n" + "=" * 72)
    print("EXP 3  REALISTIC: independent init + data_align (phase 5) + per-block generation")
    print("=" * 72)
    for L in depths:
        cfg = cfg_at(L)
        task = make_tasks(1, cfg, L, args, seed=L)[0]          # one shared task per depth
        N = args.n_train + args.n_held
        probe_x = probe_inputs(task, cfg, args.n_probe)
        zoo = build_donor_zoo(cfg, [task] * N, steps=donor_steps_for(L, args), tied=False, verbose=False)
        dm = sum(z["acc"] for z in zoo) / N
        ref = zoo[0]["A"]
        aligned = align_zoo(zoo, ref, "data", cfg=cfg, probe_x=probe_x)
        train, held = aligned[:args.n_train], aligned[args.n_train:]
        line = f"  L={L} donor {dm:.0%} |"
        for name, cls in (("mono", Translator), ("perlayer", PerLayerTranslator)):
            C, tmpl = train_translator(cfg, train, steps=args.c_steps,
                                       tasks_per_step=args.tps, translator_cls=cls)
            zs, wm, s90 = evaluate(C, tmpl, cfg, held, warm_at=args.warm)
            line += f" {name} zs {zs:.1%} warm {wm:.1%} s90 {s90:.0f} |"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="1", help="1 | 2 | 3 | all")
    ap.add_argument("--task", default="sigmak", choices=["sigmak", "hops"],
                    help="sigmak: one-gather (deep blocks idle); hops: depth-demanding")
    ap.add_argument("--maxjump", type=int, default=2, help="hops: backward jump range")
    ap.add_argument("--depths", default="1,2,4,8")
    ap.add_argument("--deep", default="4,6,8", help="exp2 transfer targets")
    ap.add_argument("--train_depth", type=int, default=2, help="exp2 training depth")
    ap.add_argument("--n_train", type=int, default=12)
    ap.add_argument("--n_held", type=int, default=5)
    ap.add_argument("--n_probe", type=int, default=64)
    ap.add_argument("--donor_steps", type=int, default=200)
    ap.add_argument("--c_steps", type=int, default=350)
    ap.add_argument("--tps", type=int, default=6)
    ap.add_argument("--warm", type=int, default=15)
    ap.add_argument("--ablate", action="store_true", default=True)
    ap.add_argument("--no-ablate", dest="ablate", action="store_false")
    ap.add_argument("--depth_frac", action="store_true", default=False)
    ap.add_argument("--both_frac", action="store_true", default=True,
                    help="exp2: run both feat-only and feat+depthfrac")
    ap.add_argument("--native_ceiling", action="store_true", default=True)
    args = ap.parse_args()
    depths = [int(s) for s in args.depths.split(",")]
    args.deep = [int(s) for s in args.deep.split(",")]

    t0 = time.time()
    want = {"1", "2", "3"} if args.exp == "all" else {args.exp}
    if "1" in want:
        exp1_isolation(depths, args)
    if "2" in want:
        exp2_depth_transfer(args)
    if "3" in want:
        exp3_realistic(depths, args)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
