"""Phase 5 — can a small Q-A dataset recover the shared basis (and transfer) for DEEP nets?

Setup (same-task, chosen by the user):
  * N donor TRANSFORMERS solve ONE shared task tau=(sigma,k) from INDEPENDENT inits.
    Same behaviour, different random residual-stream bases. Depth L in {2,4,8}.
  * "Q-A" = a fixed set of task inputs; we read each donor's activations on it.

We put every donor into a reference donor's frame three ways and ask whether the
cross-architecture translator C (transformer -> SSM) then transfers to HELD donors:
  raw    : no alignment            (control; different bases -> should fail)
  weight : orthogonal Procrustes on tok+pos weights (phase-3c; underdetermined)
  data   : orthogonal Procrustes on per-layer activations over the Q-A probe (NEW)

The residual stream is ONE d-dim basis shared across all layers, so activations from
more layers only add constraints on the single alignment Q -> the hypothesis is that
`data` recovers transfer and HOLDS (or improves) as depth grows, where weight/raw fail.

tied rows (every donor from the same init) are the shared-frame upper bound.
"""
import argparse
import time
import torch

from models import Cfg
from task import sample_tasks, make_batch
from donor_zoo import build_donor_zoo
from align import align_zoo, method_Q, frame_residual
from xlate import train_translator, eval_translator, warm_start_acc


def probe_inputs(task, cfg, n, seed=12345):
    g = torch.Generator().manual_seed(seed)
    x, _ = make_batch(task, n, cfg.ctx, cfg.vocab, g)
    return x


def mean_warm(C, template, cfg, held, steps=15):
    accs = []
    for z in held:
        params = C.emit(z["A"])
        accs.append(warm_start_acc(template, params, cfg, z["task"], steps=steps))
    return sum(accs) / len(accs)


def mean_residual(donors, ref, cfg, probe_x, method):
    """Residual achieved by `method`'s own Q, averaged over `donors` (skips the ref)."""
    vals = [frame_residual(z["A"], ref, cfg, probe_x,
                           method_Q(z["A"], ref, cfg, probe_x, method))
            for z in donors]
    return sum(vals) / len(vals)


def run_one_depth(L, args):
    cfg = Cfg(vocab=16, d_model=64, n_layer=L, n_head=4, ctx=32, d_ff=256)
    task = sample_tasks(1, cfg.vocab, seed=L)[0]            # one shared task per depth
    N = args.n_train + args.n_held
    probe_x = probe_inputs(task, cfg, args.n_probe)

    print(f"\n=== L={L} | shared task k={task['k']} | {args.n_train}+{args.n_held} donors "
          f"| chance {1/cfg.vocab:.1%} ===")
    print("  building independent donor zoo (same task, different inits)...")
    zoo = build_donor_zoo(cfg, [task] * N, steps=args.donor_steps, tied=False, verbose=False)
    donor_mean = sum(z["acc"] for z in zoo) / N
    ref = zoo[0]["A"]
    print(f"  donor-acc mean {donor_mean:.2%}")

    rows = []
    for method in ("raw", "weight", "data"):
        amethod = "ortho" if method == "weight" else method
        aligned = align_zoo(zoo, ref, amethod, cfg=cfg, probe_x=probe_x)
        C, template = train_translator(cfg, aligned[:args.n_train], steps=args.c_steps,
                                       tasks_per_step=args.tps)
        zs = eval_translator(C, template, cfg, aligned[args.n_train:])
        wm = mean_warm(C, template, cfg, aligned[args.n_train:], steps=args.warm)
        res = f"{mean_residual(zoo[args.n_train:], ref, cfg, probe_x, method):.3f}"
        rows.append((method, zs, wm, res))
        print(f"  {method:6s} | zero-shot {zs:.2%} | warm@{args.warm} {wm:.2%} | residual {res}")

    # tied upper bound: shared frame by construction, no alignment needed
    tzoo = build_donor_zoo(cfg, [task] * N, steps=args.donor_steps, tied=True, verbose=False)
    Ct, tmpl_t = train_translator(cfg, tzoo[:args.n_train], steps=args.c_steps,
                                  tasks_per_step=args.tps)
    tzs = eval_translator(Ct, tmpl_t, cfg, tzoo[args.n_train:])
    twm = mean_warm(Ct, tmpl_t, cfg, tzoo[args.n_train:], steps=args.warm)
    rows.append(("tied", tzs, twm, "  n/a"))
    print(f"  {'tied':6s} | zero-shot {tzs:.2%} | warm@{args.warm} {twm:.2%} | (upper bound)")
    return cfg, donor_mean, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="2,4,8")
    ap.add_argument("--n_train", type=int, default=12)
    ap.add_argument("--n_held", type=int, default=5)
    ap.add_argument("--n_probe", type=int, default=64)
    ap.add_argument("--donor_steps", type=int, default=200)
    ap.add_argument("--c_steps", type=int, default=300)
    ap.add_argument("--tps", type=int, default=6)
    ap.add_argument("--warm", type=int, default=15)
    args = ap.parse_args()
    depths = [int(s) for s in args.depths.split(",")]

    t0 = time.time()
    results = {}
    for L in depths:
        results[L] = run_one_depth(L, args)

    print("\n" + "=" * 64)
    print("SUMMARY  zero-shot / warm@%d  (chance %.1f%%)" % (args.warm, 100 / 16))
    print("=" * 64)
    hdr = "  L | " + " | ".join(f"{m:^15}" for m in ("raw", "weight", "data", "tied"))
    print(hdr)
    for L in depths:
        _, _, rows = results[L]
        d = {m: (zs, wm) for m, zs, wm, _ in rows}
        cells = " | ".join(f"{d[m][0]:5.1%}/{d[m][1]:5.1%}" for m in
                           ("raw", "weight", "data", "tied"))
        print(f"  {L} |   {cells}")
    print("\nframe residual by each method's own Q (lower=better aligned):")
    for L in depths:
        _, _, rows = results[L]
        d = {m: res for m, _, _, res in rows}
        print(f"  L={L}: raw {d['raw']}  weight {d['weight']}  data {d['data']}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
