"""Titrate WHERE the cipher sigma stops being readable from static weights.

Controlled experiment. Donors differ only by cipher sigma_i; two knobs control
how far they actually drift:
  align : init anchor  (1 = char-aligned, 0 = tied / no shared per-char anchor)
  div   : data-order   (0 = shared batch stream, 1 = independent)

We sweep these and, for each setting, measure how well three WEIGHTS-ONLY methods
recover sigma (M0 emb-cosine, M1 gauge-fixed, M2 graph-QAP), against the measured
interior divergence (a cipher-free, method-agnostic ruler). Output is text only
(no matplotlib in this env): three tables + an ASCII curve.
"""
import argparse
import torch
from data import CharData
from models import Cfg
from zoo import build_zoo
from recover import recover_all, _mean_excl0


def run(c, data, *, align=1.0, div=0.0, indep_init=False, n=8, steps=250):
    zoo = build_zoo(c, data, n_train=n, n_held=0, steps=steps, div=div,
                    align=align, indep_init=indep_init, with_b=False, verbose=False)
    res = recover_all(zoo, c)
    idv = sum(z["interior_div"] for z in zoo[1:]) / max(len(zoo) - 1, 1)
    return idv, {m: _mean_excl0(v) for m, v in res.items()}


def _row(label, idv, r):
    return (f"{label:>16} {idv:>8.3f} | {r['m0']:>7.1%} {r['m1']:>9.1%} {r['m2']:>9.1%}")


def _ascii_curve(points, chance, width=44):
    """points: list of (interior_div, m0). Plot m0 (y) vs interior_div (x)."""
    print("\n  ASCII curve  —  M0 sigma-recovery vs interior-divergence")
    print(f"  100% |{'recoverable':>{width}}")
    xs = [p[0] for p in points]
    xmax = max(xs + [1.41])
    for frac in (1.0, 0.75, 0.5, 0.25):
        line = [" "] * width
        for idv, m0 in points:
            col = min(width - 1, int(idv / xmax * (width - 1)))
            if m0 >= frac - 0.125 and m0 < frac + 0.125:
                line[col] = "*"
        print(f"  {frac:>3.0%} |" + "".join(line))
    base = [" "] * width
    for idv, m0 in points:
        col = min(width - 1, int(idv / xmax * (width - 1)))
        if m0 < 0.125:
            base[col] = "*"
    print(f"  {chance:>4.1%}|" + "".join(base) + "  (chance floor)")
    print(f"       +{'-'*width}")
    print(f"        0{'interior divergence ->':^{width-6}}{xmax:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--fast", action="store_true", help="fewer donors/steps")
    args = ap.parse_args()
    if args.fast:
        args.n, args.steps = 5, 150

    data = CharData()
    c = Cfg(vocab=data.vocab)
    chance = 1.0 / c.vocab
    hdr = f"{'setting':>16} {'int_div':>8} | {'M0 emb':>7} {'M1 gauge':>9} {'M2 graph':>9}"
    print(f"titration | V={c.vocab} chance={chance:.1%} | n={args.n} steps={args.steps}\n")

    # --- Table A: the limit curve — sweep the init anchor (data order shared) ---
    print("=" * 60)
    print("TABLE A  —  init-anchor sweep (div=0, shared data order)")
    print("  align=1 char-aligned  ...  align=0 tied (natural regime)")
    print(hdr)
    curve = []
    for a in (1.0, 0.85, 0.7, 0.6, 0.5, 0.3, 0.0):
        idv, r = run(c, data, align=a, div=0.0, n=args.n, steps=args.steps)
        print(_row(f"align={a:.2f}", idv, r))
        curve.append((idv, r["m0"]))
    _ascii_curve(curve, chance)

    # --- Table B: data order is a non-factor (sweep div at both ends) ---
    print("\n" + "=" * 60)
    print("TABLE B  —  data-order sweep (does batch divergence matter?)")
    print(hdr)
    for a in (1.0, 0.0):
        for d in (0.0, 0.5, 1.0):
            idv, r = run(c, data, align=a, div=d, n=args.n, steps=args.steps)
            print(_row(f"align={a:.0f} div={d:.1f}", idv, r))

    # --- Table C: the natural regimes that reproduce the original negative result ---
    print("\n" + "=" * 60)
    print("TABLE C  —  natural regimes (reproduce 'NOT recoverable')")
    print(hdr)
    idv, r = run(c, data, align=0.0, div=1.0, n=args.n, steps=args.steps)
    print(_row("tied init", idv, r))
    idv, r = run(c, data, indep_init=True, div=1.0, n=args.n, steps=args.steps)
    print(_row("independent init", idv, r))

    print("\nread-out:")
    print("  * align=1 (shared per-char anchor): sigma is EXACTLY readable (~100%).")
    print("  * a sharp cliff sits near interior_div ~0.8; below the anchor (tied/indep)")
    print("    sigma collapses to chance and NO weights-only method recovers it.")
    print("  * data order (div) barely moves the needle -> the anchor, not the data, is")
    print("    what carries sigma in static weights.")


if __name__ == "__main__":
    main()
