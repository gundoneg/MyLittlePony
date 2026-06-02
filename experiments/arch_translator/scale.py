"""Block C — does the cliff move with network size/depth?

For several (d_model, n_layer) shapes, sweep the init anchor and locate the cliff
(the interior_div where M1 recovery crosses 50%). Hypothesis to check: more depth
gives the gauge-fix more cipher-free interior to lock onto, pushing the cliff right.
Vocab and n_head structure fixed. Text only.
"""
import argparse
from data import CharData
from models import Cfg
from cliff import sweep_knob, crossing50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=250)
    args = ap.parse_args()
    data = CharData()
    shapes = [(48, 2), (96, 2), (192, 2), (96, 1), (96, 4)]
    aligns = [0.70, 0.65, 0.60, 0.575, 0.55, 0.50, 0.45, 0.40]
    print(f"cliff vs scale | V={data.vocab} | n={args.n} steps={args.steps}")
    print(f"  {'d_model':>7} {'n_layer':>7} {'cliff int_div (M1 50%)':>24}  near-edge rows")
    for d, L in shapes:
        c = Cfg(vocab=data.vocab, d_model=d, n_layer=L)
        rows = sweep_knob(c, data, "align", aligns, n=args.n, steps=args.steps)
        x = crossing50(rows)
        edge = " ".join(f"{r['idv']:.2f}:{r['m1']:.0%}" for r in rows
                        if 0.05 < r["m1"] < 0.98)[:46]
        xs = f"{x:.3f}" if x is not None else "n/a"
        print(f"  {d:>7} {L:>7} {xs:>24}  {edge}")
    print("\nread-out: compare the cliff coordinate across width (48/96/192) and depth")
    print("(1/2/4 layers). A rightward shift with depth supports 'more interior =")
    print("more gauge signal'; a flat coordinate means the cliff is size-invariant.")


if __name__ == "__main__":
    main()
