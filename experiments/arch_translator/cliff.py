"""Block A — locate the cliff precisely and test if it is a genuine phase
transition in *divergence* (mechanism-independent).

Two independent ways to inject divergence on top of the cipher:
  * align  : weaken the shared per-char init anchor (1 -> 0)
  * noise  : keep align=1 but add donor-specific init perturbation

If sigma-recovery collapses at the *same* critical interior-divergence under both
mechanisms, the boundary is a property of divergence itself, not of the knob.
Reports each sweep against the method-agnostic interior_div axis plus the
interpolated 50%-crossing of M1 (= the cliff coordinate). Text only.
"""
import argparse
from data import CharData
from models import Cfg
from zoo import build_zoo
from recover import recover_all, _mean_excl0


def sweep_knob(c, data, knob, values, *, n=8, steps=250, methods=("m0", "m1")):
    """Run build+recover while varying one knob; return list of dicts sorted by
    measured interior divergence."""
    rows = []
    for v in values:
        kw = dict(n_train=n, n_held=0, steps=steps, div=0.0, with_b=False, verbose=False)
        kw[knob if knob != "align" else "align"] = v
        if knob == "noise":
            kw["align"] = 1.0
        zoo = build_zoo(c, data, **kw)
        r = recover_all(zoo, c, methods=methods)
        idv = sum(z["interior_div"] for z in zoo[1:]) / max(len(zoo) - 1, 1)
        rows.append(dict(v=v, idv=idv, **{m: _mean_excl0(r[m]) for m in methods}))
    rows.sort(key=lambda d: d["idv"])
    return rows


def crossing50(rows, key="m1"):
    """Interpolated interior_div where recovery crosses 50% (sorted by idv)."""
    for a, b in zip(rows, rows[1:]):
        if a[key] >= 0.5 > b[key]:
            t = (a[key] - 0.5) / (a[key] - b[key] + 1e-9)
            return a["idv"] + t * (b["idv"] - a["idv"])
    return None


def _print(title, knob, rows, methods):
    print(f"\n{title}")
    head = f"  {knob:>7} {'int_div':>8} " + " ".join(f"{m.upper():>6}" for m in methods)
    print(head)
    for d in rows:
        cells = " ".join(f"{d[m]:>6.1%}" for m in methods)
        print(f"  {d['v']:>7.3g} {d['idv']:>8.3f} {cells}")
    x = crossing50(rows)
    print(f"  -> M1 50%-crossing at interior_div = "
          f"{('%.3f' % x) if x is not None else 'n/a (no crossing in range)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=250)
    args = ap.parse_args()
    data = CharData()
    c = Cfg(vocab=data.vocab)
    print(f"cliff localisation | V={c.vocab} chance={1/c.vocab:.1%} | n={args.n} steps={args.steps}")

    a_rows = sweep_knob(c, data, "align",
                        [0.65, 0.625, 0.60, 0.575, 0.55, 0.50, 0.45],
                        n=args.n, steps=args.steps, methods=("m0", "m1", "m3"))
    _print("MECHANISM 1 — weaken the init anchor (align); M3 = stronger reader",
           "align", a_rows, ("m0", "m1", "m3"))
    xm1, xm3 = crossing50(a_rows, "m1"), crossing50(a_rows, "m3")
    print(f"  (Block B-2) M3 50%-crossing {('%.3f'%xm3) if xm3 else 'n/a'} vs "
          f"M1 {('%.3f'%xm1) if xm1 else 'n/a'} -> "
          f"{'M3 pushes the cliff right' if (xm3 and xm1 and xm3>xm1+0.01) else 'M3 does NOT move the cliff'}")

    n_rows = sweep_knob(c, data, "noise",
                        [1.0, 2.0, 2.5, 3.0, 4.0, 6.0],
                        n=args.n, steps=args.steps)
    _print("MECHANISM 2 — donor-specific init noise (align=1)", "noise", n_rows, ("m0", "m1"))

    xa, xn = crossing50(a_rows), crossing50(n_rows)
    print("\nphase-transition check:")
    if xa is not None and xn is not None:
        print(f"  align-cliff @ int_div {xa:.3f}  vs  noise-cliff @ int_div {xn:.3f}  "
              f"(|Δ|={abs(xa-xn):.3f})")
        print("  close => the boundary is a property of divergence, not of the knob.")
    else:
        print(f"  align-crossing={xa}  noise-crossing={xn}  (widen a sweep if one is n/a)")


if __name__ == "__main__":
    main()
