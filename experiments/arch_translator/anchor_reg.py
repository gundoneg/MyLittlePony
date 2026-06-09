"""Block B-1 — can a shared per-char anchor INJECTED DURING TRAINING move the
cliff? Starting from the natural tied init (align=0, sigma at chance), add a
regulariser pulling each character's embedding toward a shared per-char
reference, sweep its weight `lam`, and watch recovery vs the cost in task CE.

Honest framing: this measures how much shared signal is needed to restore
weight-readability of sigma -- not reading a random cipher "from nothing".
"""
import argparse
from data import CharData
from models import Cfg
from zoo import build_zoo
from recover import recover_all, _mean_excl0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[0.0, 1.0, 10.0, 100.0, 1000.0])
    args = ap.parse_args()
    data = CharData()
    c = Cfg(vocab=data.vocab)
    print(f"anchor-reg @ align=0 (tied) | V={c.vocab} chance={1/c.vocab:.1%} "
          f"| n={args.n} steps={args.steps}")
    print(f"  {'lam':>8} {'int_div':>8} {'A_ce':>6} {'M0':>7} {'M1':>7}")
    for lam in args.lams:
        zoo = build_zoo(c, data, n_train=args.n, n_held=0, steps=args.steps, div=0.0,
                        align=0.0, anchor_lam=lam, with_b=False, verbose=False)
        r = recover_all(zoo, c, methods=("m0", "m1"))
        idv = sum(z["interior_div"] for z in zoo[1:]) / max(len(zoo) - 1, 1)
        ace = sum(z["a_ce"] for z in zoo) / len(zoo)
        print(f"  {lam:>8.1f} {idv:>8.3f} {ace:>6.2f} "
              f"{_mean_excl0(r['m0']):>7.1%} {_mean_excl0(r['m1']):>7.1%}")
    print("\nread-out: if recovery stays at chance until lam is large enough to force")
    print("the embeddings back to a shared anchor (and CE starts paying for it), the")
    print("conclusion is that the cliff is sharp -- a weak injected anchor does not move")
    print("it; you must essentially re-impose the full anchor (= align=1).")


if __name__ == "__main__":
    main()
