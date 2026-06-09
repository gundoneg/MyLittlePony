"""Close the loop: does a weights-read sigma give a WORKING translated model?

The translator C is now trivial *given* sigma: to turn the canonical target B_0
into donor i's target, relabel its vocab rows by the cipher recovered from A_i's
weights:  B_hat_i.tok[j] = B_0.tok[ sigma_hat(j) ]  (same for head; interior copied).

We build a small zoo WITH distilled SSM targets, recover sigma from each A_i, build
B_hat_i, and compare it to the TRUE B_i:
  * CE on cipher-sigma_i validation text   (lower = better; ceiling = true B_i)
  * top-1 agreement between B_hat_i and B_i (1.0 = behaviourally identical)
against a naive baseline that skips relabelling (uses B_0 as-is).

Run at align=1 (clean anchor) and align=0 (natural regime) to see the loop work
then fail exactly where sigma-recovery does.
"""
import argparse
import torch
import torch.nn.functional as F
from data import CharData
from models import Cfg, LM
from zoo import build_zoo, batch
from recover import recover_m1, recover_m0, true_map, _acc


def relabel_B(B0_state, pred, cfg):
    """Build B_hat by permuting B_0's vocab rows with the recovered map pred
    (donor-i row j  <-  donor-0 row pred[j]); interior copied verbatim."""
    s = {k: v.clone() for k, v in B0_state.items()}
    s["tok.weight"] = B0_state["tok.weight"][pred]
    s["head.weight"] = B0_state["head.weight"][pred]
    return s


@torch.no_grad()
def compare(Bhat_state, Btrue_state, c, data, perm, iters=20, bs=64):
    """CE of each model on cipher-perm val text + their top-1 agreement."""
    Bh, Bt = LM(c, "ssm"), LM(c, "ssm")
    Bh.load_state_dict(Bhat_state); Bt.load_state_dict(Btrue_state)
    Bh.eval(); Bt.eval()
    g = torch.Generator().manual_seed(999)
    ce_h = ce_t = agree = 0.0
    for _ in range(iters):
        x, y = batch(data, "val", perm, bs, c.ctx, g)
        lh, lt = Bh(x)[0], Bt(x)[0]
        ce_h += F.cross_entropy(lh.reshape(-1, c.vocab), y.reshape(-1)).item()
        ce_t += F.cross_entropy(lt.reshape(-1, c.vocab), y.reshape(-1)).item()
        agree += (lh.argmax(-1) == lt.argmax(-1)).float().mean().item()
    return ce_h / iters, ce_t / iters, agree / iters


def run(c, data, align, n=4, steps=250):
    zoo = build_zoo(c, data, n_train=n, n_held=0, steps=steps, div=0.0,
                    align=align, with_b=True, verbose=False)
    perm0 = zoo[0]["perm"]
    tok0 = zoo[0]["A"]["tok.weight"]
    B0 = zoo[0]["B"]
    rows = []
    for i, z in enumerate(zoo[1:], start=1):
        true = true_map(z["perm"], perm0)
        pred = recover_m1(z["A"], zoo[0]["A"], c)            # best weights-only sigma
        sigma_acc = _acc(pred, true)
        # translated model (relabel B_0 by recovered sigma) vs naive (B_0 as-is)
        Bhat = relabel_B(B0, pred.tolist(), c)
        ce_h, ce_t, agree = compare(Bhat, z["B"], c, data, z["perm"])
        ce_n, _, agree_n = compare(B0, z["B"], c, data, z["perm"])  # naive: no relabel
        rows.append((i, sigma_acc, ce_h, ce_t, ce_n, agree, agree_n))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=250)
    args = ap.parse_args()
    data = CharData()
    c = Cfg(vocab=data.vocab)
    print(f"translator loop | V={c.vocab} | n={args.n} steps={args.steps}")
    print("  B_hat = relabel(B_0, recovered sigma);  ceiling = true B_i;  naive = B_0 as-is\n")
    for align in (1.0, 0.0):
        tag = "char-aligned (anchor)" if align == 1.0 else "tied (natural)"
        print(f"=== align={align:.1f}  [{tag}] ===")
        print(f"{'pair':>4} {'sigma_acc':>9} | {'B_hat CE':>9} {'true CE':>8} "
              f"{'naive CE':>9} | {'agree(hat,true)':>15} {'agree(naive)':>13}")
        rows = run(c, data, align, n=args.n, steps=args.steps)
        for (i, sa, ce_h, ce_t, ce_n, ag, ag_n) in rows:
            print(f"{i:>4} {sa:>9.1%} | {ce_h:>9.3f} {ce_t:>8.3f} {ce_n:>9.3f} "
                  f"| {ag:>15.1%} {ag_n:>13.1%}")
        def avg(j): return sum(r[j] for r in rows) / len(rows)
        print(f"{'MEAN':>4} {avg(1):>9.1%} | {avg(2):>9.3f} {avg(3):>8.3f} {avg(4):>9.3f} "
              f"| {avg(5):>15.1%} {avg(6):>13.1%}\n")
    print("read-out: at align=1 the relabelled B_hat should match true B_i (CE ~ true,")
    print("agreement high) while the naive B_0 does not; at align=0 sigma is lost so the")
    print("translated model is no better than naive -> the loop fails where recovery does.")


if __name__ == "__main__":
    main()
