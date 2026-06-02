"""Step 2: build a zoo of (A_i, B_i) cipher-shakespeare pairs.

Each pair has its own random vocab permutation (cipher) sigma_i:
  - A_i (Transformer) is TRAINED on cipher-sigma_i text.
  - B_i (SSM) is DISTILLED from A_i (matches A_i's logits) -> a target that is
    functionally tied to this specific A_i. B_i is the "ground truth" the
    translator C will later learn to predict from A_i's weights.

The cipher gives the zoo real diversity: to produce a correct B_i the translator
MUST read sigma_i out of A_i's weights -- it cannot ignore A and output one B.
"""
import argparse, math, time
import torch
import torch.nn.functional as F
from data import CharData
from models import Cfg, LM, n_params


def random_cipher(vocab, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(vocab, generator=g)


def batch(data, split, perm, bs, ctx, g):
    x, y = data.batch(split, bs, ctx, generator=g)
    return perm[x], perm[y]            # relabel tokens by the cipher


def train_donor(c, data, perm, steps, bs=32, lr=3e-3, init_seed=1, data_seed=1):
    torch.manual_seed(init_seed)
    A = LM(c, "transformer")
    opt = torch.optim.AdamW(A.parameters(), lr=lr)
    g = torch.Generator().manual_seed(100 + data_seed)
    A.train()
    for _ in range(steps):
        x, y = batch(data, "train", perm, bs, c.ctx, g)
        _, loss = A(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return A


def distill_target(A, c, data, perm, steps, bs=32, lr=3e-3, init_seed=2, data_seed=2):
    """Train a fresh SSM B to match A's output distribution (KL)."""
    torch.manual_seed(init_seed)
    B = LM(c, "ssm")
    opt = torch.optim.AdamW(B.parameters(), lr=lr)
    g = torch.Generator().manual_seed(200 + data_seed)
    A.eval(); B.train()
    for _ in range(steps):
        x, y = batch(data, "train", perm, bs, c.ctx, g)
        with torch.no_grad():
            pa = A(x)[0].softmax(-1)
        logb = B(x)[0].log_softmax(-1)
        loss = F.kl_div(logb.reshape(-1, c.vocab), pa.reshape(-1, c.vocab), reduction="batchmean")
        opt.zero_grad(); loss.backward(); opt.step()
    return B


@torch.no_grad()
def evaluate(A, B, c, data, perm, iters=20, bs=64):
    A.eval(); B.eval()
    g = torch.Generator().manual_seed(999)
    a_ce = b_ce = agree = 0.0
    for _ in range(iters):
        x, y = batch(data, "val", perm, bs, c.ctx, g)
        la = A(x)[0]; lb = B(x)[0]
        a_ce += F.cross_entropy(la.reshape(-1, c.vocab), y.reshape(-1)).item()
        b_ce += F.cross_entropy(lb.reshape(-1, c.vocab), y.reshape(-1)).item()
        agree += (la.argmax(-1) == lb.argmax(-1)).float().mean().item()
    return a_ce / iters, b_ce / iters, agree / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=8)
    ap.add_argument("--n_held", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", default="zoo.pt")
    ap.add_argument("--tied", action="store_true", default=True,
                    help="all donors share one init (common basis); default on")
    ap.add_argument("--independent", dest="tied", action="store_false",
                    help="each donor gets its own random init")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_train, args.n_held, args.steps = 2, 2, 200

    data = CharData()
    c = Cfg(vocab=data.vocab)
    rnd = math.log(data.vocab)
    n = args.n_train + args.n_held
    print(f"building zoo: {args.n_train} train + {args.n_held} held = {n} pairs | "
          f"steps={args.steps} | tied_init={args.tied} | random CE={rnd:.3f}")

    zoo = []
    t0 = time.time()
    for i in range(n):
        split = "train" if i < args.n_train else "held"
        perm = random_cipher(data.vocab, seed=1000 + i)
        # tied: shared init across donors (common basis) -> only the cipher differs.
        a_init = 0 if args.tied else 1 + i
        b_init = 50 if args.tied else 101 + i
        A = train_donor(c, data, perm, args.steps, init_seed=a_init, data_seed=1 + i)
        B = distill_target(A, c, data, perm, args.steps, init_seed=b_init, data_seed=101 + i)
        a_ce, b_ce, agree = evaluate(A, B, c, data, perm)
        zoo.append(dict(split=split, perm=perm,
                        A=A.state_dict(), B=B.state_dict(),
                        a_ce=a_ce, b_ce=b_ce, agree=agree))
        print(f"  [{i:2d}|{split:5s}] A_ce {a_ce:.3f}  B_ce {b_ce:.3f}  "
              f"A/B top1-agree {agree:.2%}  ({time.time()-t0:.0f}s)")

    torch.save(dict(cfg=c.__dict__, zoo=zoo), args.out)
    tr = [z for z in zoo if z["split"] == "train"]
    he = [z for z in zoo if z["split"] == "held"]
    def avg(rows, k): return sum(r[k] for r in rows) / max(len(rows), 1)
    print(f"\nsaved {args.out} ({n} pairs)")
    print(f"  TRAIN: A_ce {avg(tr,'a_ce'):.3f}  B_ce {avg(tr,'b_ce'):.3f}  agree {avg(tr,'agree'):.2%}")
    print(f"  HELD : A_ce {avg(he,'a_ce'):.3f}  B_ce {avg(he,'b_ce'):.3f}  agree {avg(he,'agree'):.2%}")
    print(f"  (random CE {rnd:.3f} -> both archs should be well below; agree high = good distill)")


if __name__ == "__main__":
    main()
