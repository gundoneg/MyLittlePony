"""Build a zoo of cipher-shakespeare donors with CONTROLLED divergence.

Each donor i gets its own random vocab permutation (cipher) sigma_i and is a
TinyTransformer A_i trained on cipher-sigma_i text. Optionally a TinySSM B_i is
distilled from A_i (the translator target).

Two knobs control how far donors drift apart -- this is what lets us titrate
exactly where the cipher stops being readable from static weights:

  --char_aligned  : init the embedding/head by CHARACTER, then place char c's
                    vector at row sigma_i(c). With a shared (tied) interior and a
                    shared data order this makes every donor LITERALLY the same
                    network with its vocab rows permuted -> sigma is exactly the
                    row-permutation, recoverable from weights with ~100% accuracy.
                    This is the controlled anchor (zero "real" divergence).

  --div FLOAT     : fraction of training steps on which a donor draws its own
                    PRIVATE batch order; on the other (1-div) steps every donor
                    draws from a COMMON batch stream (same char indices, then
                    relabelled by its own sigma_i). div=0 -> identical data
                    exposure (only the cipher differs). div=1 -> fully
                    independent data order (the realistic high-divergence end,
                    the regime the original zoo lived in).

The coin that decides shared-vs-private per step is seeded identically for every
donor, so the shared/private generators advance in lockstep across donors and
"shared" steps really do feed identical char indices to all of them.
"""
import argparse
import math
import re
import time
import torch
import torch.nn.functional as F
from data import CharData
from models import Cfg, LM

# Fixed seeds so the divergence knob is fully deterministic and donor-aligned.
COIN_SEED = 7
SHARED_DATA_SEED = 100        # common batch stream (A); same for all donors
PRIV_DATA_SEED = 1000         # base for per-donor private batch stream (A)
SHARED_DISTILL_SEED = 200     # common batch stream (B)
PRIV_DISTILL_SEED = 2000      # base for per-donor private batch stream (B)
BASE_INIT_SEED = 0            # tied interior + char-aligned base vectors

# Interior weights have NO vocabulary axis: the cipher cannot touch them, so the
# distance between donor i's and donor 0's interior is a clean, method-agnostic
# measure of "real" (data-order/init) divergence.
_INTERIOR_RE = re.compile(r"(mix\.qkv|mix\.o|mlp\.fc|mlp\.proj)\.weight$")


def random_cipher(vocab, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(vocab, generator=g)


def inv(p):
    return torch.argsort(p)


def batch(data, split, perm, bs, ctx, g):
    x, y = data.batch(split, bs, ctx, generator=g)
    return perm[x], perm[y]            # relabel tokens by the cipher


def _step_generators(donor_id, shared_seed, priv_seed):
    coin = torch.Generator().manual_seed(COIN_SEED)
    shared_g = torch.Generator().manual_seed(shared_seed)
    priv_g = torch.Generator().manual_seed(priv_seed + donor_id)
    return coin, shared_g, priv_g


def _pick(coin, shared_g, priv_g, div):
    """Pick the batch generator for this step (deterministic, donor-aligned)."""
    if div <= 0.0:
        return shared_g                       # never consult the coin -> all shared
    use_priv = torch.rand(1, generator=coin).item() < div
    return priv_g if use_priv else shared_g


def make_base_init(c):
    """Per-character base vectors for tok/head, shared across donors.

    Built from a reference model so the scale matches the module defaults.
    """
    torch.manual_seed(BASE_INIT_SEED)
    ref = LM(c, "transformer")
    return ref.tok.weight.detach().clone(), ref.head.weight.detach().clone()


def _blend_init(W, base, perm, align):
    """Interpolate a donor's vocab-row init between two regimes.

    align=1 -> CHAR-aligned: row j holds char inv(sigma)[j], i.e. char c always
               starts from base[c] regardless of donor (shared per-char anchor).
    align=0 -> TIED: row j holds base[j], so char c starts from base[sigma(c)]
               -> a different vector in every donor (no shared per-char anchor;
               this is the natural regime the original zoo lived in).
    0<align<1 smoothly removes the anchor. Global scale is preserved so training
    dynamics stay comparable across the sweep.
    """
    ip = inv(perm)
    w = (1.0 - align) * base + align * base[ip]
    w = w * (base.norm() / (w.norm() + 1e-9))
    W.data.copy_(w)


def train_donor(c, data, perm, steps, donor_id, div, align,
                base_tok=None, base_head=None, bs=32, lr=3e-3, init_seed=0,
                noise=0.0, anchor_lam=0.0):
    torch.manual_seed(init_seed)
    A = LM(c, "transformer")
    if align is not None:
        _blend_init(A.tok.weight, base_tok, perm, align)
        _blend_init(A.head.weight, base_head, perm, align)
    if noise > 0.0:
        # donor-specific init perturbation: a second, independent way to inject
        # divergence on top of a clean (align=1) char anchor.
        gn = torch.Generator().manual_seed(5000 + donor_id)
        A.tok.weight.data += noise * torch.randn(A.tok.weight.shape, generator=gn)
        A.head.weight.data += noise * torch.randn(A.head.weight.shape, generator=gn)
    # anchor regulariser pulls char c's CURRENT embedding (row perm[c]) toward the
    # shared per-char reference base[c] -> can re-create readability during training.
    opt = torch.optim.AdamW(A.parameters(), lr=lr)
    coin, shared_g, priv_g = _step_generators(donor_id, SHARED_DATA_SEED, PRIV_DATA_SEED)
    A.train()
    for _ in range(steps):
        g = _pick(coin, shared_g, priv_g, div)
        x, y = batch(data, "train", perm, bs, c.ctx, g)
        _, loss = A(x, y)
        if anchor_lam > 0.0:
            reg = ((A.tok.weight[perm] - base_tok) ** 2).mean() \
                + ((A.head.weight[perm] - base_head) ** 2).mean()
            loss = loss + anchor_lam * reg
        opt.zero_grad()
        loss.backward()
        opt.step()
    return A


def distill_target(A, c, data, perm, steps, donor_id, div,
                   bs=32, lr=3e-3, init_seed=50):
    """Train a fresh SSM B to match A's output distribution (KL)."""
    torch.manual_seed(init_seed)
    B = LM(c, "ssm")
    opt = torch.optim.AdamW(B.parameters(), lr=lr)
    coin, shared_g, priv_g = _step_generators(donor_id, SHARED_DISTILL_SEED, PRIV_DISTILL_SEED)
    A.eval()
    B.train()
    for _ in range(steps):
        g = _pick(coin, shared_g, priv_g, div)
        x, y = batch(data, "train", perm, bs, c.ctx, g)
        with torch.no_grad():
            pa = A(x)[0].softmax(-1)
        logb = B(x)[0].log_softmax(-1)
        loss = F.kl_div(logb.reshape(-1, c.vocab), pa.reshape(-1, c.vocab), reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
    return B


@torch.no_grad()
def evaluate(A, B, c, data, perm, iters=20, bs=64):
    A.eval()
    if B is not None:
        B.eval()
    g = torch.Generator().manual_seed(999)
    a_ce = b_ce = agree = 0.0
    for _ in range(iters):
        x, y = batch(data, "val", perm, bs, c.ctx, g)
        la = A(x)[0]
        a_ce += F.cross_entropy(la.reshape(-1, c.vocab), y.reshape(-1)).item()
        if B is not None:
            lb = B(x)[0]
            b_ce += F.cross_entropy(lb.reshape(-1, c.vocab), y.reshape(-1)).item()
            agree += (la.argmax(-1) == lb.argmax(-1)).float().mean().item()
    return a_ce / iters, b_ce / iters, agree / iters


def interior_vec(state):
    """Flatten the cipher-invariant interior matrices into one vector."""
    parts = [state[k].reshape(-1) for k in state if _INTERIOR_RE.search(k)]
    return torch.cat(parts)


def interior_divergence(zoo):
    """Mean relative L2 distance of each donor's interior to donor 0's."""
    v0 = interior_vec(zoo[0]["A"])
    base = v0.norm().item() + 1e-12
    for z in zoo:
        z["interior_div"] = (interior_vec(z["A"]) - v0).norm().item() / base
    return [z["interior_div"] for z in zoo]


def build_zoo(c, data, n_train=8, n_held=4, steps=600, div=0.0,
              align=1.0, indep_init=False, noise=0.0, anchor_lam=0.0,
              with_b=True, verbose=True):
    """Build donors. `align` (0=tied .. 1=char-aligned) is the init-anchor knob;
    `indep_init` gives each donor its own random init (no shared base at all);
    `noise` adds donor-specific init perturbation; `anchor_lam` regularises each
    char's embedding toward a shared per-char reference during training."""
    n = n_train + n_held
    use_base = ((align is not None) and (not indep_init)) or anchor_lam > 0.0
    base_tok, base_head = make_base_init(c) if use_base else (None, None)
    zoo = []
    t0 = time.time()
    for i in range(n):
        split = "train" if i < n_train else "held"
        perm = random_cipher(c.vocab, seed=1000 + i)
        a_align = None if (indep_init or align is None) else align
        a_init = (1 + i) if indep_init else 0
        A = train_donor(c, data, perm, steps, donor_id=i, div=div, align=a_align,
                        base_tok=base_tok, base_head=base_head, init_seed=a_init,
                        noise=noise, anchor_lam=anchor_lam)
        B = distill_target(A, c, data, perm, steps, donor_id=i, div=div) if with_b else None
        a_ce, b_ce, agree = evaluate(A, B, c, data, perm)
        zoo.append(dict(split=split, perm=perm, A=A.state_dict(),
                        B=(B.state_dict() if B is not None else None),
                        a_ce=a_ce, b_ce=b_ce, agree=agree))
        if verbose:
            extra = f"  B_ce {b_ce:.3f}  agree {agree:.2%}" if with_b else ""
            print(f"  [{i:2d}|{split:5s}] A_ce {a_ce:.3f}{extra}  ({time.time()-t0:.0f}s)")
    interior_divergence(zoo)
    return zoo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=8)
    ap.add_argument("--n_held", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--div", type=float, default=0.0,
                    help="fraction of steps using PRIVATE data order (0=shared, 1=independent)")
    ap.add_argument("--align", type=float, default=1.0,
                    help="init anchor: 1=char-aligned (clean), 0=tied (no per-char anchor)")
    ap.add_argument("--indep_init", action="store_true",
                    help="each donor gets its own random init (overrides --align)")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="donor-specific init perturbation on tok/head (2nd divergence axis)")
    ap.add_argument("--anchor_lam", type=float, default=0.0,
                    help="weight of the shared per-char anchor regulariser during training")
    ap.add_argument("--no_b", dest="with_b", action="store_false", default=True,
                    help="skip SSM distillation (only A donors; faster for sigma-recovery)")
    ap.add_argument("--out", default="zoo.pt")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_train, args.n_held, args.steps = 2, 2, 200

    data = CharData()
    c = Cfg(vocab=data.vocab)
    rnd = math.log(data.vocab)
    n = args.n_train + args.n_held
    print(f"building zoo: {args.n_train} train + {args.n_held} held = {n} pairs | "
          f"steps={args.steps} | div={args.div} | align={args.align} | "
          f"indep_init={args.indep_init} | with_b={args.with_b} | random CE={rnd:.3f}")

    zoo = build_zoo(c, data, args.n_train, args.n_held, args.steps,
                    div=args.div, align=args.align, indep_init=args.indep_init,
                    noise=args.noise, anchor_lam=args.anchor_lam, with_b=args.with_b)
    meta = dict(div=args.div, align=args.align, indep_init=args.indep_init,
                noise=args.noise, anchor_lam=args.anchor_lam)
    torch.save(dict(cfg=c.__dict__, zoo=zoo, meta=meta), args.out)

    div_mean = sum(z["interior_div"] for z in zoo[1:]) / max(len(zoo) - 1, 1)
    print(f"\nsaved {args.out} ({n} pairs)")
    print(f"  mean interior-divergence vs donor 0: {div_mean:.3f}  (0=identical, ~1.41=random)")
    tr = [z for z in zoo if z["split"] == "train"]
    he = [z for z in zoo if z["split"] == "held"]

    def avg(rows, k):
        return sum(r[k] for r in rows) / max(len(rows), 1)
    print(f"  TRAIN: A_ce {avg(tr,'a_ce'):.3f}  B_ce {avg(tr,'b_ce'):.3f}  agree {avg(tr,'agree'):.2%}")
    print(f"  HELD : A_ce {avg(he,'a_ce'):.3f}  B_ce {avg(he,'b_ce'):.3f}  agree {avg(he,'agree'):.2%}")


if __name__ == "__main__":
    main()
