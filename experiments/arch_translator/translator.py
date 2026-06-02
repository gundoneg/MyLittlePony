"""Step 3: the weight translator C  (A's weights -> B's weights), two variants.

C_naive  : flat MLP hypernet. feat(A) -> low-rank delta on a base B. Sees the
           raw "costume"; expected to memorize train and fail on held ciphers.
C_struct : learns ONE canonical B0 and reads the cipher sigma out of A's token
           embeddings as a soft permutation P, then relabels B0 by P. This is the
           "look at topology, not the numbers" lens; expected to generalize.

Both are trained with a BEHAVIORAL loss: build the predicted B, run it on the
pair's cipher text, cross-entropy to the true next char. The donor A is only
*read* (its weights are the input), never run for the loss.
"""
import argparse, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from data import CharData
from models import Cfg, LM


# ----------------------------- helpers -----------------------------
def flat(sd, keys):
    return torch.cat([sd[k].reshape(-1) for k in keys])


def unflat(vec, shapes, keys):
    out, i = {}, 0
    for k in keys:
        n = 1
        for s in shapes[k]:
            n *= s
        out[k] = vec[i:i + n].view(shapes[k]); i += n
    return out


def cipher_batch(data, perm, split, bs, ctx, g):
    x, y = data.batch(split, bs, ctx, generator=g)
    return perm[x], perm[y]


# ----------------------------- C_naive -----------------------------
class CNaive(nn.Module):
    def __init__(self, c: Cfg, b_keys, b_shapes, base_vec, dz=48):
        super().__init__()
        # feature: input->output affinity G = tok @ head^T (carries sigma) + flat emb/head
        in_dim = c.vocab * c.vocab + c.vocab * c.d_model * 2
        self.enc = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(),
                                 nn.Linear(256, dz))
        self.base = nn.Parameter(base_vec.clone())
        self.M = nn.Parameter(torch.randn(base_vec.numel(), dz) * 0.01)
        self.b_keys, self.b_shapes, self.c = b_keys, b_shapes, c

    def feat(self, A_sd):
        tok, head = A_sd["tok.weight"], A_sd["head.weight"]
        G = tok @ head.t()
        return torch.cat([G.reshape(-1), tok.reshape(-1), head.reshape(-1)])

    def forward(self, A_sd):
        z = self.enc(self.feat(A_sd))
        vec = self.base + self.M @ z
        return unflat(vec, self.b_shapes, self.b_keys)


# ----------------------------- C_struct -----------------------------
class CStruct(nn.Module):
    def __init__(self, c: Cfg, B0_sd, E_canon, tau=0.5, sinkhorn_iters=8):
        super().__init__()
        self.c, self.tau, self.sk = c, tau, sinkhorn_iters
        self.E_canon = nn.Parameter(E_canon.clone())               # (vocab, d)
        self.B0 = nn.ParameterDict({k.replace(".", "__"): nn.Parameter(v.clone())
                                    for k, v in B0_sd.items()})

    def b0(self):
        return {k.replace("__", "."): v for k, v in self.B0.items()}

    def soft_perm(self, A_tok):
        # match A's token rows to the canonical frame -> P[j, t]
        sim = (A_tok @ self.E_canon.t()) / self.tau                # (vocab, vocab)
        log = F.log_softmax(sim, dim=1)
        for _ in range(self.sk):                                   # Sinkhorn -> ~doubly stochastic
            log = log - torch.logsumexp(log, dim=0, keepdim=True)
            log = log - torch.logsumexp(log, dim=1, keepdim=True)
        return log.exp()

    def forward(self, A_sd):
        P = self.soft_perm(A_sd["tok.weight"])
        params = dict(self.b0())
        params["tok.weight"] = P @ params["tok.weight"]            # relabel rows by cipher
        params["head.weight"] = P @ params["head.weight"]
        return params


# ----------------------------- train / eval -----------------------------
def run_B(template, params, x, y, vocab):
    logits = functional_call(template, params, (x,))
    return F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1)), logits


@torch.no_grad()
def eval_C(C, template, zoo, data, c, split, iters=15, bs=64):
    C.eval()
    g = torch.Generator().manual_seed(7)
    ces, agrees = [], []
    A_tmpl = LM(c, "transformer")
    for z in [r for r in zoo if r["split"] == split]:
        params = C(z["A"])
        ce = ag = 0.0
        for _ in range(iters):
            x, y = cipher_batch(data, z["perm"], "val", bs, c.ctx, g)
            loss, logits = run_B(template, params, x, y, c.vocab)
            la = functional_call(A_tmpl, z["A"], (x,))
            ce += loss.item()
            ag += (logits.argmax(-1) == la.argmax(-1)).float().mean().item()
        ces.append(ce / iters); agrees.append(ag / iters)
    return sum(ces) / len(ces), sum(agrees) / len(agrees)


def train_C(kind, zoo_path="zoo.pt", steps=400, bs=16, lr=3e-3):
    blob = torch.load(zoo_path, weights_only=False)
    c = Cfg(**blob["cfg"]); zoo = blob["zoo"]
    data = CharData()
    template = LM(c, "ssm")
    for p in template.parameters():
        p.requires_grad_(False)
    train = [z for z in zoo if z["split"] == "train"]

    b_keys = list(train[0]["B"].keys())
    b_shapes = {k: tuple(train[0]["B"][k].shape) for k in b_keys}
    base_vec = torch.stack([flat(z["B"], b_keys) for z in train]).mean(0)

    if kind == "naive":
        C = CNaive(c, b_keys, b_shapes, base_vec)
    else:
        C = CStruct(c, train[0]["B"], train[0]["A"]["tok.weight"])

    opt = torch.optim.AdamW(C.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    rnd = math.log(c.vocab)
    print(f"=== train C_{kind} | {len(train)} train pairs | random CE {rnd:.3f} ===")
    t0 = time.time()
    for s in range(1, steps + 1):
        C.train()
        z = train[torch.randint(len(train), (1,), generator=g).item()]
        params = C(z["A"])
        x, y = cipher_batch(data, z["perm"], "train", bs, c.ctx, g)
        loss, _ = run_B(template, params, x, y, c.vocab)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0 or s == 1:
            tr_ce, tr_ag = eval_C(C, template, zoo, data, c, "train")
            he_ce, he_ag = eval_C(C, template, zoo, data, c, "held")
            print(f"  step {s:4d} | loss {loss.item():.3f} | "
                  f"TRAIN ce {tr_ce:.3f} agreeA {tr_ag:.2%} | "
                  f"HELD ce {he_ce:.3f} agreeA {he_ag:.2%} | {time.time()-t0:.0f}s")
    return C


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["naive", "struct"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--zoo", default="zoo.pt")
    a = ap.parse_args()
    train_C(a.kind, a.zoo, a.steps)
