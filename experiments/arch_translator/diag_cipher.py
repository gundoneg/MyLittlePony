"""Is the cipher sigma recoverable from a donor's weights? (premise of C_struct)

Canonical = pair 0. For each pair i, the TRUE relative permutation mapping
donor-i token index j -> donor-0 token index is  t = perm_0[perm_i^{-1}[j]]
(same underlying char). Test how well different weight features recover it.
"""
import torch
from data import CharData
from models import Cfg

blob = torch.load("zoo.pt", weights_only=False)
c = Cfg(**blob["cfg"]); zoo = blob["zoo"]
V = c.vocab
perm0 = zoo[0]["perm"]
inv = lambda p: torch.argsort(p)

def true_map(perm_i):
    # j -> perm_0[perm_i^{-1}[j]]
    return perm0[inv(perm_i)]

def acc(pred, true):
    return (pred == true).float().mean().item()

tok0 = zoo[0]["A"]["tok.weight"]
head0 = zoo[0]["A"]["head.weight"]
G0 = tok0 @ head0.t()

print(f"{'pair':>4} {'split':>5} | emb-argmax | emb-cos | G-row-argmax")
for i, z in enumerate(zoo):
    toki = z["A"]["tok.weight"]; headi = z["A"]["head.weight"]
    true = true_map(z["perm"])
    # method 1: raw inner product
    s1 = toki @ tok0.t()
    a1 = acc(s1.argmax(1), true)
    # method 2: cosine
    tn = toki / toki.norm(dim=1, keepdim=True)
    t0n = tok0 / tok0.norm(dim=1, keepdim=True)
    a2 = acc((tn @ t0n.t()).argmax(1), true)
    # method 3: match rows of G_i to rows of G_0 (input->output affinity)
    Gi = toki @ headi.t()
    # row j of Gi (after relabel) ~ row t of G0; compare row distributions
    s3 = -torch.cdist(Gi, G0)
    a3 = acc(s3.argmax(1), true)
    print(f"{i:>4} {z['split']:>5} |   {a1:5.2%}   |  {a2:5.2%} |   {a3:5.2%}")
