"""Weights-only recovery of the cipher sigma_i (relative to canonical donor 0).

Three methods of increasing sophistication, all reading ONLY static weights:

  M0  emb-cos    : optimal (Hungarian) match of tok rows by cosine. Baseline.
  M1  gauge-fix  : first remove the hidden-channel gauge between donor i and 0
                   (orthogonal Procrustes on the cipher-free interior matrices),
                   then match tok rows. Tests whether aligning the hidden basis
                   rescues recovery as divergence grows.
  M2  graph-QAP  : Frank-Wolfe on the behaviour matrix G = tok.head^T, solving
                   G_i P ~ P G_0 (a gauge-invariant view). The most "topological"
                   method -- needs no embedding alignment at all.

Accuracy is the fraction of vocab indices mapped to the true relative target,
true_map(j) = perm_0[ inv(perm_i)[j] ]  (same underlying character). Chance = 1/V.
"""
import numpy as np
import torch
from hungarian import assign_from_similarity


def inv(p):
    return torch.argsort(p)


def true_map(perm_i, perm0):
    return perm0[inv(perm_i)]


def _acc(pred, true):
    return (torch.as_tensor(pred) == true).float().mean().item()


def _unit(x):
    return x / (x.norm(dim=1, keepdim=True) + 1e-9)


# --------------------------- M0: embedding cosine ---------------------------
def recover_m0(toki, tok0):
    sim = (_unit(toki) @ _unit(tok0).t()).numpy()
    return torch.tensor(assign_from_similarity(sim))


# --------------------------- M1: gauge-aligned ------------------------------
def _interior_inputs(state, keys):
    """Stack interior matrices that take the hidden dim as INPUT (out, d)."""
    return torch.cat([state[k] for k in keys], dim=0)


def _gauge_R(state_i, state0, cfg):
    """Orthogonal R (d x d) s.t. donor-0 hidden coords x map to donor-i as R x.

    Estimated from cipher-free interior matrices W (out, d): W_i ~ W_0 R^T, so
    R^T = orthogonal Procrustes of W_0 -> W_i. Averaged over layers via stacking.
    """
    keys = [k for k in state0
            if (k.endswith("mix.qkv.weight") or k.endswith("mlp.fc.weight"))]
    A = _interior_inputs(state0, keys)        # (sum_out, d)
    B = _interior_inputs(state_i, keys)
    M = A.t() @ B                              # (d, d)
    U, _, Vt = torch.linalg.svd(M)
    Rt = U @ Vt                                # R^T
    return Rt


def recover_m1(state_i, state0, cfg):
    Rt = _gauge_R(state_i, state0, cfg)
    tok0_aligned = state0["tok.weight"] @ Rt   # rotate donor-0 emb into donor-i basis
    sim = (_unit(state_i["tok.weight"]) @ _unit(tok0_aligned).t()).numpy()
    return torch.tensor(assign_from_similarity(sim))


# --------------------------- M2: graph-matching QAP -------------------------
def _perm_mat(col, V):
    S = torch.zeros(V, V, dtype=torch.double)
    S[torch.arange(V), torch.as_tensor(col)] = 1.0
    return S


def _qap_obj(col, Gi, G0):
    P = _perm_mat(col, Gi.shape[0])
    R = Gi @ P - P @ G0
    return float((R * R).sum())


def recover_m2(state_i, state0, init, fw_iters=40):
    """Frank-Wolfe graph-matching: refine `init` toward P minimising
    ||G_i P - P G_0||_F, where G = tok @ head^T (V x V) is invariant to the
    hidden-channel gauge. Objective-guarded: returns the best PERMUTATION seen
    along the path, so it can never degrade the warm start.
    """
    G0 = (state0["tok.weight"] @ state0["head.weight"].t()).double()
    Gi = (state_i["tok.weight"] @ state_i["head.weight"].t()).double()
    V = G0.shape[0]
    G0 = G0 / (G0.norm() + 1e-9) * V          # scale-match the two donors
    Gi = Gi / (Gi.norm() + 1e-9) * V
    best_col = torch.as_tensor(init).clone()
    best = _qap_obj(best_col, Gi, G0)
    P = _perm_mat(init, V)
    for k in range(fw_iters):
        R = Gi @ P - P @ G0
        grad = Gi.t() @ R - R @ G0.t()
        col = assign_from_similarity((-grad).numpy())     # LMO: minimise <grad,S>
        o = _qap_obj(col, Gi, G0)                         # rounded vertex
        if o < best:
            best, best_col = o, torch.tensor(col)
        gamma = 2.0 / (k + 2.0)                           # standard FW step
        P = (1 - gamma) * P + gamma * _perm_mat(col, V)
    col = assign_from_similarity(P.numpy())               # final rounding
    if _qap_obj(col, Gi, G0) < best:
        best_col = torch.tensor(col)
    return best_col


# --------------------------------- driver -----------------------------------
def recover_all(zoo, cfg, methods=("m0", "m1", "m2")):
    perm0 = zoo[0]["perm"]
    tok0 = zoo[0]["A"]["tok.weight"]
    out = {m: [] for m in methods}
    for z in zoo:
        si = z["A"]
        true = true_map(z["perm"], perm0)
        if "m0" in methods:
            out["m0"].append(_acc(recover_m0(si["tok.weight"], tok0), true))
        if "m1" in methods:
            out["m1"].append(_acc(recover_m1(si, zoo[0]["A"], cfg), true))
        if "m2" in methods:
            warm = recover_m0(si["tok.weight"], tok0)   # graph-refine the emb guess
            out["m2"].append(_acc(recover_m2(si, zoo[0]["A"], init=warm), true))
    return out


def _mean_excl0(vals):
    return sum(vals[1:]) / max(len(vals) - 1, 1)


if __name__ == "__main__":
    from models import Cfg
    blob = torch.load("zoo.pt", weights_only=False)
    c = Cfg(**blob["cfg"])
    zoo = blob["zoo"]
    meta = blob.get("meta", {})
    res = recover_all(zoo, c)
    chance = 1.0 / c.vocab
    print(f"zoo: div={meta.get('div','?')} char_aligned={meta.get('char_aligned','?')} "
          f"| V={c.vocab} chance={chance:.2%}")
    print(f"{'pair':>4} {'split':>5} {'int_div':>8} | {'M0 emb':>7} {'M1 gauge':>9} {'M2 graph':>9}")
    for i, z in enumerate(zoo):
        print(f"{i:>4} {z['split']:>5} {z.get('interior_div',0):>8.3f} | "
              f"{res['m0'][i]:>7.1%} {res['m1'][i]:>9.1%} {res['m2'][i]:>9.1%}")
    print(f"{'MEAN(excl.0)':>18} | {_mean_excl0(res['m0']):>7.1%} "
          f"{_mean_excl0(res['m1']):>9.1%} {_mean_excl0(res['m2']):>9.1%}")
