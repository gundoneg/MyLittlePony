"""Phase 9 — effective rank of the interior: how many FUNCTIONALLY distinct transformations
do supra50m's deep layers carry? This decides whether the adapter failure was due to genuine
diversity (need ~8 types) or merely forward-compounding dynamics (few types, badly conditioned).

Gauge-invariant, measured in function space: run every interior block on a COMMON input
distribution H (the residual stream entering the interior), take each block's residual UPDATE
U_i = block_i(H) - H, and compute the participation ratio (effective rank) of the set {U_i}.
PR ~ 1 => all updates collinear (one type, a shared exemplar could serve all);
PR ~ 8 => mutually distinct (genuinely need full per-slot capacity).
"""
import torch
from llama_min import MinLlama


def participation_ratio(M):
    """M: (k, D) rows = flattened per-layer update fields. PR of the kxk Gram spectrum."""
    M = M / (M.norm(dim=1, keepdim=True) + 1e-9)
    G = M @ M.t()
    ev = torch.linalg.eigvalsh(G).clamp(min=0)
    return (ev.sum() ** 2 / (ev.pow(2).sum() + 1e-12)).item(), ev.flip(0)


@torch.no_grad()
def updates_on_common_input(m, H, slots):
    pos = torch.arange(H.shape[1])
    rows = []
    for i in slots:
        U = m._block(H, i, pos) - H                      # residual update of block i on common H
        rows.append(U.reshape(-1))
    return torch.stack(rows)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = MinLlama()
    seqs = [m.generate(torch.tensor([[1]]), 160, temperature=0.9)[:, 1:] for _ in range(3)]
    idx = torch.cat(seqs, 0)
    _, hs = m.forward(idx, capture=True)

    interior = list(range(3, 11))
    H = hs[3]                                            # common input = stream entering the interior
    M = updates_on_common_input(m, H, interior)
    pr, ev = participation_ratio(M)

    print(f"supra50m interior layers {interior} on a COMMON input (stream entering slot 3)\n")
    # pairwise cosine of update fields
    Mn = M / (M.norm(dim=1, keepdim=True) + 1e-9)
    C = Mn @ Mn.t()
    print("pairwise cos of residual-update fields:")
    print("      " + " ".join(f"{j:>5}" for j in interior))
    for a, i in enumerate(interior):
        print(f"  {i:>3} " + " ".join(f"{C[a,b]:5.2f}" for b in range(len(interior))))

    print(f"\n  participation ratio (effective #distinct update directions): {pr:.2f}  of {len(interior)}")
    print(f"  normalized Gram eigenvalues: " + " ".join(f"{e/ev.sum():.2f}" for e in ev))

    # how much does a single shared exemplar capture? variance explained by the top component
    print(f"  top-1 component explains {ev[0]/ev.sum():.0%} of update-field energy "
          f"(=> a shared exemplar covers ~this much; the rest needs per-slot capacity)")

    # context: same PR over ALL layers and over just the early layers
    for label, sl in [("all 12", list(range(12))), ("early 0-2", [0, 1, 2])]:
        prx, _ = participation_ratio(updates_on_common_input(m, hs[0] if sl[0] == 0 else H, sl))
        print(f"  [context] PR over {label:>8}: {prx:.2f} of {len(sl)}")
