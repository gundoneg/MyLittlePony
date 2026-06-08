"""Phase 9 finale — type-coverage -> depth transfer on the REAL target (supra50m).

The scheme under test (the user's): instead of matching a target's depth, COVER its
repertoire of layer 'types' with a small zoo of exemplars, then reproduce the full-depth
network by grafting exemplars into every slot. Phase-9 measurements predict real models
need only a few types (a couple unique early layers + one redundant deep type).

Operationalization (weights-only graft, fully functional via llama_min.remap):
  G[i,j] = KL(full || run slot i with layer j's weights)           -- coverage cost
  zoo Z  : a set of exemplar layers. Reconstruct depth-12 by assigning every slot its
           best-covering exemplar in Z (argmin_j in Z G[i,j]) and grafting ALL at once.
  curve  : end-to-end KL of the reconstructed 12-layer model vs |Z|, exemplars chosen
           greedily. Few exemplars reaching ~0 == 'cover the types, get arbitrary depth'.
"""
import torch
from llama_min import MinLlama, load_vocab, decode
from entangle import kl


@torch.no_grad()
def coverage_matrix(m, idx, full):
    G = torch.zeros(m.L, m.L)
    for i in range(m.L):
        for j in range(m.L):
            G[i, j] = 0.0 if i == j else kl(full, m.forward(idx, remap={i: j}))
    return G


@torch.no_grad()
def reconstruct_kl(m, idx, full, Z):
    """Assign each slot its best exemplar in Z, graft all, return end-to-end KL + map."""
    remap = {}
    for i in range(m.L):
        e = min(Z, key=lambda j: GMAT[i, j].item())
        remap[i] = e
    return kl(full, m.forward(idx, remap=remap)), remap


def greedy_zoo(m, idx, full, K):
    Z, curve = [], []
    for _ in range(K):
        cand = [l for l in range(m.L) if l not in Z]
        best = min(cand, key=lambda c: reconstruct_kl(m, idx, full, Z + [c])[0])
        Z.append(best)
        k_kl, _ = reconstruct_kl(m, idx, full, Z)
        curve.append((len(Z), best, k_kl))
    return Z, curve


if __name__ == "__main__":
    m = MinLlama()
    inv = load_vocab()
    torch.manual_seed(0)
    seqs = [m.generate(torch.tensor([[1]]), 160, temperature=0.9)[:, 1:] for _ in range(3)]
    idx = torch.cat(seqs, 0)
    full = m.forward(idx)
    print(f"supra50m  L={m.L}  (real 50M Llama) — type-coverage -> depth transfer\n")

    GMAT = coverage_matrix(m, idx, full)
    # show the graft-cost matrix (rows=slot i, cols=source j); look for a redundant block
    print("graft-cost KL[i<-j]  (rows=slot, cols=source; '.'<0.5 => j covers i)")
    print("      " + " ".join(f"{j:>4}" for j in range(m.L)))
    for i in range(m.L):
        cells = " ".join((" .  " if GMAT[i, j] < 0.5 else f"{GMAT[i,j]:4.1f}") for j in range(m.L))
        print(f"  i{i:>2} {cells}")

    # how many slots can a single deep exemplar cover?
    deep = 6
    covered = (GMAT[:, deep] < 0.5).sum().item()
    print(f"\none deep exemplar (layer {deep}) grafts cleanly into {covered}/{m.L} slots (KL<0.5)")

    # diagnostic: tile ONE best deep exemplar into the whole redundant interior (slots 3-10),
    # keeping the unique types (0,1,2,11) exact. Tests 'is the deep block one tileable type?'
    uniq = {0, 1, 2, 11}
    interior = [s for s in range(m.L) if s not in uniq]
    best_e = min(range(3, 11), key=lambda e: sum(GMAT[i, e].item() for i in interior))
    tile = {s: best_e for s in interior}
    tile_kl = kl(full, m.forward(idx, remap=tile))
    sum_single = sum(GMAT[i, best_e].item() for i in interior)
    print(f"\ndeep-tile: copy layer {best_e} into all interior slots {interior}, keep {sorted(uniq)} exact")
    print(f"  sum of INDEPENDENT single-graft costs = {sum_single:.2f}")
    print(f"  actual JOINT end-to-end KL            = {tile_kl:.2f}   (compounding => not one tileable type)")

    Z, curve = greedy_zoo(m, idx, full, K=6)
    print(f"\ngreedy type-zoo — reconstruct ALL {m.L} layers from k exemplars:")
    print(f"  {'k':>2} | {'added':>5} | {'end-to-end KL':>13} | zoo")
    for k, added, kkl in curve:
        print(f"  {k:>2} | {added:>5} | {kkl:>13.3f} | {sorted(Z[:k])}")

    # generation from a small-zoo reconstruction (sanity that text stays coherent)
    k_small = 4
    _, remap = reconstruct_kl(m, idx, full, Z[:k_small])
    torch.manual_seed(1)
    # sample through the remapped (reconstructed) stack
    seq = torch.tensor([[1]])
    for _ in range(50):
        lo = m.forward(seq[:, -256:], remap=remap)[:, -1, :] / 0.7
        v, _ = torch.topk(lo, 40); lo[lo < v[:, [-1]]] = float("-inf")
        seq = torch.cat([seq, torch.multinomial(lo.softmax(-1), 1)], 1)
    print(f"\n{k_small}-type reconstruction (zoo={sorted(Z[:k_small])}) sample:")
    print("  ", repr(decode(seq[0], inv)[:240]))
