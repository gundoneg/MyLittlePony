"""Phase 8 — coverage metric: does a shallow donor zoo's per-block feature distribution
SPAN the per-block features of a deep target?

`PerLayerTranslator` (xlate.py) emits each block l as a LOCAL function of
`per_block_feature(A, l)` = the sigma-free positional attention signature
`P @ (Wq^T Wk) @ P^T`. So a shallow-trained C can only translate a deep block whose
feature falls inside the feature distribution it was trained on. This module measures
exactly that overlap, on the same vectors C consumes. The feature is invariant to the
data-align orthogonal rotation (P@Q (Q^T M Q) Q^T@P^T = P M P^T), so coverage is a
frame-independent property of the weights.
"""
import torch
from xlate import per_block_feature, _donor_n_layer


def block_feature_bank(zoo, cfg):
    """Stack per-block features over (donor, layer): (n_blocks, ctx*ctx)."""
    feats = []
    for z in zoo:
        for l in range(_donor_n_layer(z["A"])):
            feats.append(per_block_feature(z["A"], cfg, l))
    return torch.stack(feats)


def nn_reconstruction_coverage(shallow_zoo, deep_zoo, cfg):
    """For each deep block feature, the nearest-neighbour cosine to any shallow block
    feature. Returns (mean, worst-case-min). Worst-case is the diagnostic: the
    hypothesis predicts transfer fails exactly when some deep block has NO near
    neighbour in the shallow bank (a 'deep-only' layer type)."""
    S = block_feature_bank(shallow_zoo, cfg)
    D = block_feature_bank(deep_zoo, cfg)
    Sn = S / (S.norm(dim=1, keepdim=True) + 1e-9)
    Dn = D / (D.norm(dim=1, keepdim=True) + 1e-9)
    best = (Dn @ Sn.t()).max(dim=1).values            # nearest cosine per deep block
    return best.mean().item(), best.min().item()


def subspace_overlap_coverage(shallow_zoo, deep_zoo, cfg, energy=0.95):
    """Fraction of deep-feature energy captured by the shallow zoo's principal
    subspace (PCA span overlap). 1.0 => deep features live in the shallow span."""
    S = block_feature_bank(shallow_zoo, cfg)
    D = block_feature_bank(deep_zoo, cfg)
    Sc = S - S.mean(0)
    Vt = torch.linalg.svd(Sc, full_matrices=False)[2]  # (r, F) right singular vecs
    sv = torch.linalg.svdvals(Sc)
    k = int((torch.cumsum(sv ** 2, 0) / (sv ** 2).sum() < energy).sum()) + 1
    B = Vt[:k].t()                                     # (F, k) shallow principal basis
    Dc = D - D.mean(0)
    proj = Dc @ B @ B.t()
    return (proj.pow(2).sum() / (Dc.pow(2).sum() + 1e-9)).item()
