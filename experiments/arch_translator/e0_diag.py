"""Phase 10 / E0 step 1 — no-retrain diagnostic: is the phase-8 'coverage ~0' an artifact
of the data_align gauge bug (ROTATE_KEYS rotates only blocks.0.mix.qkv)?

per_block_feature S = P (Wq^T Wk) P^T is invariant to the residual gauge Q ONLY if BOTH
pos and qkv are rotated. data_align rotates pos for all layers but qkv for block 0 only,
so for l>=1 the ALIGNED feature carries a per-donor random Q (corrupted), while the RAW
feature (each donor in its own consistent frame) stays the clean invariant and is
cross-donor comparable. Prediction: raw-coverage >> aligned-coverage for deep blocks.

Cheap: builds small zoos, NO translator training.
"""
import time, torch
from run_cover import make_zoo
from coverage import nn_reconstruction_coverage, subspace_overlap_coverage
from xlate import per_block_feature, _donor_n_layer


def per_layer_norms(zoo, cfg, tag):
    L = _donor_n_layer(zoo[0]["A"])
    print(f"  {tag}: per-layer mean |feature| (over donors)")
    for l in range(L):
        ns = torch.tensor([per_block_feature(z["A"], cfg, l).norm() for z in zoo])
        print(f"    layer {l}: {ns.mean():.2f} +/- {ns.std():.2f}")


def cov(shallow, deep, cfg, tag):
    m, w = nn_reconstruction_coverage(shallow, deep, cfg)
    sub = subspace_overlap_coverage(shallow, deep, cfg)
    print(f"  [{tag}] NN cosine mean {m:.3f}  worst-case {w:.3f}  | subspace {sub:.3f}")
    return m, w


if __name__ == "__main__":
    t0 = time.time()
    n = 8
    ds = 250                                   # light training: gauge bug is training-independent
    print("E0 diagnostic: raw vs aligned per-block coverage (L2 shallow -> L3 deep)\n")
    cfg_s, raw_s, aln_s = make_zoo(2, n, seed=7, realistic=True, donor_steps=ds)
    cfg_d, raw_d, aln_d = make_zoo(3, n, seed=200, realistic=True, donor_steps=ds)
    print(f"  donor-acc shallow {sum(z['acc'] for z in raw_s)/n:.0%} | deep {sum(z['acc'] for z in raw_d)/n:.0%}\n")

    ntr = 5
    print("COVERAGE shallow-train -> deep-target:")
    cov(raw_s[:ntr], raw_d[ntr:], cfg_d, "RAW   (clean invariant)")
    cov(aln_s[:ntr], aln_d[ntr:], cfg_d, "ALIGNED (current pipeline = buggy)")

    print()
    per_layer_norms(raw_d, cfg_d, "RAW deep")
    per_layer_norms(aln_d, cfg_d, "ALIGNED deep")
    print(f"\ntotal {time.time()-t0:.0f}s")
