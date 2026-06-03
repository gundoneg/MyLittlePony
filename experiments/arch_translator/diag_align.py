"""Phase 3c diagnostic — set expectations before climbing the alignment ladder.

Two questions:
  1. WHICH SYMMETRY best describes how two donors' frames differ -- a rotation
     (orthogonal Procrustes) or a signed channel permutation (Re-Basin)? Measured
     as the residual after each, on the task-agnostic embedding anchor.
  2. THE CEILING / PRINCIPLE: how alignable two donors are depends on how much
     COMPUTATION they share. Compare the residual after the best rotation of FULL
     pre-norm activations for SAME-task donors vs DIFFERENT-task donors (and, for
     contrast, the embedding-only residual, which cannot tell them apart).
"""
import torch

from models import Cfg
from task import sample_tasks
from donor_zoo import train_donor
from align import (procrustes, residual_activations, embed_activations,
                   align_one, ANCHOR_KEYS)


def rel(a, b):
    return (a - b).norm().item() / (b.norm().item() + 1e-9)


def main():
    cfg = Cfg(vocab=16, d_model=64, n_layer=1, n_head=4, ctx=32, d_ff=256)
    steps = 250
    tasks = sample_tasks(6, cfg.vocab, seed=0)
    shared = tasks[0]
    print(f"alignment diagnostic | V={cfg.vocab} d={cfg.d_model} | donor steps={steps}")

    same = [train_donor(cfg, shared, steps, init_seed=20 + s).state_dict() for s in range(4)]
    diff = [train_donor(cfg, tasks[1 + j], steps, init_seed=30 + j).state_dict() for j in range(4)]
    ref = same[0]
    probe = torch.randint(cfg.vocab, (64, cfg.ctx), generator=torch.Generator().manual_seed(7))

    # 1. which symmetry fits the embedding-anchor relationship?
    def anchor(s):
        return torch.cat([s[k] for k in ANCHOR_KEYS])
    ar = anchor(ref)
    others = same[1:] + diff
    ro = sum(rel(anchor(align_one(s, ref, "ortho")), ar) for s in others) / len(others)
    rp = sum(rel(anchor(align_one(s, ref, "signperm")), ar) for s in others) / len(others)
    print("\n1) symmetry fit on embedding anchor (lower = better explained):")
    print(f"     rotation (Procrustes) residual : {ro:.3f}")
    print(f"     signed-permutation residual    : {rp:.3f}")
    print(f"     -> cross-donor frames look more like a "
          f"{'ROTATION' if ro < rp else 'SIGNED PERMUTATION'}")

    # 2. shared-computation principle: full-activation alignability, same vs diff task
    Hr_full = residual_activations(ref, probe, cfg)
    Hr_emb = embed_activations(ref, probe, cfg)

    def full_resid(s):
        Hi = residual_activations(s, probe, cfg)
        return rel(Hi @ procrustes(Hi, Hr_full), Hr_full)

    def emb_resid(s):
        Hi = embed_activations(s, probe, cfg)
        return rel(Hi @ procrustes(Hi, Hr_emb), Hr_emb)

    sf = sum(full_resid(s) for s in same[1:]) / 3
    df = sum(full_resid(s) for s in diff) / 4
    se = sum(emb_resid(s) for s in same[1:]) / 3
    de = sum(emb_resid(s) for s in diff) / 4
    print("\n2) best-rotation residual, same-task vs different-task donors:")
    print(f"     FULL pre-norm activations : same={sf:.3f}  diff={df:.3f}")
    print(f"     embedding-only activations: same={se:.3f}  diff={de:.3f}")
    print("     -> if FULL same << FULL diff while embedding same ~= diff, the deep")
    print("        residual is alignable only where donors SHARE computation.")


if __name__ == "__main__":
    main()
