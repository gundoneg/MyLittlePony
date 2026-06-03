"""Phase 3b — find a shared coordinate frame for ALREADY-TRAINED donors.

Independent donors each live in their own random residual-stream basis, so the
translator's shared vocab maps cannot work and zero-shot transfer dies. Here we
align every donor to a reference donor R *post hoc*, with no retraining:

  estimate an orthogonal Q (d x d) that rotates donor i's frame onto R's, using a
  TASK-AGNOSTIC anchor that is row-comparable across donors -- the input token
  embedding and the positional embedding (same token/position identities for all
  donors; no sigma, no cipher). Then express everything the translator reads
  (tok, pos, head, layer-0 qkv) in R's frame.

Note: the lag feature S_pos = P (Wq^T Wk) P^T is already invariant to an orthogonal
frame change, so alignment is really about putting the VOCAB maps into one frame --
exactly what the equivariant W_t/W_h/W_p need.
"""
import torch

from xlate import VOCAB_KEYS  # noqa: F401  (kept for reference of what C reads)

ANCHOR_KEYS = ("tok.weight", "pos.weight")   # task-agnostic, row-comparable
ROTATE_KEYS = ("tok.weight", "pos.weight", "head.weight", "blocks.0.mix.qkv.weight")


def procrustes(src, ref):
    """Orthogonal Q minimising ||src @ Q - ref||_F (rows of src/ref correspond)."""
    U, _, Vt = torch.linalg.svd(src.t() @ ref)
    return U @ Vt


def estimate_Q(state, ref_state):
    src = torch.cat([state[k] for k in ANCHOR_KEYS], dim=0)   # (V+ctx, d)
    ref = torch.cat([ref_state[k] for k in ANCHOR_KEYS], dim=0)
    return procrustes(src, ref)


def gauge_align(state, ref_state):
    """Return a copy of `state` rotated into `ref_state`'s frame (only the matrices
    the translator reads are rotated; the rest is untouched/unused)."""
    Q = estimate_Q(state, ref_state)
    out = dict(state)
    for k in ROTATE_KEYS:
        out[k] = state[k] @ Q          # input/frame-side rotation (columns = residual dim)
    return out


def align_zoo(zoo, ref_state):
    return [dict(task=z["task"], A=gauge_align(z["A"], ref_state), acc=z["acc"])
            for z in zoo]
