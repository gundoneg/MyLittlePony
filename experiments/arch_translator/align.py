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

from hungarian import assign_from_similarity
from models import LM
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


def _apply_cols(state, op):
    """Return a copy with `op` applied to the residual (column) dim of read matrices."""
    out = dict(state)
    for k in ROTATE_KEYS:
        out[k] = op(state[k])
    return out


# --- method 2: signed-permutation (Re-Basin), the architecture-correct gauge ---
def align_signperm(state, ref_state):
    """Match residual CHANNELS of the donor to the reference by channel-profile
    similarity (with sign), via Hungarian. Exact symmetry group of an RMSNorm net."""
    a_i = torch.cat([state[k] for k in ANCHOR_KEYS], dim=0)      # (N, d)
    a_r = torch.cat([ref_state[k] for k in ANCHOR_KEYS], dim=0)
    ci = a_i / (a_i.norm(dim=0, keepdim=True) + 1e-9)            # unit channel profiles
    cr = a_r / (a_r.norm(dim=0, keepdim=True) + 1e-9)
    S = ci.t() @ cr                                              # (d, d) cos(donor a, ref b)
    assign = torch.as_tensor(assign_from_similarity(S.abs().numpy()))
    d = S.shape[0]
    signs = torch.sign(S[torch.arange(d), assign])
    signs[signs == 0] = 1.0

    def op(W):                                                  # place donor col a (signed) at ref col assign[a]
        new = torch.zeros_like(W)
        new[:, assign] = W * signs
        return new
    return _apply_cols(state, op)


# --- method 3/4: activation-based Procrustes ---
def embed_activations(state, probe_x, cfg):
    """z0 = tok[probe] + pos : task-agnostic embedding activations (N, d)."""
    z = state["tok.weight"][probe_x] + state["pos.weight"][:probe_x.shape[1]][None]
    return z.reshape(-1, cfg.d_model)


def residual_activations(state, probe_x, cfg):
    """Full pre-final-norm residual over a shared probe (N, d). Spans all d only
    when donors share the computation that populates it (e.g. same task)."""
    m = LM(cfg, "transformer")
    m.load_state_dict(state)
    m.eval()
    cap = {}
    h = m.norm.register_forward_pre_hook(lambda mod, inp: cap.__setitem__("h", inp[0]))
    with torch.no_grad():
        m(probe_x)
    h.remove()
    return cap["h"].reshape(-1, cfg.d_model)


def align_activation(state, ref_state, cfg, probe_x, full=False):
    act = residual_activations if full else embed_activations
    Hi = act(state, probe_x, cfg)
    Hr = act(ref_state, probe_x, cfg)
    Q = procrustes(Hi, Hr)
    return _apply_cols(state, lambda W: W @ Q)


# --------------------------------- dispatcher ---------------------------------
def align_one(state, ref_state, method="ortho", cfg=None, probe_x=None):
    if method == "ortho":
        return gauge_align(state, ref_state)
    if method == "signperm":
        return align_signperm(state, ref_state)
    if method == "act":
        return align_activation(state, ref_state, cfg, probe_x, full=False)
    if method == "act_full":
        return align_activation(state, ref_state, cfg, probe_x, full=True)
    raise ValueError(method)


def align_zoo(zoo, ref_state, method="ortho", cfg=None, probe_x=None):
    return [dict(task=z["task"],
                 A=align_one(z["A"], ref_state, method, cfg, probe_x), acc=z["acc"])
            for z in zoo]
