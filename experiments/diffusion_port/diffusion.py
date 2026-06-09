"""Model-agnostic core for adapting an autoregressive (AR) decoder-only
transformer into a Mercury-like DISCRETE-DIFFUSION language model.

Recipe (DiffuLLaMA / Dream / LLaDA):
  * bidirectional attention with a causal->full annealing schedule,
  * an absorbing-state ("[MASK]") masked-diffusion training objective with the
    LLaDA 1/t ELBO weight,
  * an iterative parallel denoising sampler (confidence-based unmasking).

Nothing here is specific to the toy `LM` or to HuggingFace -- the same
functions are exercised on the CPU toy model in `toy_adapt.py` /
`test_diffusion.py` and on Qwen3.5-0.8B in `qwen_adapt.py`. torch + stdlib only.

References:
  LLaDA           -- Nie et al., arXiv:2502.09992 (eq. for q_{t|0} and the ELBO).
  DiffuLLaMA      -- Gong et al., arXiv:2410.17891 (attention-mask annealing).
  Dream 7B        -- arXiv:2508.15487 (confidence/entropy remasking sampler).
"""
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Noise schedule
# --------------------------------------------------------------------------
def sample_mask_rate(batch_size, *, eps=1e-3, generator=None, device="cpu"):
    """Per-sequence mask probability t ~ U(eps, 1].

    eps>0 keeps the 1/t ELBO weight finite. Returns shape (batch_size,).
    """
    u = torch.rand(batch_size, generator=generator, device=device)
    return eps + (1.0 - eps) * u


def anneal_causal_prob(step, total, *, schedule="linear"):
    """Probability that a sequence still uses the *causal* mask at `step`.

    1.0 -> fully causal (preserves the pretrained AR behaviour early on),
    0.0 -> fully bidirectional (the diffusion regime). DiffuLLaMA anneals the
    right-context in; we use the equivalent per-sequence Bernoulli mixture
    (same end state, cheaper transient). `step` is clamped into [0, total].
    """
    if total <= 0:
        return 0.0
    frac = min(max(step, 0), total) / total
    if schedule == "linear":
        rho = 1.0 - frac
    elif schedule == "cosine":
        import math
        rho = 0.5 * (1.0 + math.cos(math.pi * frac))
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    return float(max(0.0, min(1.0, rho)))


# --------------------------------------------------------------------------
# Forward noising (absorbing state)
# --------------------------------------------------------------------------
def forward_mask(x0, t, mask_id, *, generator=None):
    """Absorbing-state forward process q_{t|0}.

    Each position is independently replaced by `mask_id` with probability t[b].
    Guarantees >=1 masked position per row (otherwise the loss for that row is
    undefined). Returns (x_t, is_masked) with is_masked a bool tensor.

    x0: (B, L) long, t: (B,) float in (0, 1].
    """
    B, L = x0.shape
    probs = t[:, None].expand(B, L)
    noise = torch.rand(B, L, generator=generator, device=x0.device)
    is_masked = noise < probs
    # Force at least one masked token per row: where a row is all-False, mask
    # its argmin-noise position (the position closest to being masked).
    empty = ~is_masked.any(dim=1)
    if empty.any():
        force_pos = noise[empty].argmin(dim=1)
        rows = empty.nonzero(as_tuple=True)[0]
        is_masked[rows, force_pos] = True
    x_t = torch.where(is_masked, torch.full_like(x0, mask_id), x0)
    return x_t, is_masked


# --------------------------------------------------------------------------
# Training objective (LLaDA ELBO)
# --------------------------------------------------------------------------
def diffusion_loss(logits, x0, is_masked, t, *, reduction="mean"):
    """Masked cross-entropy with the LLaDA 1/t ELBO weight.

        L = E_t[ (1/t) * sum_i 1[masked_i] * -log p(x0_i | x_t) / L ]

    Dividing by the full length L (not by the masked count) keeps this an
    unbiased ELBO estimator (matches nanoLLaDA's CE/t). The 1/t weight is the
    load-bearing difference from a plain BERT-style MLM loss.

    logits: (B, L, V), x0: (B, L), is_masked: (B, L) bool, t: (B,).
    """
    B, L, V = logits.shape
    ce = F.cross_entropy(
        logits.reshape(-1, V), x0.reshape(-1), reduction="none"
    ).view(B, L)
    per_seq = (ce * is_masked).sum(dim=1) / (t * L)
    if reduction == "mean":
        return per_seq.mean()
    if reduction == "sum":
        return per_seq.sum()
    if reduction == "none":
        return per_seq
    raise ValueError(f"unknown reduction {reduction!r}")


# --------------------------------------------------------------------------
# Attention-mode control (toy models)
# --------------------------------------------------------------------------
def set_bidirectional(model, *, causal):
    """Walk `model.modules()` and set `.causal` on every attention module that
    exposes it (the toy `Attention`). Returns the number of modules touched.

    Lets a driver flip causal<->bidirectional at runtime (e.g. for annealing)
    without rebuilding the model.
    """
    n = 0
    for m in model.modules():
        if hasattr(m, "causal"):
            m.causal = causal
            n += 1
    return n


# --------------------------------------------------------------------------
# Confidence scoring for the sampler
# --------------------------------------------------------------------------
def _confidence(probs, alg):
    """Per-position confidence score (higher == more confident), shape (B, L)."""
    if alg in ("maskgit_plus", "origin"):
        return probs.max(dim=-1).values
    if alg == "topk_margin":
        top2 = probs.topk(2, dim=-1).values
        return top2[..., 0] - top2[..., 1]
    if alg == "entropy":
        ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1)
        return -ent
    raise ValueError(f"unknown alg {alg!r}")


# --------------------------------------------------------------------------
# Iterative parallel denoising sampler
# --------------------------------------------------------------------------
@torch.no_grad()
def diffusion_generate(forward_fn, *, length, mask_id, steps=64, prompt_ids=None,
                       temperature=0.0, alg="maskgit_plus", alg_temp=0.0,
                       batch_size=1, generator=None, device="cpu"):
    """Generate by iterative confidence-based unmasking (LLaDA/Dream reverse).

    forward_fn(ids) -> logits (B, L, V); MUST be bidirectional & no-grad-safe.
    Start from an all-[MASK] sequence (prompt positions, if any, are pinned and
    never re-masked). Over `steps` rounds, reveal the highest-confidence masked
    tokens until none remain. Returns (B, length) long.

    prompt_ids: optional (B, P) long prefix; the continuation is length-P long.
    temperature: 0 -> argmax decode; >0 -> sample from softmax(logits/T).
    alg_temp: >0 adds Gumbel noise to the confidence ranking (stochastic order).
    """
    if prompt_ids is not None:
        B, P = prompt_ids.shape
        ids = torch.full((B, length), mask_id, dtype=torch.long, device=device)
        ids[:, :P] = prompt_ids.to(device)
        frozen = torch.zeros(B, length, dtype=torch.bool, device=device)
        frozen[:, :P] = True
    else:
        B = batch_size
        ids = torch.full((B, length), mask_id, dtype=torch.long, device=device)
        frozen = torch.zeros(B, length, dtype=torch.bool, device=device)

    for s in range(steps):
        masked = (ids == mask_id) & ~frozen
        n_left = int(masked.sum().item())
        if n_left == 0:
            break
        logits = forward_fn(ids)
        # The model may have [MASK] as an output class; never emit it.
        logits[..., mask_id] = float("-inf")
        if temperature and temperature > 0:
            probs = (logits / temperature).softmax(dim=-1)
            flat = probs.view(-1, probs.size(-1))
            pred = torch.multinomial(flat, 1, generator=generator).view(B, length)
        else:
            probs = logits.softmax(dim=-1)
            pred = probs.argmax(dim=-1)

        conf = _confidence(probs, alg)
        # Only masked positions are eligible to be revealed this round.
        conf = conf.masked_fill(~masked, float("-inf"))
        if alg_temp and alg_temp > 0:
            g = torch.rand(conf.shape, generator=generator, device=device).clamp_min(1e-12)
            gumbel = -(-g.log()).log()
            conf = conf + alg_temp * gumbel

        # Reveal ~ (remaining masked) / (remaining steps) tokens this round,
        # at least one, so we always finish within `steps` rounds.
        steps_left = steps - s
        k = max(1, n_left // steps_left)
        flat_conf = conf.view(-1)
        k = min(k, n_left)
        reveal = flat_conf.topk(k).indices
        flat_ids = ids.view(-1)
        flat_pred = pred.view(-1)
        flat_ids[reveal] = flat_pred[reveal]

    # Final safety: if any masks remain (e.g. ties), fill them with the argmax.
    masked = (ids == mask_id) & ~frozen
    if masked.any():
        logits = forward_fn(ids)
        logits[..., mask_id] = float("-inf")
        pred = logits.argmax(dim=-1)
        ids = torch.where(masked, pred, ids)
    return ids
