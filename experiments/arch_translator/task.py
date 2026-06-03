"""Phase 3 — the visible axis: a synthetic task family tau = (sigma, k).

Given a random token sequence x_1..x_T over a small vocab, the target at position t
is  y_t = sigma(x_{t-k})  -- apply substitution sigma to the token k steps back.
Positions t < k have no defined target (ignore_index = -1).

This leaves a clear, weight-visible trace: sigma in the embedding/unembedding
(via G = tok . head^T), and the lag k in the positional/attention structure.
Both the transformer and the SSM (which has a short causal conv) can learn it.
"""
import torch
import torch.nn.functional as F

IGNORE = -1


def sample_tasks(n, vocab, ks=(1, 2, 3), seed=0):
    """A list of tasks; each is dict(sigma=perm over vocab, k=lag)."""
    g = torch.Generator().manual_seed(seed)
    tasks = []
    for _ in range(n):
        sigma = torch.randperm(vocab, generator=g)
        k = int(ks[torch.randint(len(ks), (1,), generator=g)])
        tasks.append(dict(sigma=sigma, k=k))
    return tasks


def make_batch(task, bs, ctx, vocab, generator=None):
    """(x, y) with y_t = sigma(x_{t-k}); y_t = IGNORE for t < k."""
    sigma, k = task["sigma"], task["k"]
    x = torch.randint(vocab, (bs, ctx), generator=generator)
    y = torch.full((bs, ctx), IGNORE, dtype=torch.long)
    if ctx > k:
        y[:, k:] = sigma[x[:, :ctx - k]]
    return x, y


def masked_ce(logits, y):
    V = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=IGNORE)


@torch.no_grad()
def task_accuracy(logits, y):
    """Top-1 accuracy over the valid (non-ignored) positions."""
    pred = logits.argmax(-1)
    valid = y != IGNORE
    return (pred[valid] == y[valid]).float().mean().item()
