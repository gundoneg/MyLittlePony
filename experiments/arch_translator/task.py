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


def sample_tasks_hops(n, vocab, hops, maxjump=4, seed=0):
    """Depth-demanding pointer-chasing tasks. Each = dict(sigma=perm, hops=m, maxjump).

    Why: the (sigma,k) task is a single gather -- one attention layer solves it at any
    lag, so deeper donors leave their deep blocks idle (layer-ablation ~0% drop) and
    those un-pinned deep weights are un-translatable. Pointer-chasing forces depth:
    each position holds a backward jump; following the chain m times needs ~m
    sequential gathers, i.e. ~m layers (one attention layer = one hop). sigma is still
    visible in the embedding/head, so the translator's vocab-equivariant part stays
    meaningful. The pointer rule is fixed; only sigma and chain depth vary.
    """
    g = torch.Generator().manual_seed(seed)
    return [dict(sigma=torch.randperm(vocab, generator=g), hops=hops, maxjump=maxjump)
            for _ in range(n)]


def make_batch(task, bs, ctx, vocab, generator=None):
    """(x, y). Dispatches on task type:
      * (sigma,k):    y_t = sigma(x_{t-k}),                 y_t=IGNORE for t<k
      * pointer-hops: y_t = sigma(x[p_m(t)]),  p_0=t,  p_{j+1}=p_j-1-(x_{p_j} mod J),
                      clamped at 0; y_t=IGNORE for t<hops (chain not yet meaningful).
    """
    if "hops" in task:
        sigma, hops, J = task["sigma"], task["hops"], task["maxjump"]
        x = torch.randint(vocab, (bs, ctx), generator=generator)
        jump = (x % J) + 1                                  # backward jump 1..J
        addr = (torch.arange(ctx) - jump).clamp(min=0)      # each pos -> a past pos
        p = torch.arange(ctx).expand(bs, ctx).clone()
        for _ in range(hops):
            p = torch.gather(addr, 1, p)                    # one backward hop
        y = sigma[torch.gather(x, 1, p)]
        y[:, :hops] = IGNORE                                # warm-up positions
        return x, y
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


# --------------------------- multi-task (M tasks per model) ---------------------------
def sample_task_bundles(n, M, vocab, ks=(1, 2, 3), seed=0):
    """n donors, each a LIST of M tasks the single model must solve in-context."""
    g = torch.Generator().manual_seed(seed)
    bundles = []
    for _ in range(n):
        b = []
        for _ in range(M):
            sigma = torch.randperm(vocab, generator=g)
            k = int(ks[torch.randint(len(ks), (1,), generator=g)])
            b.append(dict(sigma=sigma, k=k))
        bundles.append(b)
    return bundles


def make_multitask_batch(tasks, bs, ctx, vocab, generator=None):
    """One model, M tasks. Position 0 is a task selector token m in [0,M); the rest
    is content. Target y_t = sigma_m(x_{t-k_m}) where t-k references content (>=1)."""
    M = len(tasks)
    m_list = torch.randint(M, (bs,), generator=generator)
    x = torch.randint(vocab, (bs, ctx), generator=generator)
    x[:, 0] = m_list                                   # in-context task selector
    y = torch.full((bs, ctx), IGNORE, dtype=torch.long)
    for j, t in enumerate(tasks):
        rows = (m_list == j).nonzero(as_tuple=True)[0]
        k = t["k"]
        if len(rows) == 0 or ctx <= k + 1:
            continue
        cols = torch.arange(k + 1, ctx)                # valid target positions
        y[rows.unsqueeze(1), cols.unsqueeze(0)] = t["sigma"][x[rows][:, 1:ctx - k]]
    return x, y
