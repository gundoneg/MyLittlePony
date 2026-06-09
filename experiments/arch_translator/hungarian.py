"""Optimal assignment (Hungarian / Kuhn-Munkres) in pure numpy.

scipy is unavailable in this environment, so we ship a compact O(n^3) solver
(the classic e-maxx potential method). Used to match donor vocab rows to the
canonical donor's rows when recovering the cipher permutation sigma.
"""
import numpy as np


def hungarian(cost):
    """Minimum-cost assignment for a square cost matrix.

    Returns (assign, total) where assign[row] = chosen column, and total is the
    summed cost. Pass cost = -similarity to maximise a similarity instead.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    assert n == m, "square cost matrix expected"
    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)      # p[col] = row matched to this column
    way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assign = np.zeros(n, dtype=int)
    for j in range(1, n + 1):
        assign[p[j] - 1] = j - 1
    total = float(cost[np.arange(n), assign].sum())
    return assign, total


def assign_from_similarity(sim):
    """Convenience: best assignment maximising a similarity matrix (numpy/torch)."""
    sim = np.asarray(sim, dtype=np.float64)
    assign, _ = hungarian(-sim)
    return assign


if __name__ == "__main__":
    # tiny self-test: a known permutation must be recovered exactly
    rng = np.random.default_rng(0)
    n = 12
    perm = rng.permutation(n)
    base = rng.standard_normal((n, 8))
    sim = base @ base[perm].T          # sim[i,j] peaks at j = inv(perm)[i]
    want = np.argsort(perm)            # inv(perm)
    got = assign_from_similarity(sim)
    ok = (got == want).all()
    print("hungarian self-test:", "OK" if ok else "FAIL", "| recovered", got.tolist())
