"""Finite-difference grad-check для autograd. Критично для корректности.

Запуск: python test_autograd.py
"""
import numpy as np
from autograd import Tensor, concat, stack, rmsnorm, cross_entropy

rng = np.random.default_rng(0)


def numeric_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        fp = f(x)
        x[i] = old - eps
        fm = f(x)
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


def check(name, build, *arrays, tol=1e-4):
    """build(*tensors)->scalar Tensor. Сверяем аналитич. и числ. градиенты."""
    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = build(*tensors)
    out.backward()
    ok = True
    for k, (t, a) in enumerate(zip(tensors, arrays)):
        def f(x, k=k):
            ts = [Tensor(arr.copy()) for arr in arrays]
            ts[k] = Tensor(x)
            return float(build(*ts).data)
        ng = numeric_grad(f, a.copy())
        err = np.max(np.abs(ng - t.grad)) / (np.max(np.abs(ng)) + 1e-8)
        if err > tol:
            ok = False
            print(f"  [FAIL] {name} arg{k}: rel_err={err:.2e}")
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
    return ok


def main():
    results = []
    A = rng.standard_normal((4, 5))
    B = rng.standard_normal((5, 3))
    C = rng.standard_normal((4, 5))

    results.append(check("add", lambda a, b: (a + b).sum(), A, C))
    results.append(check("mul", lambda a, b: (a * b).sum(), A, C))
    results.append(check("sub", lambda a, b: (a - b).sum(), A, C))
    results.append(check("div", lambda a, b: (a / (b * b + 2.0)).sum(), A, C))
    results.append(check("matmul", lambda a, b: (a @ b).sum(), A, B))
    results.append(check("matmul_bcast",
                         lambda a, b: (a @ b).sum(),
                         rng.standard_normal((2, 4, 5)), B))
    results.append(check("mean", lambda a: a.mean(), A))
    results.append(check("mean_axis", lambda a: a.mean(axis=-1).sum(), A))
    results.append(check("relu", lambda a: a.relu().sum(), A))
    results.append(check("sigmoid", lambda a: a.sigmoid().sum(), A))
    results.append(check("silu", lambda a: a.silu().sum(), A))
    results.append(check("exp", lambda a: (a * 0.1).exp().sum(), A))
    results.append(check("log", lambda a: (a * a + 1.0).log().sum(), A))
    results.append(check("reshape", lambda a: a.reshape(2, 10).sum(), A))
    results.append(check("transpose", lambda a: (a.T @ a).sum(), A))
    results.append(check("softmax", lambda a: (a.softmax(axis=-1) * Tensor(C)).sum(), A))
    results.append(check("rmsnorm",
                         lambda x, w: rmsnorm(x, w).sum(),
                         A, rng.standard_normal((5,))))
    results.append(check("concat",
                         lambda a, b: concat([a, b], axis=0).sum(), A, C))
    results.append(check("stack",
                         lambda a, b: stack([a, b], axis=0).sum(), A, C))

    # gather_rows
    E = rng.standard_normal((6, 4))
    idx = np.array([0, 3, 5, 1])
    W = rng.standard_normal((4, 2))
    results.append(check("gather_rows",
                         lambda e, w: (e.gather_rows(idx) @ w).sum(), E, W))

    # cross_entropy
    L = rng.standard_normal((4, 5))
    tgt = np.array([0, 2, 4, 1])
    results.append(check("cross_entropy",
                         lambda l: cross_entropy(l, tgt), L))

    print("\n%d/%d passed" % (sum(results), len(results)))
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
