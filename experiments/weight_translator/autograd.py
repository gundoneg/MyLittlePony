"""Минимальный reverse-mode autograd поверх NumPy.

Только то, что нужно для крошечных моделей этого эксперимента. Никаких
зависимостей кроме numpy => запускается на телефоне (Termux/OnePlus 15).

Tensor оборачивает np.ndarray и хранит ленту (граф) для обратного прохода.
Градиенты накапливаются в .grad. Поддержан broadcast (через unbroadcast).
"""
from __future__ import annotations
import numpy as np

def _unbroadcast(grad, shape):
    """Свернуть grad к форме shape (обратная операция к numpy-broadcast)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    __slots__ = ("data", "grad", "_backward", "_prev", "requires_grad")

    def __init__(self, data, requires_grad=False, _children=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)

    # --- helpers ---------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def _ensure(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)

    def _result(self, data, parents, backward):
        rg = any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=rg, _children=parents)
        if rg:
            out._backward = backward
        return out

    @staticmethod
    def _acc(t, g):
        if not t.requires_grad:
            return
        g = _unbroadcast(g, t.data.shape)
        t.grad = g if t.grad is None else t.grad + g

    # --- ops -------------------------------------------------------------
    def __add__(self, other):
        other = self._ensure(other)
        out = self._result(self.data + other.data, (self, other), None)
        def bw():
            Tensor._acc(self, out.grad)
            Tensor._acc(other, out.grad)
        if out.requires_grad:
            out._backward = bw
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = self._ensure(other)
        out = self._result(self.data * other.data, (self, other), None)
        def bw():
            Tensor._acc(self, out.grad * other.data)
            Tensor._acc(other, out.grad * self.data)
        if out.requires_grad:
            out._backward = bw
        return out

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (self._ensure(other) * -1.0)

    def __rsub__(self, other):
        return (self * -1.0) + other

    def __truediv__(self, other):
        other = self._ensure(other)
        out = self._result(self.data / other.data, (self, other), None)
        def bw():
            Tensor._acc(self, out.grad / other.data)
            Tensor._acc(other, -out.grad * self.data / (other.data ** 2))
        if out.requires_grad:
            out._backward = bw
        return out

    def matmul(self, other):
        other = self._ensure(other)
        out = self._result(self.data @ other.data, (self, other), None)
        def bw():
            g = out.grad
            a, b = self.data, other.data
            if b.ndim == 1 and a.ndim == 2:        # (m,k)@(k,)->(m,)
                Tensor._acc(self, np.outer(g, b))
                Tensor._acc(other, a.T @ g)
            elif a.ndim == 1 and b.ndim == 2:      # (k,)@(k,n)->(n,)
                Tensor._acc(self, b @ g)
                Tensor._acc(other, np.outer(a, g))
            else:                                  # батч/2D-матмул
                Tensor._acc(self, g @ np.swapaxes(b, -1, -2))
                Tensor._acc(other, np.swapaxes(a, -1, -2) @ g)
        if out.requires_grad:
            out._backward = bw
        return out

    __matmul__ = matmul

    def sum(self, axis=None, keepdims=False):
        out = self._result(self.data.sum(axis=axis, keepdims=keepdims), (self,), None)
        def bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            Tensor._acc(self, np.broadcast_to(g, self.data.shape))
        if out.requires_grad:
            out._backward = bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else (
            np.prod([self.data.shape[a] for a in (axis if isinstance(axis, tuple) else (axis,))]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(n))

    def relu(self):
        out = self._result(np.maximum(self.data, 0.0), (self,), None)
        def bw():
            Tensor._acc(self, out.grad * (self.data > 0))
        if out.requires_grad:
            out._backward = bw
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = self._result(s, (self,), None)
        def bw():
            Tensor._acc(self, out.grad * s * (1.0 - s))
        if out.requires_grad:
            out._backward = bw
        return out

    def silu(self):
        # x * sigmoid(x)
        s = 1.0 / (1.0 + np.exp(-self.data))
        val = self.data * s
        out = self._result(val, (self,), None)
        def bw():
            # d/dx [x*sigm] = sigm + x*sigm*(1-sigm)
            Tensor._acc(self, out.grad * (s + self.data * s * (1.0 - s)))
        if out.requires_grad:
            out._backward = bw
        return out

    def exp(self):
        e = np.exp(self.data)
        out = self._result(e, (self,), None)
        def bw():
            Tensor._acc(self, out.grad * e)
        if out.requires_grad:
            out._backward = bw
        return out

    def log(self):
        out = self._result(np.log(self.data), (self,), None)
        def bw():
            Tensor._acc(self, out.grad / self.data)
        if out.requires_grad:
            out._backward = bw
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        old = self.data.shape
        out = self._result(self.data.reshape(shape), (self,), None)
        def bw():
            Tensor._acc(self, out.grad.reshape(old))
        if out.requires_grad:
            out._backward = bw
        return out

    def transpose(self, axes=None):
        out = self._result(np.transpose(self.data, axes), (self,), None)
        def bw():
            if axes is None:
                Tensor._acc(self, np.transpose(out.grad))
            else:
                inv = np.argsort(axes)
                Tensor._acc(self, np.transpose(out.grad, inv))
        if out.requires_grad:
            out._backward = bw
        return out

    @property
    def T(self):
        return self.transpose()

    def softmax(self, axis=-1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=axis, keepdims=True)
        out = self._result(p, (self,), None)
        def bw():
            g = out.grad
            # jacobian-vector: p * (g - sum(g*p))
            dot = (g * p).sum(axis=axis, keepdims=True)
            Tensor._acc(self, p * (g - dot))
        if out.requires_grad:
            out._backward = bw
        return out

    def gather_rows(self, idx):
        """Выбор строк (embedding lookup). idx: np.ndarray целых, любой формы.
        Результат формы idx.shape + (self.shape[1],)."""
        idx = np.asarray(idx)
        out = self._result(self.data[idx], (self,), None)
        def bw():
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)
            Tensor._acc(self, g)
        if out.requires_grad:
            out._backward = bw
        return out

    # --- backward --------------------------------------------------------
    def backward(self):
        topo, seen = [], set()
        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for c in v._prev:
                build(c)
            topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def concat(tensors, axis=0):
    tensors = list(tensors)
    out_data = np.concatenate([t.data for t in tensors], axis=axis)
    rg = any(t.requires_grad for t in tensors)
    out = Tensor(out_data, requires_grad=rg, _children=tensors)
    if rg:
        sizes = [t.data.shape[axis] for t in tensors]
        bounds = np.cumsum([0] + sizes)
        def bw():
            g = out.grad
            for i, t in enumerate(tensors):
                sl = [slice(None)] * g.ndim
                sl[axis] = slice(bounds[i], bounds[i + 1])
                Tensor._acc(t, g[tuple(sl)])
        out._backward = bw
    return out


def stack(tensors, axis=0):
    tensors = list(tensors)
    out_data = np.stack([t.data for t in tensors], axis=axis)
    rg = any(t.requires_grad for t in tensors)
    out = Tensor(out_data, requires_grad=rg, _children=tensors)
    if rg:
        def bw():
            gs = np.moveaxis(out.grad, axis, 0)
            for i, t in enumerate(tensors):
                Tensor._acc(t, gs[i])
        out._backward = bw
    return out


def rmsnorm(x: Tensor, weight: Tensor, eps=1e-5):
    """RMSNorm по последней оси: x / sqrt(mean(x^2)+eps) * weight."""
    ms = (x * x).mean(axis=-1, keepdims=True)
    inv = (ms + eps)
    # inv^{-1/2} через примитивы: exp(-0.5*log(inv))
    rms = (inv.log() * -0.5).exp()
    return x * rms * weight


def cross_entropy(logits: Tensor, targets: np.ndarray):
    """logits: (N, V) Tensor; targets: (N,) int np. Возвращает скаляр-Tensor."""
    p = logits.softmax(axis=-1)
    N = logits.shape[0]
    # -mean(log p[target])
    onehot = np.zeros(logits.shape, dtype=np.float64)
    onehot[np.arange(N), targets] = 1.0
    logp = (p + 1e-12).log()
    return (logp * Tensor(onehot)).sum() * (-1.0 / float(N))
