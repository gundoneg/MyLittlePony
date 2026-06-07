"""Two genuinely different tiny architectures for the translator sandbox.

A = TinyTransformer  : causal multi-head SELF-ATTENTION mixer.
B = TinySSM          : diagonal linear-recurrence + short causal depthwise conv
                       mixer (Mamba-lite). NO Q/K/V/O, no attention at all.

Both share the *peripheral* interface (token+pos embedding, RMSNorm, a GELU MLP,
unembed) on purpose -- that is standard and not the point. The architectural
divergence we care about lives entirely in the SEQUENCE MIXER.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    vocab: int = 65
    d_model: int = 96
    n_layer: int = 2
    n_head: int = 3
    ctx: int = 64
    d_ff: int = 256
    conv_w: int = 4          # B's short causal depthwise conv width
    bidirectional: bool = False   # if True, Attention drops the causal mask
                                  # (diffusion-port adaptation; default keeps AR behavior)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class MLP(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.fc = nn.Linear(c.d_model, c.d_ff)
        self.proj = nn.Linear(c.d_ff, c.d_model)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


# ----------------------------- A: Transformer -----------------------------
class Attention(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.nh, self.hd = c.n_head, c.d_model // c.n_head
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.o = nn.Linear(c.d_model, c.d_model)
        # Instance flag (not a forward-arg) so existing positional callers of
        # `mix(x)` stay untouched. Default causal == original AR numerics.
        self.causal = not c.bidirectional

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        if self.causal:
            mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
            att = att + mask
        att = att.softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(y)


# ----------------------------- B: SSM mixer -----------------------------
class SSMMixer(nn.Module):
    """Diagonal linear recurrence h_t = a*h_{t-1} + b*x_t, y = c*h, plus a short
    causal depthwise conv and a SiLU gate. Per-channel, no token-token attention."""
    def __init__(self, c: Cfg):
        super().__init__()
        d = c.d_model
        self.in_proj = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.conv = nn.Conv1d(d, d, c.conv_w, groups=d, padding=c.conv_w - 1)
        self.a_logit = nn.Parameter(torch.zeros(d))   # decay a = sigmoid(a_logit)
        self.b = nn.Parameter(torch.ones(d))
        self.cc = nn.Parameter(torch.ones(d))
        self.conv_w = c.conv_w

    def forward(self, x):
        B, T, D = x.shape
        u = self.in_proj(x)
        u = self.conv(u.transpose(1, 2))[:, :, :T].transpose(1, 2)   # causal depthwise conv
        a = torch.sigmoid(self.a_logit)
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = a * h + self.b * u[:, t]
            ys.append(self.cc * h)
        y = torch.stack(ys, dim=1)
        y = y * F.silu(self.gate(x))
        return self.out_proj(y)


class Block(nn.Module):
    def __init__(self, c: Cfg, mixer_cls):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.mix = mixer_cls(c)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class LM(nn.Module):
    """Shared skeleton; `kind` picks the mixer -> defines the architecture."""
    def __init__(self, c: Cfg, kind: str):
        super().__init__()
        assert kind in ("transformer", "ssm")
        self.cfg, self.kind = c, kind
        mixer = Attention if kind == "transformer" else SSMMixer
        self.tok = nn.Embedding(c.vocab, c.d_model)
        self.pos = nn.Embedding(c.ctx, c.d_model)
        self.blocks = nn.ModuleList([Block(c, mixer) for _ in range(c.n_layer)])
        self.norm = RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8):
        for _ in range(n):
            logits, _ = self(idx[:, -self.cfg.ctx:])
            probs = (logits[:, -1] / temperature).softmax(-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def n_params(m):
    return sum(p.numel() for p in m.parameters())
