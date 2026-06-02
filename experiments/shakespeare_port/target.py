"""Target: a small SSM (Mamba-lite, LTI) causal decoder (torch).

Mixer per block = short causal depthwise conv (the transfer-friendly conv from
iteration 12) followed by a *diagonal LTI state-space model*. Because the SSM
is linear-time-invariant (a, b, c are static, not input-dependent), its action
is a causal convolution with a per-channel impulse kernel

    K[0] = c*b + d,   K[tau] = c * a**tau * b   (tau >= 1)

so a whole sequence is processed with one F.conv1d -- no Python time loop, fast
on CPU. Shares emb/pos/head layout with the GPT donor for direct transfer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from donor import Config, RMSNorm


class SSMMixer(nn.Module):
    """Short causal conv + diagonal LTI SSM + gating, all depthwise per channel."""

    def __init__(self, cfg: Config, conv_width=4):
        super().__init__()
        d = cfg.d_model
        self.d = d
        self.ctx = cfg.ctx
        self.conv_width = conv_width
        self.in_proj = nn.Linear(d, d)
        self.gate_proj = nn.Linear(d, d)
        # short depthwise causal conv (local mixing)
        self.conv = nn.Conv1d(d, d, conv_width, groups=d, padding=conv_width - 1)
        # diagonal LTI SSM params, per channel. Parameterize a in (0,1) via sigmoid
        # of a_logit so the system stays stable.
        self.a_logit = nn.Parameter(torch.zeros(d))   # a = sigmoid(a_logit)
        self.b = nn.Parameter(torch.ones(d) * 0.5)
        self.c = nn.Parameter(torch.ones(d) * 0.5)
        self.d_skip = nn.Parameter(torch.zeros(d))
        self.out_proj = nn.Linear(d, d)
        self.in_silu = True   # zero-shot port sets this False (linear value path, like attention's V)

    def ssm_kernel(self, L):
        # K[tau] for tau = 0..L-1, shape (d, L)
        a = torch.sigmoid(self.a_logit)                 # (d,)
        tau = torch.arange(L, device=a.device).float()  # (L,)
        a_pow = a.unsqueeze(1) ** tau.unsqueeze(0)       # (d, L) = a**tau
        K = self.c.unsqueeze(1) * a_pow * self.b.unsqueeze(1)  # c*a**tau*b
        K[:, 0] = self.c * self.b + self.d_skip          # tau=0 includes skip term
        return K

    def causal_ssm(self, x):
        # x: (B, d, T) -> y: (B, d, T) via depthwise causal conv with SSM kernel
        B, d, T = x.shape
        K = self.ssm_kernel(T).flip(1)                   # flip for conv (causal)
        xp = F.pad(x, (T - 1, 0))                         # left-pad for causality
        y = F.conv1d(xp, K.unsqueeze(1), groups=d)       # depthwise
        return y[:, :, :T]

    def forward(self, x):
        # x: (B, T, d)
        B, T, d = x.shape
        u = self.in_proj(x).transpose(1, 2)              # (B, d, T)
        u = self.conv(u)[:, :, :T]                       # short causal conv
        if self.in_silu:
            u = F.silu(u)
        y = self.causal_ssm(u)                           # diagonal LTI SSM
        y = y.transpose(1, 2)                            # (B, T, d)
        y = y * F.silu(self.gate_proj(x))                # gating
        return self.out_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.w2 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.silu(self.w1(x)) * self.w2(x)))


class SSMBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.mix = SSMMixer(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


class SSMLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)   # gauge-free: emb
        self.pos = nn.Embedding(cfg.ctx, cfg.d_model)          # gauge-free: pos
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([SSMBlock(cfg) for _ in range(cfg.n_layer)])
        self.nf = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # gauge-free: unemb
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.nf(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new):
            idx_c = idx[:, -self.cfg.ctx :]
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
