"""Zero-shot, NO-inference, NO-data port: Llama attention -> LINEAR ATTENTION
(= a selective linear SSM), to recover the coherence the LTI-SSM port loses.

Why this is the fundamental fix
-------------------------------
Softmax attention routes by CONTENT: A(x)=softmax(QK^T/sqrt(d)). The LTI-SSM port
(llama_port.py) throws Q,K away and replaces A(x) with a fixed recency kernel, so
it cannot represent content routing -- a function-CLASS mismatch no weight map can
fix. Linear/kernel attention keeps the routing and has an exact recurrent form

    S_t = S_{t-1} + phi(k_t) v_t^T          (state, input-dependent update)
    z_t = z_{t-1} + phi(k_t)
    y_t = (phi(q_t) . S_t) / (phi(q_t) . z_t + eps)

i.e. a linear time-varying SSM with O(1)-memory inference. It SHARES Q,K,V,O with
the donor, so the zero-shot map copies them verbatim, keeps RoPE, and the ONLY
change is softmax -> phi. We use Performer FAVOR+ positive random features
phi(x)=exp(w.x - ||x||^2/2)/sqrt(m), the unique data-free choice with
E[phi(q).phi(k)] = exp(q.k): more features -> closer to softmax. Scaling d^-1/4 on
q,k folds in softmax's 1/sqrt(d).
"""
import argparse, math, os

import torch
import torch.nn as nn
import torch.nn.functional as F

from donor import Config, RMSNorm
from target import SwiGLU
from llama_port import (
    LlamaCfg, load_cfg, load_llama, LlamaLM,
    rope_tables, apply_rope, repeat_kv, eval_ppl,
    LlamaSSMPort, port_llama_to_ssm,
)

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------- FAVOR+ orthogonal random features -------------------
def orthogonal_features(nh, m, hd, seed):
    """omega: (nh, m, hd) with per-head orthonormal rows scaled by chi-distributed
    norms (Gaussian orthogonal random features, the FAVOR+ variance-reduced draw)."""
    gen = torch.Generator().manual_seed(seed)
    out = torch.empty(nh, m, hd)
    for h in range(nh):
        blocks = []
        rem = m
        while rem > 0:
            G = torch.randn(hd, hd, generator=gen)
            Q, _ = torch.linalg.qr(G)          # (hd,hd) orthonormal columns
            blocks.append(Q.t())               # -> orthonormal rows
            rem -= hd
        M = torch.cat(blocks, 0)[:m]           # (m, hd)
        norms = torch.randn(m, hd, generator=gen).norm(dim=1, keepdim=True)  # chi
        out[h] = M * norms
    return out


# ------------------------------- the mixer -----------------------------------
class LinAttMixer(nn.Module):
    """Causal linear attention (FAVOR+). Holds the donor's q/k/v/o projections;
    mode='softmax' falls back to exact attention (hybrid escape hatch)."""

    def __init__(self, cfg: LlamaCfg, num_features=256, seed=0, mode="linear", eps=1e-6):
        super().__init__()
        H, hd = cfg.hidden, cfg.head_dim
        self.nh, self.nkv, self.hd = cfg.n_head, cfg.n_kv_head, hd
        self.m, self.eps, self.mode = num_features, eps, mode
        self.scale = hd ** -0.25
        self.q_proj = nn.Linear(H, cfg.n_head * hd, bias=False)
        self.k_proj = nn.Linear(H, cfg.n_kv_head * hd, bias=False)
        self.v_proj = nn.Linear(H, cfg.n_kv_head * hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * hd, H, bias=False)
        self.register_buffer("omega", orthogonal_features(cfg.n_head, num_features, hd, seed))

    def feature_map(self, x):
        # x: (B, nh, T, hd), already RoPE'd and * scale
        proj = torch.einsum("bhtd,hmd->bhtm", x, self.omega)     # (B,nh,T,m)
        sqn = 0.5 * (x * x).sum(-1, keepdim=True)                # (B,nh,T,1) Performer term (per position)
        stab = proj.amax(dim=(-2, -1), keepdim=True)             # per-(b,h) const over seq: cancels in ratio
        return torch.exp(proj - sqn - stab) / math.sqrt(self.m)  # (B,nh,T,m), >0

    def _qkv(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.nh // self.nkv)
        v = repeat_kv(v, self.nh // self.nkv)
        return q, k, v

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self._qkv(x, cos, sin)
        if self.mode == "softmax":                               # exact donor attention
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            att = att + torch.full((T, T), float("-inf"), device=x.device).triu(1)
            y = F.softmax(att, dim=-1) @ v
        else:                                                    # dense causal linear attention
            phi_q = self.feature_map(q * self.scale)
            phi_k = self.feature_map(k * self.scale)
            A = phi_q @ phi_k.transpose(-2, -1)                  # (B,nh,T,T)
            A = A * torch.tril(torch.ones(T, T, device=x.device))
            num = A @ v                                          # (B,nh,T,hd)
            den = A.sum(-1, keepdim=True).clamp_min(self.eps)
            y = num / den
        y = y.transpose(1, 2).reshape(B, T, self.nh * self.hd)
        return self.o_proj(y)

    @torch.no_grad()
    def recurrent_forward(self, x, cos, sin):
        """O(1)-memory recurrent form -- the literal SSM. For the equivalence test
        and generation; must match forward() (mode='linear')."""
        B, T, _ = x.shape
        q, k, v = self._qkv(x, cos, sin)
        phi_q = self.feature_map(q * self.scale)                 # (B,nh,T,m)
        phi_k = self.feature_map(k * self.scale)
        S = torch.zeros(B, self.nh, self.m, self.hd)             # state
        z = torch.zeros(B, self.nh, self.m)
        ys = []
        for t in range(T):
            S = S + phi_k[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
            z = z + phi_k[:, :, t]
            num = torch.einsum("bhm,bhmd->bhd", phi_q[:, :, t], S)
            den = torch.einsum("bhm,bhm->bh", phi_q[:, :, t], z).clamp_min(self.eps)
            ys.append(num / den.unsqueeze(-1))
        y = torch.stack(ys, 2).transpose(1, 2).reshape(B, T, self.nh * self.hd)
        return self.o_proj(y)


# ------------------------------- the target model ----------------------------
class LlamaLinAttPort(nn.Module):
    def __init__(self, cfg: LlamaCfg, num_features=256, seed=0, softmax_layers=0):
        super().__init__()
        self.cfg = cfg
        sc = Config(vocab_size=cfg.vocab_size, ctx=512, d_model=cfg.hidden,
                    n_layer=cfg.n_layer, n_head=cfg.n_head, d_ff=cfg.inter, dropout=0.0)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.mix = nn.ModuleList([
            LinAttMixer(cfg, num_features, seed + i,
                        mode="softmax" if i < softmax_layers else "linear")
            for i in range(cfg.n_layer)])
        self.n1 = nn.ModuleList([RMSNorm(cfg.hidden, cfg.rms_eps) for _ in range(cfg.n_layer)])
        self.n2 = nn.ModuleList([RMSNorm(cfg.hidden, cfg.rms_eps) for _ in range(cfg.n_layer)])
        self.ffn = nn.ModuleList([SwiGLU(sc) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.hidden, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embed_tokens(idx)
        cos, sin = rope_tables(T, self.cfg.head_dim, self.cfg.rope_theta, idx.device)
        for i in range(self.cfg.n_layer):
            x = x + self.mix[i](self.n1[i](x), cos, sin)
            x = x + self.ffn[i](self.n2[i](x))
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None if targets is None else F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8, top_k=40, ctx=512):
        self.eval()
        for _ in range(max_new):
            logits, _ = self(idx[:, -ctx:])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx


# ------------------------------- the closed-form port ------------------------
@torch.no_grad()
def port_llama_to_linatt(llama: LlamaLM, cfg: LlamaCfg, num_features=256, seed=0,
                         softmax_layers=0) -> LlamaLinAttPort:
    port = LlamaLinAttPort(cfg, num_features, seed, softmax_layers)
    # gauge-free verbatim copies (identical to the LTI port)
    port.embed_tokens.weight.copy_(llama.embed_tokens.weight)
    port.lm_head.weight.copy_(llama.lm_head.weight)
    port.norm.g.copy_(llama.norm.weight)
    for i in range(cfg.n_layer):
        lb, mix = llama.layers[i], port.mix[i]
        port.n1[i].g.copy_(lb.input_layernorm.weight)
        port.n2[i].g.copy_(lb.post_attention_layernorm.weight)
        port.ffn[i].w1.weight.copy_(lb.mlp.gate_proj.weight)
        port.ffn[i].w2.weight.copy_(lb.mlp.up_proj.weight)
        port.ffn[i].proj.weight.copy_(lb.mlp.down_proj.weight)
        for lin in (port.ffn[i].w1, port.ffn[i].w2, port.ffn[i].proj):
            lin.bias.zero_()
        # the ONLY cross-architecture step: copy Q,K,V,O verbatim (softmax -> phi)
        mix.q_proj.weight.copy_(lb.self_attn.q_proj.weight)
        mix.k_proj.weight.copy_(lb.self_attn.k_proj.weight)
        mix.v_proj.weight.copy_(lb.self_attn.v_proj.weight)
        mix.o_proj.weight.copy_(lb.self_attn.o_proj.weight)
    return port


# ------------------------------- self-checks ---------------------------------
def selfcheck(cfg):
    print("=== selfcheck (no data) ===")
    torch.manual_seed(0)
    small = LlamaCfg(vocab_size=64, hidden=64, n_layer=1, n_head=4, n_kv_head=2,
                     inter=128, rms_eps=1e-6, rope_theta=10000.0, tie=True)
    mix = LinAttMixer(small, num_features=128, seed=1)
    x = torch.randn(2, 16, small.hidden)
    cos, sin = rope_tables(16, small.head_dim, small.rope_theta, x.device)
    yd = mix(x, cos, sin)
    yr = mix.recurrent_forward(x, cos, sin)
    d = (yd - yr).abs().max().item()
    print(f"1. dense vs recurrent max|.|: {d:.2e}  -> {'OK' if d < 1e-4 else 'FAIL'}")

    # FAVOR+ -> softmax limit: linear output approaches softmax as m grows
    errs = []
    for m in [64, 256, 1024, 4096]:
        ml = LinAttMixer(small, num_features=m, seed=2, mode="linear")
        ms = LinAttMixer(small, num_features=m, seed=2, mode="softmax")
        with torch.no_grad():
            for a, b in [(ml.q_proj, ms.q_proj), (ml.k_proj, ms.k_proj),
                         (ml.v_proj, ms.v_proj), (ml.o_proj, ms.o_proj)]:
                b.weight.copy_(a.weight)
        e = (ml(x, cos, sin) - ms(x, cos, sin)).abs().mean().item()
        errs.append((m, e))
    print("2. linear->softmax mean|err| vs m:", "  ".join(f"m={m}:{e:.3f}" for m, e in errs),
          "->", "OK (decreasing)" if errs[-1][1] < errs[0][1] else "FAIL")

    # RoPE sanity: swapping non-adjacent positions changes the output
    x2 = x.clone(); x2[:, [3, 11]] = x2[:, [11, 3]]
    changed = (mix(x, cos, sin) - mix(x2, cos, sin)).abs().max().item()
    print(f"3. position sensitivity (swap 3<->11): {changed:.3e} -> {'OK' if changed > 1e-3 else 'FAIL'}")


# ------------------------------- main ----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "models", "supra50m"))
    ap.add_argument("--text", default=os.path.join(HERE, "data", "input.txt"))
    ap.add_argument("--num_features", type=int, default=256)
    ap.add_argument("--softmax_layers", type=int, default=0, help="keep exact softmax in first N layers (hybrid)")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--prompt", default="The meaning of life is")
    args = ap.parse_args()

    cfg = load_cfg(os.path.join(args.dir, "config.json"))
    if args.selfcheck:
        selfcheck(cfg)
        return

    from tokenizers import Tokenizer
    print(f"Llama cfg: vocab {cfg.vocab_size}, hidden {cfg.hidden}, {cfg.n_layer}L, "
          f"{cfg.n_head}H/{cfg.n_kv_head}KV, head_dim {cfg.head_dim}")
    tok = Tokenizer.from_file(os.path.join(args.dir, "tokenizer.json"))
    llama = load_llama(cfg, os.path.join(args.dir, "model.safetensors"))
    with open(args.text) as f:
        ids = torch.tensor(tok.encode(f.read()).ids, dtype=torch.long)
    print(f"eval stream: {len(ids)} tokens; FAVOR+ features m={args.num_features}, "
          f"softmax_layers={args.softmax_layers}")

    lti = port_llama_to_ssm(llama, cfg)
    lin = port_llama_to_linatt(llama, cfg, args.num_features, softmax_layers=args.softmax_layers)
    rnd = LlamaLinAttPort(cfg, args.num_features)

    print("\n=== zero-shot weight port: Llama -> linear-attention SSM (0 inference, 0 data) ===")
    print(f"donor Llama        val ppl : {eval_ppl(llama, ids):8.1f}")
    print(f"LTI-SSM port       val ppl : {eval_ppl(lti, ids):8.1f}   (content-independent, old)")
    print(f"linear-attn port   val ppl : {eval_ppl(lin, ids):8.1f}   <-- content-dependent (new)")
    print(f"random             val ppl : {eval_ppl(rnd, ids):8.1f}")

    ids0 = torch.tensor([tok.encode(args.prompt).ids], dtype=torch.long)
    print(f"\n--- prompt: {args.prompt!r} ---")
    torch.manual_seed(0); print("[donor   ]", tok.decode(llama.generate(ids0, 60)[0].tolist()))
    torch.manual_seed(0); print("[LTI-SSM ]", tok.decode(lti.generate(ids0, 60)[0].tolist()))
    torch.manual_seed(0); print("[lin-attn]", tok.decode(lin.generate(ids0, 60)[0].tolist()))


if __name__ == "__main__":
    main()
