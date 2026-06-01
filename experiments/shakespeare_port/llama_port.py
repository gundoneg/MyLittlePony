"""Load a Llama-style SLM (e.g. SupraLabs/Supra-50M-Instruct) and port it into an
SSM by the same zero-shot, NO-inference, NO-data closed-form weight map.

Config-driven: reads the model's own config.json, so it adapts to whatever
hidden size / layers / heads / GQA / vocab the checkpoint has.

  emb / lm_head      -> copied verbatim (gauge-free bridge)
  RMSNorms           -> copied verbatim
  SwiGLU MLP         -> copied verbatim (target uses the same SwiGLU)
  attention -> SSM   -> closed form (the only cross-architecture step):
      in_proj  <- V projection (GQA groups expanded to full width)
      out_proj <- O projection
      short conv <- identity; LTI kernel K[tau]=(1-a) a**tau (recency average);
      gate -> constant 1.  RoPE is simply dropped -- the SSM carries position
      through its causal kernel instead.
"""
import argparse, json, math, os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer

from donor import Config, RMSNorm
from target import SSMMixer, SwiGLU

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------- config --------------------------------------
@dataclass
class LlamaCfg:
    vocab_size: int
    hidden: int
    n_layer: int
    n_head: int
    n_kv_head: int
    inter: int
    rms_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie: bool = False

    @property
    def head_dim(self):
        return self.hidden // self.n_head


def load_cfg(path):
    with open(path) as f:
        c = json.load(f)
    return LlamaCfg(
        vocab_size=c["vocab_size"],
        hidden=c["hidden_size"],
        n_layer=c["num_hidden_layers"],
        n_head=c["num_attention_heads"],
        n_kv_head=c.get("num_key_value_heads", c["num_attention_heads"]),
        inter=c["intermediate_size"],
        rms_eps=c.get("rms_norm_eps", 1e-5),
        rope_theta=c.get("rope_theta", 10000.0),
        tie=c.get("tie_word_embeddings", False),
    )


# ------------------------------- RoPE ----------------------------------------
def rope_tables(T, hd, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(T, device=device).float()
    freqs = torch.outer(t, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, nkv, T, hd = x.shape
    return x[:, :, None, :, :].expand(B, nkv, n_rep, T, hd).reshape(B, nkv * n_rep, T, hd)


# ------------------------------- Llama donor ---------------------------------
class LlamaAttn(nn.Module):
    def __init__(self, cfg: LlamaCfg):
        super().__init__()
        H, hd = cfg.hidden, cfg.head_dim
        self.nh, self.nkv, self.hd = cfg.n_head, cfg.n_kv_head, hd
        self.q_proj = nn.Linear(H, cfg.n_head * hd, bias=False)
        self.k_proj = nn.Linear(H, cfg.n_kv_head * hd, bias=False)
        self.v_proj = nn.Linear(H, cfg.n_kv_head * hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * hd, H, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.nh // self.nkv)
        v = repeat_kv(v, self.nh // self.nkv)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        att = F.softmax(att + mask, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


class LlamaMLP(nn.Module):
    def __init__(self, cfg: LlamaCfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden, cfg.inter, bias=False)
        self.up_proj = nn.Linear(cfg.hidden, cfg.inter, bias=False)
        self.down_proj = nn.Linear(cfg.inter, cfg.hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaRMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class LlamaBlock(nn.Module):
    def __init__(self, cfg: LlamaCfg):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(cfg.hidden, cfg.rms_eps)
        self.self_attn = LlamaAttn(cfg)
        self.post_attention_layernorm = LlamaRMSNorm(cfg.hidden, cfg.rms_eps)
        self.mlp = LlamaMLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaLM(nn.Module):
    def __init__(self, cfg: LlamaCfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.layers = nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm = LlamaRMSNorm(cfg.hidden, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.embed_tokens(idx)
        cos, sin = rope_tables(T, self.cfg.head_dim, self.cfg.rope_theta, idx.device)
        for blk in self.layers:
            x = blk(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
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


def load_llama(cfg: LlamaCfg, st_path):
    sd = load_file(st_path)
    sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in sd.items()}
    model = LlamaLM(cfg)
    msd = model.state_dict()
    miss = []
    for k in msd:
        src = k
        if k == "lm_head.weight" and "lm_head.weight" not in sd:
            src = "embed_tokens.weight" if cfg.tie else k  # tied head
        if src in sd:
            msd[k].copy_(sd[src])
        else:
            miss.append(k)
    if miss:
        print(f"  [warn] {len(miss)} tensors not found in checkpoint, e.g. {miss[:3]}")
    model.load_state_dict(msd)
    return model


# ------------------------- SSM target + closed-form port ---------------------
class LlamaSSMPort(nn.Module):
    """SSM with SwiGLU FFN + RMSNorm matching Llama, so non-mixer parts copy 1:1.
    No positional embedding -- the SSM kernel supplies position."""

    def __init__(self, cfg: LlamaCfg):
        super().__init__()
        self.cfg = cfg
        sc = Config(vocab_size=cfg.vocab_size, ctx=512, d_model=cfg.hidden,
                    n_layer=cfg.n_layer, n_head=cfg.n_head, d_ff=cfg.inter, dropout=0.0)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.mix = nn.ModuleList([SSMMixer(sc) for _ in range(cfg.n_layer)])
        self.n1 = nn.ModuleList([RMSNorm(cfg.hidden, cfg.rms_eps) for _ in range(cfg.n_layer)])
        self.n2 = nn.ModuleList([RMSNorm(cfg.hidden, cfg.rms_eps) for _ in range(cfg.n_layer)])
        self.ffn = nn.ModuleList([SwiGLU(sc) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.hidden, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x = self.embed_tokens(idx)
        for i in range(self.cfg.n_layer):
            x = x + self.mix[i](self.n1[i](x))
            x = x + self.ffn[i](self.n2[i](x))
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
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


def _logit(a):
    a = torch.clamp(a, 1e-4, 1 - 1e-4)
    return torch.log(a / (1 - a))


@torch.no_grad()
def port_llama_to_ssm(llama: LlamaLM, cfg: LlamaCfg, a_lo=0.5, a_hi=0.95) -> LlamaSSMPort:
    port = LlamaSSMPort(cfg)
    port.embed_tokens.weight.copy_(llama.embed_tokens.weight)
    port.lm_head.weight.copy_(llama.lm_head.weight)
    port.norm.g.copy_(llama.norm.weight)
    H, hd, rep = cfg.hidden, cfg.head_dim, cfg.n_head // cfg.n_kv_head
    ch = torch.arange(H).float() / max(H - 1, 1)
    a = a_lo + (a_hi - a_lo) * ch
    for i in range(cfg.n_layer):
        lb, mix = llama.layers[i], port.mix[i]
        port.n1[i].g.copy_(lb.input_layernorm.weight)
        port.n2[i].g.copy_(lb.post_attention_layernorm.weight)
        # SwiGLU FFN verbatim (Llama has no biases -> zero ours)
        port.ffn[i].w1.weight.copy_(lb.mlp.gate_proj.weight)
        port.ffn[i].w2.weight.copy_(lb.mlp.up_proj.weight)
        port.ffn[i].proj.weight.copy_(lb.mlp.down_proj.weight)
        for lin in (port.ffn[i].w1, port.ffn[i].w2, port.ffn[i].proj):
            lin.bias.zero_()
        # attention -> SSM mixer (closed form)
        Wv = lb.self_attn.v_proj.weight.view(cfg.n_kv_head, hd, H)
        Wv = Wv[:, None, :, :].expand(cfg.n_kv_head, rep, hd, H).reshape(H, H)  # GQA -> full width
        mix.in_proj.weight.copy_(Wv)
        mix.in_proj.bias.zero_()
        mix.out_proj.weight.copy_(lb.self_attn.o_proj.weight)
        mix.out_proj.bias.zero_()
        mix.conv.weight.zero_(); mix.conv.weight[:, 0, -1] = 1.0
        if mix.conv.bias is not None:
            mix.conv.bias.zero_()
        mix.a_logit.copy_(_logit(a)); mix.b.copy_(1.0 - a); mix.c.copy_(torch.ones(H)); mix.d_skip.zero_()
        mix.gate_proj.weight.zero_(); mix.gate_proj.bias.fill_(1.27846)
    return port


# ------------------------------- eval ----------------------------------------
@torch.no_grad()
def eval_ppl(model, ids, ctx=256, n_batches=20, bs=8, seed=0):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(n_batches):
        ix = torch.randint(0, len(ids) - ctx - 1, (bs,), generator=g)
        x = torch.stack([ids[i:i + ctx] for i in ix])
        y = torch.stack([ids[i + 1:i + 1 + ctx] for i in ix])
        _, loss = model(x, y)
        tot += loss.item()
    return math.exp(tot / n_batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "models", "supra50m"),
                    help="dir with config.json, model.safetensors, tokenizer.json")
    ap.add_argument("--text", default=os.path.join(HERE, "data", "input.txt"),
                    help="plain-text file to measure perplexity on")
    ap.add_argument("--a_lo", type=float, default=0.5)
    ap.add_argument("--a_hi", type=float, default=0.95)
    ap.add_argument("--prompt", default="The meaning of life is")
    args = ap.parse_args()

    cfg = load_cfg(os.path.join(args.dir, "config.json"))
    print(f"Llama cfg: vocab {cfg.vocab_size}, hidden {cfg.hidden}, {cfg.n_layer}L, "
          f"{cfg.n_head}H/{cfg.n_kv_head}KV, inter {cfg.inter}, tie {cfg.tie}")
    tok = Tokenizer.from_file(os.path.join(args.dir, "tokenizer.json"))
    llama = load_llama(cfg, os.path.join(args.dir, "model.safetensors"))
    n = sum(p.numel() for p in llama.parameters())
    print(f"loaded donor: {n/1e6:.1f}M params")

    with open(args.text) as f:
        ids = torch.tensor(tok.encode(f.read()).ids, dtype=torch.long)
    print(f"eval stream: {len(ids)} tokens")

    port = port_llama_to_ssm(llama, cfg, args.a_lo, args.a_hi)
    rand = LlamaSSMPort(cfg)

    pd = eval_ppl(llama, ids)
    pp = eval_ppl(port, ids)
    pr = eval_ppl(rand, ids)
    print("\n=== zero-shot weight port: Llama Transformer -> SSM (0 inference, 0 data) ===")
    print(f"donor Llama   val ppl : {pd:8.1f}")
    print(f"ported SSM    val ppl : {pp:8.1f}   <-- pure weight transplant")
    print(f"random SSM    val ppl : {pr:8.1f}")

    ids0 = torch.tensor([tok.encode(args.prompt).ids], dtype=torch.long)
    print(f"\n--- prompt: {args.prompt!r} ---")
    print("[donor]", tok.decode(llama.generate(ids0, 60)[0].tolist()))
    print("[port ]", tok.decode(port.generate(ids0, 60)[0].tolist()))


if __name__ == "__main__":
    main()
