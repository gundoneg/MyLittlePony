"""Zero-shot, NO-inference, NO-data weight port: Transformer -> SSM.

This is the genuine "translate the weights directly" regime: we read a trained
Transformer's parameters and emit an SSM's parameters by a closed-form map.
Nothing is trained, no data is seen, the teacher is never run.

The map:
  * emb / pos / unembedding  -> copied verbatim (gauge-free bridge: shared
    vocabulary and d_model make these tensors architecture-agnostic).
  * RMSNorms and the FFN      -> copied verbatim (the SSM block uses the same
    GELU MLP as the donor, so these transfer exactly).
  * attention -> SSM mixer    -> closed form, the only real cross-architecture
    step:
       in_proj  <- W_V, b_V     (attention's value projection)
       out_proj <- W_O, b_O     (attention's output projection)
       short conv <- identity (delta)
       diagonal LTI kernel K[tau] = (1-a) * a**tau  per channel, with the
         per-channel decay a spread across [a_lo, a_hi]; this is a normalized
         causal weighted average (sums to 1) -- a *content-free* stand-in for
         attention's soft averaging over recent context.
       gate -> constant 1 (so the gated SSM reduces to out_proj(EMA(V x))).

Honest expectation: attention routes by *content*; an LTI-SSM cannot, so the
temporal mixing is replaced by a fixed recency kernel. The channel-mixing parts
of attention (V, O) and everything else transfer exactly. We measure how far
that pure transplant gets, with zero training.
"""
import argparse, math, os

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_tokenizer, load_ids, Batcher
from donor import GPT, Config, RMSNorm, MLP
from target import SSMMixer

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- target: an SSM whose FFN/norm match the donor so they copy verbatim ----
class PortBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.mix = SSMMixer(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class SSMPort(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.ctx, cfg.d_model)
        self.drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([PortBlock(cfg) for _ in range(cfg.n_layer)])
        self.nf = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
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
            idx_c = idx[:, -self.cfg.ctx:]
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ----------------------------- the closed-form map ---------------------------
def _logit(a):
    a = torch.clamp(a, 1e-4, 1 - 1e-4)
    return torch.log(a / (1 - a))


@torch.no_grad()
def port_attention_to_ssm(attn, mix: SSMMixer, d, n_head, a_lo=0.5, a_hi=0.95):
    """Closed-form: fill an SSMMixer from a CausalAttention layer. No data."""
    qkv_w, qkv_b = attn.qkv.weight, attn.qkv.bias        # (3d, d), (3d,)
    Wv = qkv_w[2 * d:3 * d, :]                            # value projection
    bv = qkv_b[2 * d:3 * d]
    Wo, bo = attn.proj.weight, attn.proj.bias            # output projection

    mix.in_proj.weight.copy_(Wv)
    mix.in_proj.bias.copy_(bv)
    mix.out_proj.weight.copy_(Wo)
    mix.out_proj.bias.copy_(bo)

    # short causal conv -> identity (delta on the most recent tap)
    mix.conv.weight.zero_()
    mix.conv.weight[:, 0, -1] = 1.0
    if mix.conv.bias is not None:
        mix.conv.bias.zero_()

    # diagonal LTI kernel = normalized recency average: K[tau] = (1-a) a**tau
    ch = torch.arange(d).float() / max(d - 1, 1)
    a = a_lo + (a_hi - a_lo) * ch                         # spread decays per channel
    mix.a_logit.copy_(_logit(a))
    mix.b.copy_(1.0 - a)                                  # makes kernel sum to 1
    mix.c.copy_(torch.ones(d))
    mix.d_skip.zero_()

    # gate -> constant ~1 so silu(gate)=1 and the mixer is out_proj(EMA(silu(Vx)))
    mix.gate_proj.weight.zero_()
    mix.gate_proj.bias.fill_(1.27846)                     # silu(1.27846) ~= 1.0


@torch.no_grad()
def port_from_donor(donor: GPT, cfg: Config, a_lo=0.5, a_hi=0.95) -> SSMPort:
    port = SSMPort(cfg)
    # gauge-free bridge: copy verbatim
    port.tok.weight.copy_(donor.tok.weight)
    port.pos.weight.copy_(donor.pos.weight)
    port.head.weight.copy_(donor.head.weight)
    port.nf.g.copy_(donor.nf.g)
    for i, (db, pb) in enumerate(zip(donor.blocks, port.blocks)):
        pb.n1.g.copy_(db.n1.g)                            # norms verbatim
        pb.n2.g.copy_(db.n2.g)
        pb.mlp.fc.weight.copy_(db.mlp.fc.weight)          # FFN verbatim
        pb.mlp.fc.bias.copy_(db.mlp.fc.bias)
        pb.mlp.proj.weight.copy_(db.mlp.proj.weight)
        pb.mlp.proj.bias.copy_(db.mlp.proj.bias)
        port_attention_to_ssm(db.attn, pb.mix, cfg.d_model, cfg.n_head, a_lo, a_hi)
    return port


# --------------------------------- evaluation --------------------------------
@torch.no_grad()
def eval_ppl(model, batcher, iters=50, bs=32, seed=0):
    model.eval()
    batcher.rng.seed(seed)
    tot = 0.0
    for _ in range(iters):
        x, y = batcher.batch("val", bs, torch)
        _, loss = model(x, y)
        tot += loss.item()
    return math.exp(tot / iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", default=os.path.join(HERE, "donor.pt"))
    ap.add_argument("--a_lo", type=float, default=0.5)
    ap.add_argument("--a_hi", type=float, default=0.95)
    ap.add_argument("--out", default=os.path.join(HERE, "port_zeroshot.pt"))
    args = ap.parse_args()

    bpe = get_tokenizer()
    ids = load_ids(bpe)

    ckpt = torch.load(args.donor, weights_only=False)
    cfg = Config(**ckpt["cfg"])
    batcher = Batcher(ids, cfg.ctx)

    donor = GPT(cfg)
    donor.load_state_dict(ckpt["model"])
    donor.eval()

    port = port_from_donor(donor, cfg, args.a_lo, args.a_hi)
    rand = SSMPort(cfg)  # untrained SSM baseline (same arch, random interior+frozen-ish)

    ppl_donor = eval_ppl(donor, batcher)
    ppl_port = eval_ppl(port, batcher)
    ppl_rand = eval_ppl(rand, batcher)

    print("=== zero-shot weight port: Transformer -> SSM (0 inference, 0 data) ===")
    print(f"donor Transformer val ppl : {ppl_donor:7.1f}")
    print(f"ported SSM     val ppl    : {ppl_port:7.1f}   <-- pure weight transplant")
    print(f"random SSM     val ppl    : {ppl_rand:7.1f}   (= vocab size, no signal)")
    print(f"vocab size                : {cfg.vocab_size}")

    torch.save({"model": port.state_dict(), "cfg": cfg.__dict__, "val_ppl": ppl_port}, args.out)
    prompt = torch.tensor([bpe.encode("ROMEO:\n")], dtype=torch.long)
    out = port.generate(prompt, 80, temperature=0.8, top_k=40)
    print("\n--- ported-SSM sample (no training) ---")
    print(bpe.decode(out[0].tolist()))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
