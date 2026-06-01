"""Diagnose why the linear-attention (FAVOR+) port collapsed to ~random."""
import math, os
import torch
import llama_port as LP
import linatt_port as LA
from tokenizers import Tokenizer

d = os.path.join(os.path.dirname(__file__), "models", "supra50m")
cfg = LP.load_cfg(os.path.join(d, "config.json"))
tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
llama = LP.load_llama(cfg, os.path.join(d, "model.safetensors"))
ids = torch.tensor(tok.encode(open(os.path.join(os.path.dirname(__file__), "data", "input.txt")).read()).ids[:20000], dtype=torch.long)

# (a) PLUMBING: an all-softmax "linear-attn port" must equal the donor.
allsm = LA.port_llama_to_linatt(llama, cfg, 256, softmax_layers=cfg.n_layer)
print(f"(a) all-softmax port ppl = {LA.eval_ppl(allsm, ids):.1f}   (donor is 130.8; equal => plumbing OK)")

# (b) SHARPNESS: distribution of donor attention logits q.k/sqrt(d) per layer.
x = llama.embed_tokens(ids[:256].unsqueeze(0))
cos, sin = LP.rope_tables(256, cfg.head_dim, cfg.rope_theta, x.device)
h = x
print("(b) donor attention logits q.k/sqrt(d):")
for i, blk in enumerate(llama.layers):
    hn = blk.input_layernorm(h)
    B, T, _ = hn.shape
    q = blk.self_attn.q_proj(hn).view(B, T, cfg.n_head, cfg.head_dim).transpose(1, 2)
    k = blk.self_attn.k_proj(hn).view(B, T, cfg.n_kv_head, cfg.head_dim).transpose(1, 2)
    q, k = LP.apply_rope(q, k, cos, sin)
    k = LP.repeat_kv(k, cfg.n_head // cfg.n_kv_head)
    logit = (q @ k.transpose(-2, -1)) / math.sqrt(cfg.head_dim)
    tri = torch.tril(torch.ones(T, T)).bool()
    fin = logit[:, :, tri]
    if i in (0, 3, 6, 9, 11):
        print(f"   layer {i:2d}: max {fin.max():6.1f}  p99 {fin.quantile(0.99):6.1f}  "
              f"mean {fin.mean():5.1f}   => exp(max)=10^{fin.max().item()/math.log(10):.0f}")
    h = blk(h, cos, sin)

# (c) COLLAPSE: in the linear mixer (layer 0), how degenerate is the normalizer?
mix = LA.LinAttMixer(cfg, num_features=256, seed=0, mode="linear")
with torch.no_grad():
    mix.q_proj.weight.copy_(llama.layers[0].self_attn.q_proj.weight)
    mix.k_proj.weight.copy_(llama.layers[0].self_attn.k_proj.weight)
    mix.v_proj.weight.copy_(llama.layers[0].self_attn.v_proj.weight)
    mix.o_proj.weight.copy_(llama.layers[0].self_attn.o_proj.weight)
hn = llama.layers[0].input_layernorm(x)
q, k, v = mix._qkv(hn, cos, sin)
phi_q = mix.feature_map(q * mix.scale); phi_k = mix.feature_map(k * mix.scale)
A = (phi_q @ phi_k.transpose(-2, -1)) * torch.tril(torch.ones(256, 256))
den = A.sum(-1)
# effective number of attended keys per query (1/sum p^2), vs softmax's
p = A / A.sum(-1, keepdim=True).clamp_min(1e-9)
eff = 1.0 / (p.pow(2).sum(-1) + 1e-9)
print(f"(c) linear-mixer layer0: phi_k max/min ratio = {(phi_k.max()/phi_k.clamp_min(1e-30).min()).item():.1e}")
print(f"    normalizer den: min {den.min():.2e}  median {den.median():.2e}  max {den.max():.2e}")
print(f"    effective #keys attended (mean) {eff.mean():.1f}  (1 = collapsed onto a single noisy key)")
