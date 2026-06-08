"""Minimal CPU Llama forward for the local supra50m checkpoint (12-layer, hidden 512,
GQA 8q/4kv, head_dim 64, SwiGLU, RoPE, tied embeddings) -- loaded straight from
safetensors, NO transformers dependency. Supports per-layer interventions (skip / swap)
and hidden-state capture so we can measure depth-entanglement (phase 9).
"""
import json, os
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "shakespeare_port", "models", "supra50m")


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


class MinLlama:
    def __init__(self, path=DEFAULT):
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.H = cfg["num_attention_heads"]
        self.KV = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.d = cfg["hidden_size"]
        self.L = cfg["num_hidden_layers"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_parameters"]["rope_theta"]
        self.vocab = cfg["vocab_size"]
        sd = load_file(os.path.join(path, "model.safetensors"))
        self.w = {k: v.float() for k, v in sd.items()}        # bf16 -> f32 for CPU
        self.embed = self.w["model.embed_tokens.weight"]
        self.path = path

    def _rms(self, x, w):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * w

    def _rope(self, x, pos):                                  # x: (B,h,T,hd)
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2).float() / self.hd))
        fr = pos[:, None].float() * inv[None, :]              # (T, hd/2)
        cos = torch.cat([fr.cos(), fr.cos()], -1)[None, None]
        sin = torch.cat([fr.sin(), fr.sin()], -1)[None, None]
        return x * cos + rotate_half(x) * sin

    def _attn(self, x, l, pos):
        B, T, _ = x.shape
        g = lambda n: self.w[f"model.layers.{l}.self_attn.{n}.weight"]
        q = (x @ g("q_proj").T).view(B, T, self.H, self.hd).transpose(1, 2)
        k = (x @ g("k_proj").T).view(B, T, self.KV, self.hd).transpose(1, 2)
        v = (x @ g("v_proj").T).view(B, T, self.KV, self.hd).transpose(1, 2)
        q, k = self._rope(q, pos), self._rope(k, pos)
        rep = self.H // self.KV
        k = k.repeat_interleave(rep, dim=1)                   # GQA expand
        v = v.repeat_interleave(rep, dim=1)
        att = (q @ k.transpose(-1, -2)) / (self.hd ** 0.5)
        mask = torch.full((T, T), float("-inf")).triu(1)
        att = (att + mask).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, self.d)
        return o @ g("o_proj").T

    def _mlp(self, x, l):
        g = lambda n: self.w[f"model.layers.{l}.mlp.{n}.weight"]
        return (F.silu(x @ g("gate_proj").T) * (x @ g("up_proj").T)) @ g("down_proj").T

    def _block(self, x, l, pos):
        x = x + self._attn(self._rms(x, self.w[f"model.layers.{l}.input_layernorm.weight"]), l, pos)
        x = x + self._mlp(self._rms(x, self.w[f"model.layers.{l}.post_attention_layernorm.weight"]), l)
        return x

    @torch.no_grad()
    def forward(self, idx, skip=None, swap=None, capture=False, remap=None):
        """skip: stack slot to bypass (identity). swap: (a,b) order swap.
        remap: dict {slot -> source layer} -- run slot using another layer's weights
        (graft, for the type-coverage experiment). capture=True also returns residuals."""
        B, T = idx.shape
        pos = torch.arange(T)
        x = self.embed[idx]
        order = list(range(self.L))
        if swap is not None:
            a, b = swap; order[a], order[b] = order[b], order[a]
        hs = [x]
        for slot in range(self.L):
            if skip is not None and slot == skip:
                hs.append(x); continue
            src = remap.get(slot, order[slot]) if remap is not None else order[slot]
            x = self._block(x, src, pos)
            hs.append(x)
        x = self._rms(x, self.w["model.norm.weight"])
        logits = x @ self.embed.T                             # tied head
        return (logits, hs) if capture else logits

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8, top_k=40):
        for _ in range(n):
            lo = self.forward(idx[:, -256:])[:, -1, :] / temperature
            v, _ = torch.topk(lo, top_k)
            lo[lo < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(lo.softmax(-1), 1)
            idx = torch.cat([idx, nxt], 1)
        return idx


def load_vocab(path=DEFAULT):
    """id -> token string, from tokenizer.json (decoding only; merges not needed)."""
    tj = json.load(open(os.path.join(path, "tokenizer.json")))
    vocab = tj["model"]["vocab"]                              # token -> id
    inv = {i: t for t, i in vocab.items()}
    return inv


def decode(ids, inv):
    return "".join(inv.get(int(i), "?") for i in ids).replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\n")


if __name__ == "__main__":
    m = MinLlama()
    inv = load_vocab()
    print(f"loaded supra50m: L={m.L} d={m.d} H={m.H}/{m.KV} hd={m.hd} vocab={m.vocab}")
    # sanity: greedy-ish generation should be coherent English if the forward is correct
    bos = torch.tensor([[1]])                                 # pad/bos
    out = m.generate(bos, 60, temperature=0.7)
    print("SAMPLE:", repr(decode(out[0], inv)))
    # confidence sanity: entropy of next-token on self-generated seq should be low
    lo = m.forward(out)[0]
    ent = -(lo.softmax(-1) * lo.log_softmax(-1)).sum(-1).mean().item()
    print(f"mean next-token entropy on self-gen seq: {ent:.2f} nats (uniform={torch.log(torch.tensor(float(m.vocab))):.2f})")
