"""Phase 9 finale+ — does a LIGHT per-slot adapter close the depth-transfer gap?

The verbatim-graft finale showed: copying one deep exemplar into all interior slots (3-10)
of supra50m costs joint KL ~6.5 (errors compound). Prediction from the single-graft costs
(<0.5): the per-slot correction is SMALL, so a light learned adapter on top of ONE shared
exemplar should recover the full model. Test it directly.

Reconstruction: slots {0,1,2,11} = own (frozen) weights; interior slots 3-10 all share ONE
exemplar layer's frozen weights PLUS a trainable per-slot adapter = rank-r LoRA delta on
each of the 7 projections + a per-slot RMSNorm gain. Train (Adam) to distill the full
model's logits (KL). delta init 0 => step 0 == verbatim tile (~6.5). Report train+held-out
KL vs verbatim (6.5) and exact (0).
"""
import os, torch, torch.nn.functional as F
from llama_min import MinLlama, load_vocab, decode

PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
EXEMPLAR = 6
INTERIOR = list(range(3, 11))
RANK = int(os.environ.get("RANK", 4))
NSEQ = int(os.environ.get("NSEQ", 4))
STEPS = int(os.environ.get("STEPS", 250))
WD = float(os.environ.get("WD", 0.0))


class Recon:
    """Differentiable supra50m reconstruction with shared-exemplar + per-slot adapters."""
    def __init__(self, m):
        self.m = m
        self.inv = 1.0 / (m.theta ** (torch.arange(0, m.hd, 2).float() / m.hd))
        # trainable adapters, one set per interior slot, init delta=0 (=> verbatim tile)
        self.A, self.B, self.gin, self.gpost = {}, {}, {}, {}
        self.params = []
        for i in INTERIOR:
            self.A[i], self.B[i] = {}, {}
            for n in PROJ:
                W = m.w[f"model.layers.{EXEMPLAR}.{n}.weight"]
                a = (torch.randn(RANK, W.shape[1]) * 0.02).requires_grad_()
                b = torch.zeros(W.shape[0], RANK).requires_grad_()
                self.A[i][n], self.B[i][n] = a, b
                self.params += [a, b]
            self.gin[i] = m.w[f"model.layers.{EXEMPLAR}.input_layernorm.weight"].clone().requires_grad_()
            self.gpost[i] = m.w[f"model.layers.{EXEMPLAR}.post_attention_layernorm.weight"].clone().requires_grad_()
            self.params += [self.gin[i], self.gpost[i]]

    def _w(self, slot, n):
        """Weight for projection n at this slot: own (exact) or exemplar+LoRA (interior)."""
        if slot in INTERIOR:
            W = self.m.w[f"model.layers.{EXEMPLAR}.{n}.weight"]      # frozen exemplar
            return W + self.B[slot][n] @ self.A[slot][n]            # + low-rank delta
        return self.m.w[f"model.layers.{slot}.{n}.weight"]          # frozen own

    def _rms(self, x, g):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.m.eps) * g

    def _rope(self, x, pos):
        fr = pos[:, None].float() * self.inv[None, :]
        from llama_min import rotate_half
        cos = torch.cat([fr.cos(), fr.cos()], -1)[None, None]
        sin = torch.cat([fr.sin(), fr.sin()], -1)[None, None]
        return x * cos + rotate_half(x) * sin

    def _block(self, x, slot, pos):
        m = self.m
        gin = self.gin[slot] if slot in INTERIOR else m.w[f"model.layers.{slot}.input_layernorm.weight"]
        gpost = self.gpost[slot] if slot in INTERIOR else m.w[f"model.layers.{slot}.post_attention_layernorm.weight"]
        B, T, _ = x.shape
        h = self._rms(x, gin)
        q = (h @ self._w(slot, "self_attn.q_proj").T).view(B, T, m.H, m.hd).transpose(1, 2)
        k = (h @ self._w(slot, "self_attn.k_proj").T).view(B, T, m.KV, m.hd).transpose(1, 2)
        v = (h @ self._w(slot, "self_attn.v_proj").T).view(B, T, m.KV, m.hd).transpose(1, 2)
        q, k = self._rope(q, pos), self._rope(k, pos)
        rep = m.H // m.KV
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        att = (q @ k.transpose(-1, -2)) / (m.hd ** 0.5) + torch.full((T, T), float("-inf")).triu(1)
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, m.d)
        x = x + o @ self._w(slot, "self_attn.o_proj").T
        h2 = self._rms(x, gpost)
        mlp = (F.silu(h2 @ self._w(slot, "mlp.gate_proj").T) * (h2 @ self._w(slot, "mlp.up_proj").T)) @ self._w(slot, "mlp.down_proj").T
        return x + mlp

    def logits(self, idx):
        pos = torch.arange(idx.shape[1])
        x = self.m.embed[idx]
        for slot in range(self.m.L):
            x = self._block(x, slot, pos)
        return self._rms(x, self.m.w["model.norm.weight"]) @ self.m.embed.T


def klmean(target_logits, pred_logits):
    pt = target_logits.log_softmax(-1)
    return (pt.exp() * (pt - pred_logits.log_softmax(-1))).sum(-1).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    m = MinLlama()
    inv = load_vocab()
    gen = lambda s, n=128: torch.cat([m.generate(torch.tensor([[1]]), n, temperature=0.9)[:, 1:] for _ in range(s)], 0)
    train = gen(NSEQ); torch.manual_seed(2); heldout = gen(3)
    tgt_tr = m.forward(train).detach()
    tgt_ho = m.forward(heldout).detach()
    print(f"adapter test: exemplar=L{EXEMPLAR} -> interior {INTERIOR}, rank={RANK}, "
          f"nseq={NSEQ} wd={WD}, {sum(p.numel() for p in Recon(m).params)} trainable params\n")

    r = Recon(m)
    opt = torch.optim.Adam(r.params, lr=3e-3, weight_decay=WD)
    with torch.no_grad():
        k0 = klmean(tgt_tr, r.logits(train)).item()
    print(f"  step   0  (verbatim tile)   train-KL {k0:6.3f}   [exact=0.0]")
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        loss = klmean(tgt_tr, r.logits(train))
        loss.backward(); opt.step()
        if step % 25 == 0:
            with torch.no_grad():
                kho = klmean(tgt_ho, r.logits(heldout)).item()
            print(f"  step {step:>3}   train-KL {loss.item():6.3f}   held-out-KL {kho:6.3f}")

    with torch.no_grad():
        seq = torch.tensor([[1]])
        for _ in range(50):
            lo = r.logits(seq[:, -256:])[:, -1, :] / 0.7
            v, _ = torch.topk(lo, 40); lo[lo < v[:, [-1]]] = float("-inf")
            seq = torch.cat([seq, torch.multinomial(lo.softmax(-1), 1)], 1)
    print(f"\n  adapted reconstruction sample:\n   ", repr(decode(seq[0], inv)[:240]))
