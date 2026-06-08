"""Phase 9 (Kaggle) — run the SAME depth-entanglement metrics on a real production LLM
(Qwen3.5-0.8B) via transformers, to confirm the supra50m finding at scale: a few distinct
early layers + a long redundant, interchangeable interior.

Run on Kaggle (GPU, transformers installed). NOT runnable in the CPU dev container
(no transformers / no HF). Mirrors experiments/depth_entangle/entangle.py.
"""
import copy
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"   # swap for the exact Qwen3.5-0.8B id on Kaggle


def kl(p, q):
    lp, lq = p.log_softmax(-1), q.log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1).mean().item()


def get_layers(model):
    return model.model.layers          # Qwen/Llama decoder layer ModuleList


class Identity(torch.nn.Module):
    """Residual passthrough matching a decoder layer's (hidden, *rest) signature."""
    def forward(self, hidden_states, *a, **k):
        return (hidden_states,)


@torch.no_grad()
def logits_with(model, ids, skip=None, swap=None):
    layers = get_layers(model)
    orig = list(layers)
    new = list(orig)
    if skip is not None:
        new[skip] = Identity().to(next(model.parameters()).device)
    if swap is not None:
        a, b = swap; new[a], new[b] = new[b], new[a]
    layers._modules = {str(i): m for i, m in enumerate(new)}
    out = model(ids).logits
    layers._modules = {str(i): m for i, m in enumerate(orig)}
    return out


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    L = len(get_layers(model))
    dev = next(model.parameters()).device

    # on-distribution probe: self-generate from BOS
    bos = torch.tensor([[tok.bos_token_id or 0]], device=dev)
    ids = model.generate(bos, max_new_tokens=160, do_sample=True, temperature=0.9, top_k=40)[:, 1:]

    full = model(ids, output_hidden_states=True)
    hs, flog = full.hidden_states, full.logits           # hs: L+1 residual streams
    print(f"{MODEL}  L={L}\n{'layer':>5} | {'update':>7} | {'cos':>5} | {'drop-KL':>8} | {'swap-KL':>8}")
    print("-" * 50)
    dropk, swapk = [], []
    for l in range(L):
        a, b = hs[l], hs[l + 1]
        upd = ((b - a).norm(dim=-1) / (a.norm(dim=-1) + 1e-6)).mean().item()
        cos = F.cosine_similarity(a, b, dim=-1).mean().item()
        dk = kl(flog, logits_with(model, ids, skip=l)); dropk.append(dk)
        sk = kl(flog, logits_with(model, ids, swap=(l, l + 1))) if l < L - 1 else float("nan")
        if l < L - 1: swapk.append(sk)
        print(f"{l:>5} | {upd:>7.2f} | {cos:>5.2f} | {dk:>8.2f} | {sk:>8.2f}")

    import statistics as st
    print(f"\nmean drop-KL {st.mean(dropk):.2f} | frac<0.5 {sum(d<0.5 for d in dropk)/L:.0%} "
          f"| mean swap-KL {st.mean(swapk):.2f}")
    print("expected: large layer-0/early, low drop & swap KL across the deep interior "
          "(redundant, interchangeable) -> coverage-by-shallow viable for real models.")


if __name__ == "__main__":
    main()
