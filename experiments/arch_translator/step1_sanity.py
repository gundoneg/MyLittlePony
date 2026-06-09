"""Step 1: confirm A (Transformer) and B (SSM) are both ALIVE and GENUINELY DIFFERENT.

Trains one tiny A and one tiny B on char-shakespeare, reports val loss vs the
random baseline, prints param/mixer summaries, and a short sample from each.
"""
import math
import time
import torch
from data import CharData
from models import Cfg, LM, n_params


def train(model, data, steps, ctx, bs=32, lr=3e-3, log=100):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    model.train()
    t0 = time.time()
    for s in range(1, steps + 1):
        x, y = data.batch("train", bs, ctx, generator=g)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % log == 0 or s == 1:
            print(f"    step {s:4d}  train loss {loss.item():.3f}  ({time.time()-t0:.0f}s)")
    return model


@torch.no_grad()
def val_loss(model, data, ctx, bs=64, iters=20):
    model.eval()
    g = torch.Generator().manual_seed(123)
    tot = 0.0
    for _ in range(iters):
        x, y = data.batch("val", bs, ctx, generator=g)
        _, loss = model(x, y)
        tot += loss.item()
    return tot / iters


def mixer_params(model, kind):
    keys = ("qkv", "o.") if kind == "transformer" else ("in_proj", "a_logit", "b", "cc", "conv")
    return [n for n, _ in model.named_parameters() if any(k in n for k in keys)]


def main():
    torch.manual_seed(0)
    data = CharData()
    c = Cfg(vocab=data.vocab)
    print(f"vocab={data.vocab}  d_model={c.d_model}  n_layer={c.n_layer}  ctx={c.ctx}")
    print(f"random baseline (val) = ln(vocab) = {math.log(data.vocab):.3f}\n")

    steps = 600
    results = {}
    for kind in ("transformer", "ssm"):
        print(f"=== train {kind} ===")
        torch.manual_seed(1)
        m = LM(c, kind)
        print(f"  params: {n_params(m):,}   mixer params: {mixer_params(m, kind)}")
        train(m, data, steps, c.ctx)
        vl = val_loss(m, data, c.ctx)
        results[kind] = (m, vl)
        print(f"  -> val loss {vl:.3f}  (random {math.log(data.vocab):.3f})\n")

    print("=== sanity ===")
    a_keys = set(n for n, _ in results["transformer"][0].named_parameters())
    b_keys = set(n for n, _ in results["ssm"][0].named_parameters())
    only_a = sorted(k for k in a_keys - b_keys)
    only_b = sorted(k for k in b_keys - a_keys)
    print(f"  A-only params (attention): {only_a}")
    print(f"  B-only params (ssm):       {only_b}")
    print(f"  shared skeleton params:    {len(a_keys & b_keys)}\n")

    ctx0 = data.encode("\n").view(1, -1)
    for kind in ("transformer", "ssm"):
        m = results[kind][0]
        out = m.generate(ctx0.clone(), 120)
        print(f"--- {kind} sample ---\n{data.decode(out[0])}\n")

    print("=== verdict ===")
    rnd = math.log(data.vocab)
    for kind in ("transformer", "ssm"):
        vl = results[kind][1]
        alive = "ALIVE" if vl < rnd - 0.5 else "DEAD?"
        print(f"  {kind:11s} val {vl:.3f}  {alive}")


if __name__ == "__main__":
    main()
