"""Sample from the donor (Transformer) and/or the ported target (SSM).

One-shot:
    python chat.py --prompt "ROMEO:" --max_new 200 --temp 0.8 --top_k 40
Interactive REPL (if you have a terminal):
    python chat.py            # type a prompt, see both models' continuations
Commands in the REPL: :temp X, :topk K, :len N, :model both|donor|target, :quit
"""
import argparse, os

import torch

from data import get_tokenizer
from donor import GPT, Config
from target import SSMLM

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path, Model):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, weights_only=False)
    cfg = Config(**ckpt["cfg"])
    m = Model(cfg)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m


def run(model, bpe, prompt, max_new, temp, top_k):
    ids = bpe.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, max_new, temperature=temp, top_k=top_k)
    return bpe.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--model", choices=["both", "donor", "target"], default="both")
    args = ap.parse_args()

    bpe = get_tokenizer()
    donor = load(os.path.join(HERE, "donor.pt"), GPT)
    target = load(os.path.join(HERE, "target.pt"), SSMLM)
    models = {}
    if donor is not None:
        models["donor (Transformer)"] = donor
    if target is not None:
        models["target (SSM)"] = target
    if not models:
        raise SystemExit("no checkpoints found; train_donor.py / transfer.py first")

    def which():
        if args.model == "donor":
            return {k: v for k, v in models.items() if "donor" in k}
        if args.model == "target":
            return {k: v for k, v in models.items() if "target" in k}
        return models

    def emit(prompt):
        for name, m in which().items():
            print(f"\n===== {name} =====")
            print(run(m, bpe, prompt, args.max_new, args.temp, args.top_k))

    if args.prompt is not None:
        emit(args.prompt)
        return

    print("interactive sampling. commands: :temp X  :topk K  :len N  :model both|donor|target  :quit")
    while True:
        try:
            line = input("\nprompt> ").rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.startswith(":"):
            parts = line[1:].split()
            cmd = parts[0] if parts else ""
            if cmd in ("quit", "q"):
                break
            elif cmd == "temp":
                args.temp = float(parts[1])
            elif cmd == "topk":
                args.top_k = int(parts[1])
            elif cmd == "len":
                args.max_new = int(parts[1])
            elif cmd == "model":
                args.model = parts[1]
            else:
                print("unknown command")
            print(f"[temp={args.temp} top_k={args.top_k} len={args.max_new} model={args.model}]")
            continue
        emit(line)


if __name__ == "__main__":
    main()
