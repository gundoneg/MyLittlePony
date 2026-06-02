"""Char-level data for the arch-translator sandbox (reuses tinyshakespeare)."""
import os
import torch

_HERE = os.path.dirname(__file__)
_DEFAULT = os.path.join(_HERE, "..", "shakespeare_port", "data", "input.txt")


class CharData:
    def __init__(self, path=_DEFAULT, device="cpu"):
        with open(path, "r") as f:
            text = f.read()
        chars = sorted(set(text))
        self.vocab = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(ids))
        self.train = ids[:n].to(device)
        self.val = ids[n:].to(device)
        self.device = device

    def batch(self, split, batch_size, ctx, generator=None):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - ctx - 1, (batch_size,), generator=generator)
        x = torch.stack([data[i:i + ctx] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + ctx] for i in ix])
        return x.to(self.device), y.to(self.device)

    def encode(self, s):
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)
