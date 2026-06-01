"""TinyShakespeare data pipeline: minimal BPE tokenizer + windowed batches.

No external deps beyond the stdlib here (torch is only used by the model code).
The BPE is a small, readable byte-pair encoder trained directly on the corpus.
"""
import json, os, re, random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "input.txt")
TOK = os.path.join(HERE, "data", "tokenizer.json")

# Split into "words" (runs that keep punctuation/newlines as their own pieces).
# A trailing </w> marker lets the decoder know where a word ends so we can
# reinsert spaces unambiguously.
_PAT = re.compile(r"\s+|[^\s]+")


def _words(text):
    """Yield (chars_tuple, joiner) preserving whitespace exactly."""
    out = []
    for m in _PAT.finditer(text):
        s = m.group(0)
        out.append(s)
    return out


class BPE:
    def __init__(self, vocab, merges):
        # vocab: list of tokens (strings); merges: list of (a,b) applied in order
        self.vocab = vocab
        self.stoi = {t: i for i, t in enumerate(vocab)}
        self.merges = merges
        self.rank = {tuple(m): i for i, m in enumerate(merges)}

    # ---- training ----
    @classmethod
    def train(cls, text, vocab_size=2048, verbose=True):
        # base alphabet = all bytes present, as single chars
        chars = sorted(set(text))
        vocab = list(chars)
        # represent corpus as list of symbol-lists (start from characters)
        # work at the whitespace-chunk level for speed
        chunks = Counter(_words(text))
        seqs = {w: list(w) for w in chunks}
        merges = []
        n_merges = max(0, vocab_size - len(vocab))
        for step in range(n_merges):
            pairs = Counter()
            for w, freq in chunks.items():
                s = seqs[w]
                for a, b in zip(s, s[1:]):
                    pairs[(a, b)] += freq
            if not pairs:
                break
            (a, b), cnt = pairs.most_common(1)[0]
            if cnt < 2:
                break
            new = a + b
            merges.append((a, b))
            vocab.append(new)
            # apply merge everywhere
            for w in seqs:
                s = seqs[w]
                if a not in s:
                    continue
                merged = []
                i = 0
                while i < len(s):
                    if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                        merged.append(new)
                        i += 2
                    else:
                        merged.append(s[i])
                        i += 1
                seqs[w] = merged
            if verbose and (step + 1) % 256 == 0:
                print(f"  merge {step+1}/{n_merges}: {a!r}+{b!r} -> {new!r} (count {cnt}), vocab {len(vocab)}")
        return cls(vocab, merges)

    # ---- encoding ----
    def _encode_chunk(self, w):
        s = list(w)
        while len(s) >= 2:
            # find best (lowest-rank) adjacent pair
            best, best_rank = None, None
            for i, pair in enumerate(zip(s, s[1:])):
                r = self.rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best = r, i
            if best is None:
                break
            i = best
            s[i : i + 2] = [s[i] + s[i + 1]]
        return s

    def encode(self, text):
        ids = []
        for w in _words(text):
            for tok in self._encode_chunk(w):
                ids.append(self.stoi[tok])
        return ids

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    # ---- persistence ----
    def save(self, path=TOK):
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f)

    @classmethod
    def load(cls, path=TOK):
        with open(path) as f:
            d = json.load(f)
        return cls(d["vocab"], [tuple(m) for m in d["merges"]])


def get_tokenizer(vocab_size=2048):
    if os.path.exists(TOK):
        return BPE.load()
    with open(DATA) as f:
        text = f.read()
    print(f"training BPE (target vocab {vocab_size}) on {len(text)} chars ...")
    bpe = BPE.train(text, vocab_size=vocab_size)
    bpe.save()
    print(f"saved tokenizer: {len(bpe.vocab)} tokens -> {TOK}")
    return bpe


class Batcher:
    """Holds the full token stream and serves random (x,y) windows."""

    def __init__(self, ids, ctx, val_frac=0.1, seed=0):
        n = len(ids)
        n_val = int(n * val_frac)
        self.train = ids[: n - n_val]
        self.val = ids[n - n_val :]
        self.ctx = ctx
        self.rng = random.Random(seed)

    def batch(self, split, bs, torch_mod):
        data = self.train if split == "train" else self.val
        ctx = self.ctx
        ix = [self.rng.randint(0, len(data) - ctx - 1) for _ in range(bs)]
        x = torch_mod.tensor([data[i : i + ctx] for i in ix], dtype=torch_mod.long)
        y = torch_mod.tensor([data[i + 1 : i + 1 + ctx] for i in ix], dtype=torch_mod.long)
        return x, y


def load_ids(bpe):
    cache = os.path.join(HERE, "data", "ids.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    with open(DATA) as f:
        text = f.read()
    ids = bpe.encode(text)
    with open(cache, "w") as f:
        json.dump(ids, f)
    return ids


if __name__ == "__main__":
    bpe = get_tokenizer()
    sample = "ROMEO:\nBut soft, what light through yonder window breaks?\n"
    ids = bpe.encode(sample)
    back = bpe.decode(ids)
    print(f"vocab size      : {len(bpe.vocab)}")
    print(f"sample          : {sample!r}")
    print(f"encoded ({len(ids)} toks): {ids}")
    print(f"decoded         : {back!r}")
    print(f"round-trip ok   : {back == sample}")
    ids_all = load_ids(bpe)
    print(f"corpus tokens   : {len(ids_all)}  (compression {len(open(DATA).read())/len(ids_all):.2f} chars/tok)")
