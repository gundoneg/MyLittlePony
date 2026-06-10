"""CPU smoke: exec c_mercury_port.ipynb code cells with tiny params against local supra50m.
Also checks the KV-cached AR sample is coherent English (validates the cache)."""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.join(HERE, "..", "shakespeare_port", "models", "supra50m")

nb = json.load(open(os.path.join(HERE, "c_mercury_port.ipynb")))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

os.environ["CKPT_DIR"] = LOCAL
repl = {
    r"N_GEN     = 1024": "N_GEN     = 4",
    r"N_HELD    = 64": "N_HELD    = 2",
    r"SEQ_LEN   = 256": "SEQ_LEN   = 24",
    r"ZOO_DEPTHS = \[2, 3, 4, 2, 3, 4\]": "ZOO_DEPTHS = [1, 2]",
    r"DONOR_STEPS = 800": "DONOR_STEPS = 4",
    r"DONOR_BS  = 16": "DONOR_BS  = 2",
    r"C_STEPS   = 3000": "C_STEPS   = 4",
    r"C_BS      = 8": "C_BS      = 2",
    r"chunks, need, gb = \[\], N_GEN \+ N_HELD, 64": "chunks, need, gb = [], N_GEN + N_HELD, 3",
    r"if step % 500 == 0": "if step % 2 == 0",
    r"hce = masked_ce_at\(fn, held_ids, 0.5, n=16\)": "hce = masked_ce_at(fn, held_ids, 0.5, n=2)",
    r"def masked_ce_at\(fn, ids, t_val, n=32\)": "def masked_ce_at(fn, ids, t_val, n=2)",
    r"held_ids\[:16, :-1\]": "held_ids[:2, :-1]",
    r"held_ids\[:16, 1:\]": "held_ids[:2, 1:]",
    r"x0 = held_ids\[:8\]": "x0 = held_ids[:2]",
    r"ids = torch.full\(\(2, 128\), MASK_ID": "ids = torch.full((2, 16), MASK_ID",
    r"frozen = torch.zeros\(2, 128, dtype=torch.bool": "frozen = torch.zeros(2, 16, dtype=torch.bool",
    r"gen = denoise\(fnB, ids, frozen, steps=64": "gen = denoise(fnB, ids, frozen, steps=4",
}
for k, v in repl.items():
    src, n = re.subn(k, v, src)
    assert n >= 1, f"replacement failed: {k}"

print("=== executing notebook code cells (CPU smoke) ===")
exec(compile(src, "c_mercury_port.ipynb", "exec"), {})
print("\n=== SMOKE OK ===")
