"""CPU smoke test: exec the e1_kaggle.ipynb code cells with tiny params against the local
supra50m, to catch shape/registration/logic bugs before handing the notebook off."""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.join(HERE, "..", "shakespeare_port", "models", "supra50m")

nb = json.load(open(os.path.join(HERE, "e1_kaggle.ipynb")))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

os.environ["CKPT_DIR"] = LOCAL              # resolve_model() picks this up (no HF download)
repl = {
    r"RANKS    = \[0, 1, 4, 16, 64\]": "RANKS    = [0, 4]",
    r"N_SEQS   = 120": "N_SEQS   = 6",
    r"SEQ_LEN  = 128": "SEQ_LEN  = 32",
    r"FIT_STEPS = 300": "FIT_STEPS = 5",
    r"INTERIOR = list\(range\(2, m\.L - 1\)\)": "INTERIOR = [3, 4]",
}
for k, v in repl.items():
    src, n = re.subn(k, v, src)
    assert n == 1, f"replacement failed ({n}x): {k}"

print("=== executing notebook code cells (CPU smoke) ===")
exec(compile(src, "e1_kaggle.ipynb", "exec"), {})
print("\n=== SMOKE OK ===")
