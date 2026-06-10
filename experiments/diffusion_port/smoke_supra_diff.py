"""CPU smoke: exec supra_to_mercury.ipynb code cells with tiny params against the local
supra50m checkpoint — catches shape/logic bugs before the Kaggle handoff."""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.join(HERE, "..", "shakespeare_port", "models", "supra50m")

nb = json.load(open(os.path.join(HERE, "supra_to_mercury.ipynb")))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

os.environ["CKPT_DIR"] = LOCAL
repl = {
    r"N_GEN    = 1536": "N_GEN    = 6",
    r"N_HELD   = 64": "N_HELD   = 2",
    r"SEQ_LEN  = 256": "SEQ_LEN  = 32",
    r"BS       = 16": "BS       = 2",
    r"STEPS    = 6000": "STEPS    = 6",
    r"WARMUP   = 100": "WARMUP   = 2",
    r"gb = 64": "gb = 4",
    r"def masked_ce_at\(model_, ids, t_val, n=32\)": "def masked_ce_at(model_, ids, t_val, n=2)",
    r"def ar_ce\(model_, ids, n=32\)": "def ar_ce(model_, ids, n=2)",
    r"if step % 500 == 0": "if step % 3 == 0",
    r"x0 = held_ids\[:8\]": "x0 = held_ids[:2]",
    r"length=128, steps=64": "length=32, steps=8",
    r"length=96, steps=48": "length=32, steps=8",
    r"prompt = held_ids\[1, :24\]\[None\]": "prompt = held_ids[1, :8][None]",
    r"print\('continued :', repr\(decode\(cont\[0, 24:\]\)\)\)": "print('continued :', repr(decode(cont[0, 8:])))",
    # generator on CPU: cuda generator fails; swap seed call",
    r"g = torch.Generator\(device=DEV\)\.manual_seed\(7\)": "g = torch.Generator(device=DEV).manual_seed(7)",
}
for k, v in repl.items():
    src, n = re.subn(k, v, src)
    assert n >= 1, f"replacement failed: {k}"

print("=== executing notebook code cells (CPU smoke) ===")
exec(compile(src, "supra_to_mercury.ipynb", "exec"), {})
print("\n=== SMOKE OK ===")
