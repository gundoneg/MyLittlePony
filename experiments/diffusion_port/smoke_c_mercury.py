"""CPU smoke: exec c_mercury_port.ipynb (run 4) code cells with tiny params against local
supra50m — catches shape/logic bugs in corpus v2, samplers, 64x64 emit, and KD before Kaggle."""
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
    r"N_BOOT    = 128": "N_BOOT    = 2",
    r"TRUNC_DEPTHS = \[2, 4, 6, 8, 10, 11\]": "TRUNC_DEPTHS = [1, 2]",
    r"C_STEPS   = 6000": "C_STEPS   = 4",
    r"C_BS      = 8": "C_BS      = 2",
    r"SUB       = 64": "SUB       = 8",
    r"chunks, need, gb = \[\], N_GEN \+ N_HELD, 64": "chunks, need, gb = [], N_GEN + N_HELD, 3",
    r"if step % 500 == 0": "if step % 2 == 0",
    r"hce = masked_ce_at\(fn, held_ids, 0.5, n=16\)": "hce = masked_ce_at(fn, held_ids, 0.5, n=2)",
    r"def masked_ce_at\(fn, ids, t_val, n=32\)": "def masked_ce_at(fn, ids, t_val, n=2)",
    r"held_ids\[:16, :-1\]": "held_ids[:2, :-1]",
    r"held_ids\[:16, 1:\]": "held_ids[:2, 1:]",
    r"x0 = held_ids\[:8\]": "x0 = held_ids[:2]",
    r"rec = denoise_v2\(fn, x_c.clone\(\), ~m, steps=32": "rec = denoise_v2(fn, x_c.clone(), ~m, steps=4",
    r"ids = torch.full\(\(2, 128\), MASK_ID": "ids = torch.full((2, 16), MASK_ID",
    r"g1 = denoise\(fnB, ids.clone\(\), fr, steps=64": "g1 = denoise(fnB, ids.clone(), fr, steps=4",
    r"g2 = denoise_v2\(fnB, ids.clone\(\), fr, steps=128": "g2 = denoise_v2(fnB, ids.clone(), fr, steps=4",
    r"g3 = semi_ar_generate\(fnB, bos, 129, block=32, steps_per_block=32": "g3 = semi_ar_generate(fnB, bos, 9, block=4, steps_per_block=3",
    r"prompt = held_ids\[1, :24\]\[None\].repeat\(2, 1\)": "prompt = held_ids[1, :8][None].repeat(2, 1)",
    r"g4 = semi_ar_generate\(fnB, prompt, 24 \+ 96, block=32, steps_per_block=32": "g4 = semi_ar_generate(fnB, prompt, 8 + 8, block=4, steps_per_block=3",
    r"print\('semi-AR continuation:', repr\(decode\(g4\[0, 24:\]\)\)\)": "print('semi-AR continuation:', repr(decode(g4[0, 8:])))",
}
for k, v in repl.items():
    src, n = re.subn(k, v, src)
    assert n >= 1, f"replacement failed: {k}"

print("=== executing notebook code cells (CPU smoke) ===")
exec(compile(src, "c_mercury_port.ipynb", "exec"), {})
print("\n=== SMOKE OK ===")
