"""CPU exec-smoke of the ACTUAL qwen_mercury_port.ipynb cells (not a parallel script):
extracts every code cell, shrinks budgets, points MODEL_ID at the local supra50m checkpoint
(a real HF Llama -> the architecture-agnostic path applies), forces fp32 for CPU, and runs
the whole pipeline end-to-end: load -> conv shim -> run_stack asserts -> diffusion core ->
C/SVD -> DRY-RUN -> corpus -> train -> attribution -> controls -> save pack.

This is what "проверяй код полностью перед отправкой" means: execute the shipped cells."""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SUPRA = os.path.join(HERE, "..", "shakespeare_port", "models", "supra50m")

nb = json.load(open(os.path.join(HERE, "qwen_mercury_port.ipynb")))
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

repl = {
    r"MODEL_ID = 'Qwen/Qwen3\.5-0\.8B' if PILOT else 'Qwen/Qwen3\.5-9B'": f"MODEL_ID = {SUPRA!r}",
    r"N_GEN    = 256 if PILOT else 512": "N_GEN    = 6",
    r"N_HELD   = 32  if PILOT else 48": "N_HELD   = 4",
    r"SEQ_LEN  = 160": "SEQ_LEN  = 24",
    r"C_STEPS  = 2000 if PILOT else 1500[^\n]*": "C_STEPS  = 4",
    r"C_BS     = 4   if PILOT else 2": "C_BS     = 2",
    r"SUB      = 48[^\n]*": "SUB      = 8",
    r"EVAL_EVERY = 200 if PILOT else 400": "EVAL_EVERY = 2",
    r"torch_dtype=torch\.float16": "torch_dtype=torch.float32",   # CPU has no fp16 matmul path
}
for k, v in repl.items():
    src, n = re.subn(k, v, src)
    assert n >= 1, f"replacement failed: {k!r}"

print("=== executing the REAL qwen_mercury_port.ipynb cells (CPU smoke vs supra50m) ===")
exec(compile(src, "qwen_mercury_port.ipynb", "exec"), {})
print("\n=== SMOKE OK: every cell of the shipped notebook ran end-to-end ===")
