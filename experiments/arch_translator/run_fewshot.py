"""Phase 10 / E3 — few-shot depth adaptation of C. E0 showed that with the pipeline fixed,
shallow->deep transfer works at alpha=0 but keeps a real gap at alpha=1 (genuinely recursive
depth). Question: how many DEEP donors does fine-tuning C need to close that gap, and WHERE in
C does the depth knowledge live?

Protocol (post-E0-fix, common-frame + scale_free):
  * train C on a shallow L=2 zoo;
  * for n in {0,1,2,4}: fine-tune a COPY of C on n deep donors, zero-shot on HELD deep donors
    (disjoint from the n) -> donors-needed curve. n=0 is the E0 transfer baseline.
  * parameter-group ablation at fixed n: fine-tune only {vocab maps | encoder | M_block | all}
    -> localize the depth gap.
Deep donors use diverse maxjump so the interior is genuinely donor-specific (avoids the
constant-interior confound). alpha=1 (the regime where E0 left a gap).

Run: python run_fewshot.py
"""
import argparse, copy, time
import torch

from models import LM
from task import sample_tasks_interdep, masked_ce
from donor_zoo import build_donor_zoo
from xlate import PerLayerTranslator, train_translator, eval_translator, run_B, batch_for
from run_interdep import cfg_at
from run_gen import probe_inputs
from run_cover_fixed import common_frame

CHANCE = 1 / 16
TKW = dict(translator_cls=PerLayerTranslator, depth_frac=True, scale_free=True)


def diverse_deep_tasks(n, cfg, depth, alpha, seed):
    tasks = []
    for j in range(n):
        mj = 1 + (j % 3)                                   # maxjump in {1,2,3}
        tasks += sample_tasks_interdep(1, cfg.vocab, depth=depth, alpha=alpha,
                                       maxjump=mj, seed=seed + 1000 * j)
    return tasks


def finetune(C, cfg_d, zoo, steps, params, lr=2e-3, tps=6, bs=64):
    if not zoo or steps == 0:
        return
    template = LM(cfg_d, "ssm")
    opt = torch.optim.AdamW(params, lr=lr)
    g = torch.Generator().manual_seed(123)
    C.train()
    for _ in range(steps):
        idx = torch.randperm(len(zoo), generator=g)[:tps]
        opt.zero_grad()
        loss = 0.0
        for i in idx:
            z = zoo[int(i)]
            x, y = batch_for(z["task"], bs, cfg_d, g)
            loss = loss + masked_ce(run_B(template, C.emit(z["A"]), x), y)
        (loss / len(idx)).backward()
        opt.step()


def zs(C, cfg_d, held):
    return eval_translator(C, LM(cfg_d, "ssm"), cfg_d, held)


def groups(C):
    return {"vocab": [C.W_t, C.W_h, C.W_p],
            "encoder": list(C.enc.parameters()),
            "M_block": [C.M_block, C.norm_w],
            "all": list(C.parameters())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--shallow", type=int, default=2)
    ap.add_argument("--deep", type=int, default=3)
    ap.add_argument("--n_shallow", type=int, default=10)
    ap.add_argument("--n_ft", type=int, default=4, help="deep fine-tune pool")
    ap.add_argument("--n_held", type=int, default=4, help="held deep targets (disjoint)")
    ap.add_argument("--c_steps", type=int, default=300)
    ap.add_argument("--ft_steps", type=int, default=150)
    args = ap.parse_args()
    t0 = time.time()
    cfg_s, cfg_d = cfg_at(args.shallow, 64), cfg_at(args.deep, 64)

    # zoos (independent init) -> one common frame via the E0-fixed pipeline
    s_tasks = sample_tasks_interdep(args.n_shallow, cfg_s.vocab, depth=args.shallow,
                                    alpha=args.alpha, maxjump=2, seed=7)
    d_tasks = diverse_deep_tasks(args.n_ft + args.n_held, cfg_d, args.deep, args.alpha, seed=200)
    s_raw = build_donor_zoo(cfg_s, s_tasks, steps=400 * args.shallow, tied=False, verbose=False)
    d_raw = build_donor_zoo(cfg_d, d_tasks, steps=400 * args.deep, tied=False, verbose=False)
    probe_s = probe_inputs(s_tasks[0], cfg_s, 64)
    probe_d = probe_inputs(d_tasks[0], cfg_d, 64)
    s_fix, d_fix = common_frame(s_raw, d_raw, cfg_s, cfg_d, probe_s, probe_d)
    ft_pool, held = d_fix[:args.n_ft], d_fix[args.n_ft:]    # disjoint

    print("=" * 70)
    print(f"E3 few-shot depth adaptation  L{args.shallow}->L{args.deep}  alpha={args.alpha}  chance {CHANCE:.1%}")
    print(f"  donor-acc shallow {sum(z['acc'] for z in s_raw)/len(s_raw):.0%} | "
          f"deep {sum(z['acc'] for z in d_raw)/len(d_raw):.0%}")
    print("=" * 70)

    C0, _ = train_translator(cfg_s, s_fix, steps=args.c_steps, tasks_per_step=6, **TKW)
    nat, _ = train_translator(cfg_d, ft_pool + held, steps=args.c_steps, tasks_per_step=6, **TKW)
    print(f"  native ceiling (C trained on deep)            zs {zs(nat, cfg_d, held):.1%}")
    print(f"\n  donors-needed curve (fine-tune ALL params, {args.ft_steps} steps):")
    for n in sorted({0, 1, 2, args.n_ft}):
        C = copy.deepcopy(C0)
        finetune(C, cfg_d, ft_pool[:n], args.ft_steps, list(C.parameters()))
        print(f"    n={n}  deep donors -> held zs {zs(C, cfg_d, held):.1%}")

    print(f"\n  parameter-group ablation (n={min(2,args.n_ft)} deep donors, where the gap lives):")
    for name, ps in groups(C0).items():
        C = copy.deepcopy(C0)
        gp = {"vocab": [C.W_t, C.W_h, C.W_p], "encoder": list(C.enc.parameters()),
              "M_block": [C.M_block, C.norm_w], "all": list(C.parameters())}[name]
        finetune(C, cfg_d, ft_pool[:2], args.ft_steps, gp)
        print(f"    fine-tune {name:>8} -> held zs {zs(C, cfg_d, held):.1%}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
