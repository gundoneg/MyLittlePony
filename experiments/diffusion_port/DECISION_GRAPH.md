# Decision/result graph — supra50m → Mercury-2 by translator C

Built to answer: "what did we miss that could strongly improve quality?" Nodes are decisions
(D) and results (R); dashed edges were unexplored until run 4.

```mermaid
flowchart TD
    T[Thesis: B = C(A), B never trained,<br/>donor-sampled data only] --> D1

    subgraph phase10 [Phase 10 priors]
        P1[E1: teacher-forced + cascade >> end-to-end;<br/>rank~16 per-block correction suffices]
        P2[E0/E3: tied per-block C extrapolates depth<br/>for refinement-style interiors]
    end

    D1{D: how does C emit weights?} -->|raw U@Vt delta| R1[R run-1: NEGATIVE<br/>delta lands in wrong basis;<br/>signature control blind]
    R1 --> D2{D: output frame}
    D2 -->|SVD-frame covariant<br/>dW = Ur A(z) Vr^T| R2[R run-2: NEGATIVE<br/>zoo works, supra OOD;<br/>C = global smoothing]
    R2 --> D3{D: zoo distribution}
    D3 -->|self-zoo: supra's own<br/>truncated sub-stacks| R3[R run-3: POSITIVE<br/>B* beats floor at all t 5.1 vs 5.5-9.5;<br/>signature control separates 8.44 vs 4.97]

    R3 --> Q[remaining gaps:<br/>1. B* CE 5.1 >> donor AR 1.5<br/>2. generation degenerate<br/>3. blocks 10-11 hurt: 0..9-only 4.34 < all 4.97]

    Q -.->|missed root cause| M1[D run-4: CORPUS<br/>bare-BOS self-gen is degenerate 'as as as';<br/>fix: unigram-sampled random prompts -> diverse continuations<br/>still donor-only data]
    Q -.->|missed strength| M2[D run-4: SAMPLER<br/>B* inherits STRONG left-context skill from AR donor;<br/>semi-autoregressive block diffusion plays to it<br/>+ cosine reveal, Gumbel ranking, remasking, temp anneal, more steps]
    Q -.->|probe told us| M3[D run-4: add L=11 truncation<br/>blocks 0..10 seen; L=12 stays held out]
    Q -.->|capacity| M4[D run-4: subspace 24x24 -> 64x64<br/>still covariant; C ~0.6M params]
    Q -.->|unused knowledge channel| M5[D run-4: KD soft targets<br/>P_supra of x_t given left context at masked positions =<br/>literally the 'Q-A dataset captured from the model']
    P1 -.->|same lesson, reused| M5
    P2 --> D3

    M1 --> R4[R run-4: ?]
    M2 --> R4
    M3 --> R4
    M4 --> R4
    M5 --> R4
```

## Why each new edge should matter

| Path | Mechanism | Expected size |
|---|---|---|
| **Corpus v2** (unigram-prompted self-gen) | C currently learns to denoise degenerate text and is *evaluated on* degenerate text — the root bound on everything | Large, global |
| **Semi-AR block sampling** | B\* = AR weights + tiny deltas: its left-context skill is the donor's intact strength; LLaDA itself generates this way. From-scratch all-parallel is the *worst* mode for it | Large, on fluency |
| **Sampler tuning** (cosine schedule, Gumbel ranking, remasking, temp anneal, 2× steps) | standard MaskGIT/Dream improvements; remasking lets early mistakes be revised | Medium |
| **L=11 truncation** | the run-3 probe localized the damage to unseen blocks 10–11 (4.34 vs 4.97); L=12 composition stays the held-out claim | Medium, targeted |
| **Subspace 64×64** | run-3's A(z) lives in a top-24 SVD subspace; E1 needed free directions to reach KL 0.55 — widen reach while staying covariant | Medium |
| **KD soft targets** | one-hot targets on a tiny corpus are sparse; the AR teacher's distribution at masked positions is dense donor knowledge, and it *is* the allowed Q-A budget | Medium |

## Dead ends the graph rules out (do not revisit)
- Explicit Procrustes rotation of zoo→target (breaks RMSNorm symmetry; phase 3c).
- Independent fresh zoos for a single trained target (run-2 distribution gap) — unless the
  zoo donors are themselves well-trained models of the same family.
- Pure end-to-end logit-loss without in-frame/in-distribution structure (runs 1–2).
- Training B directly — out of contract (that is the BASELINE notebook's role).
