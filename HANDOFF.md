# HANDOFF — J-space replication

**Written 2026-07-14 for a fresh agent. Read the "Where I was wrong" section first.**

You are inheriting a replication of Anthropic's *"Verbalizable Representations Form a
Global Workspace in Language Models"* ([paper](https://transformer-circuits.pub/2026/workspace/index.html),
[blog](https://www.anthropic.com/research/global-workspace), code
[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens), Apache-2.0,
"not maintained and not accepting contributions").

The user (Morgan Hough) has a meeting to discuss the paper. He is a systems engineer with
a neuroimaging background who develops in JAX and packages scientific software for Fedora.

## Repos

| repo | where | state |
|---|---|---|
| `jacobian-lens` | `~/Workspace/jacobian-lens` → `github.com/m9h/jacobian-lens` | 19 commits, pushed. Fork of Anthropic's, `upstream` wired. All experiments + results. |
| `jlens-lab` | `~/Workspace/jlens-lab` | 3 commits, **LOCAL ONLY — no remote**. Companion package + 4 tutorials. User has not authorised publishing it. |
| `societies-of-thought` | `~/Workspace/societies-of-thought` → `github.com/m9h/societies-of-thought` | 5 commits, pushed. Sibling project: adversarial probe of arXiv 2601.10825. See `docs/JSPACE.md` for how they connect. |

## ★ WHERE I WAS WRONG — challenge these first

I self-corrected five times. A fresh set of eyes should assume there is a sixth.

1. **I ran three variants before reproducing anything they published.** The user caught it.
   Cost ~10 GPU-hours and an invalid scaling sweep. *Reproduce a published number first.*
2. **I blamed under-fitting for an anomaly. Wrong.** Their converged lens gives near-identical
   eval numbers to my under-fit one (cosine 0.97). The real cause was the *metric*.
3. **I claimed the naive Mamba path was safer than fused kernels.** Backwards. It cost 73GB
   per layer and OOM'd a 119GB machine, taking down other users' jobs.
4. **My blockiness metric cheated**, maximising itself by isolating single layers.
5. **Raw blockiness rising with scale looked like "blocks emerge."** It was the decay profile
   steepening. Only the excess over a distance-only null means anything.
6. **I oversold the Nemotron experiment** (see below).

**Every one of these produced plausible output.** That is the theme. Assume the current
conclusions have the same problem until you have tried to break them.

## What the PAPER claims (not what we found)

A parallel session pointed out a structural gap in this document: it inventoried *our
findings* and never inventoried *the paper's assertions*. Those are different documents,
and only the second existed — so a claim we never tested was invisible here, because it
had no row.

The canonical inventory now lives in
`~/Workspace/societies-of-thought/docs/anthropic_claims_scorecard.md` — nine claims with
status. Do not duplicate it here; read it. Its provenance warning matters: most rows were
reconstructed from our own responses, not transcribed from the paper.

Two rows verified against the source text on 2026-07-19:

- **Capacity/bottleneck is stated with numbers.** The J-space component is *"never more
  than 10%"* of total activation variance and *"holds on the order of tens of concepts at
  a time."* Untested by us, and checkable from published lens files plus a forward pass.
- **Broadcast-back is NOT claimed — the paper concedes the opposite:** *"there are no
  obviously separable input processors, and the broadcast we document occurs within a
  single feedforward pass rather than through recurrent loops."* Nothing to refute. Do
  not spend on it.

The open one is **Claim 6** — post-training shaped the J-space toward a point of view
rather than pure prediction. Untestable from outside until AI2 published twelve OLMo-3
arms on one base. Design and confound analysis are in the scorecard.

## Findings, ranked by how much they matter

### 1. The tripartite CKA structure is mostly smooth drift — but real excess appears at ≥20B
The paper's headline figure (sensory / workspace / motor blocks). A **distance-only null**
(`C_null[i,j]` = observed mean CKA at `|i−j|`; zero blocks by construction) recovers
**79–91%** of it.

```
        model    L |    real    null   EXCESS
   qwen3-1.7b   27 |  0.1165  0.0915  +0.0250
     qwen3-4b   35 |  0.0775  0.0618  +0.0157
    olmo-3-7b   31 |  0.1839  0.1538  +0.0301
     qwen3-8b   35 |  0.2826  0.2493  +0.0332
    qwen3-14b   39 |  0.2934  0.2673  +0.0261
  gpt-oss-20b   23 |  0.2096  0.1565  +0.0530
  qwen3.5-27b   63 |  0.2670  0.2171  +0.0499   <- "nose" emerges here
```
Excess roughly **doubles at ≥20B**, at the same scale the behaviour appears. **Both sides
overclaimed**: Anthropic on the sharpness, Erik Hoel on the absence (his substitution
argument generalises from small models — the regime where the phenomenon doesn't exist).

⚠️ **My CKA construction is a RECONSTRUCTION.** The paper says only "geometrical matching"
and never specifies it. I used `v[l,t] = J_l^T @ W_U[t]` (the repo's own steering
convention). Blockiness and the null are **my** metrics. A fresh agent should attack this.

### 2. Anthropic's `pass@k` metric rewards noise
It scores a lens by **min rank over ~35 layers** — one lottery ticket per layer for a
diffuse distribution. At Qwen3.5-27B, on their own lens:

| lens | rank("nose") | top-5 |
|---|---|---|
| J-lens | **2** | `['smile','nose',"'^",'noses','grin']` |
| logit lens | 5 | `['Ċ','Âł','..','-','N']` |

Comparable ranks. One understands the face; one emits punctuation. **This is a genuine
measurement problem in the paper's own evaluation** and, as far as I know, nobody has
found it.

### 3. ★ RESOLVED 2026-07-17: the emergence is GRADUAL and SCALE-driven, not architectural
The one-prompt ASCII-face test was an ARTIFACT. The robust version -- the `association`
eval, 102 vignettes each evoking a concept that is NEVER named -- gives a smooth monotone
rise, and a DENSE model beats the hybrid:

```
 model            J@1     J@10    J@50      (J-lens, Anthropic's published lenses)
  0.6B          0.000    0.000   0.000
  1.7B          0.000    0.020   0.051
    4B          0.000    0.020   0.061
    8B          0.000    0.000   0.081
   14B          0.040    0.091   0.222
   27B hybrid   0.049    0.167   0.343
   32B DENSE    0.062    0.208   0.438   <- BEATS the hybrid at every k
```

**Both of my earlier stories were wrong.** "Sharp emergence at 27B" (from rank 164->2 on
the ASCII face) and the correction "emerges with linear attention/SSM" are both refuted:
on 102 prompts the effect is gradual, and the dense 32B is BEST. The ASCII face was one
lucky prompt.

**Consequence: do NOT fund QwQ-32B or Kimi-Linear-48B fits.** Their whole justification
was breaking the architecture confound, and the confound dissolved for free using lenses
we already had. (Kimi-K2-Thinking is infeasible anyway: 594GB. Kimi-Linear-48B is 98GB /
74% linear attention -- a near-perfect ratio match to Qwen3.5's 75% -- so it WOULD have
been the right independent replication had the hypothesis survived.)

**Also: the J-lens advantage SHRINKS with scale.** On association, 27B is 0.167 vs 0.098
logit (1.7x) but 32B is 0.208 vs 0.198 -- essentially tied. This echoes Anthropic's own
admission that the logit lens "captures much of the workspace-like structure."

### 4. The randomization control PASSES — a point *for* Anthropic
Randomize blocks, keep trained embed/unembed: the lens reads out **nothing**
(next_acc 0.0003, echo 0.0016). Its structure requires learned weights. **Anthropic never
ran this control.** It refutes the strong form of the popular "it's just backprop" critique
(which is also about the wrong derivative — see `jlens-lab/docs/01`).

### 5. The workspace is clearest in a model that is barely a transformer — but n=1
**Qwen3.5-27B is 48-of-64 linear attention.** It is tempting to say this undercuts the
architectural objection (Butlin & Long; Hoel) for free. BE CAREFUL: it is the only
positive, and Nemotron (also non-transformer) fails, so "clearest in non-transformers" is
not supported — "clearest in the single most capable model, which happens to be a hybrid"
is the honest statement. Do not lead with this.

### 6. Nemotron-H result (done 2026-07-15): the lens WORKS, the ASCII task FAILS
Fit converged (mean_rel_change 0.00176 at 1000 prompts — it actually converged, unlike
the timeout). ASCII-face rank(nose)=160 — does NOT surface. But the lens reads factual
prompts fine (`['Paris','France','Marseille','Louvre']`). So: a J-lens is fittable and
readable on a Mamba-2 hybrid (a real methods result — see jlens-lab layouts + the
73GB→9GB fused-kernel finding), but the unnamed-concept phenomenon does not appear at 4B.
This KILLS the "emerges with architecture" story and returns finding #3 to n=1.

## Critical technical gotchas (all cost real time)

- **`jlens.fit()` has NO stopping rule.** Anthropic fit to convergence
  (`--stop_at_delta 0.002 --min_prompts 100` — a **floor** — cap 1000). Published lenses used
  **454–775** prompts, growing with `d_model`. Fitting on 100 under-fits 4.6–6×, silently.
- **Anthropic publish converged lenses for 38 models** at `neuronpedia/jacobian-lens`, each
  with `config.yaml` (exact fit command) and `convergence.csv`. **Use theirs; don't fit.**
  `qwen3.6-27b`'s lens upload is **broken upstream** (only `.DS_Store`).
- **`model._init_weights()` is a silent no-op** in transformers v5 → a "randomized" control
  that is fully trained and reports a confident false PASS. Build from config, transplant
  embed/norm/head, and **assert**.
- **Mamba/SSM Jacobians need the fused kernels.** Naive path: **73.1 GB for ONE layer** of a
  4B model. Fused (`causal-conv1d` + `mamba-ssm`): **9.0 GB**. 8×. Not an optimisation.
- **`source_layers` must exclude the target layer**, or `jacobian_for_prompt` raises on every
  prompt. Don't wrap it in a bare `except: continue` — that hid this for 600 iterations.
- **`pgrep -f <pattern>` matches your own ssh/bash command line.** Use `[p]attern`.
- **MoE models materialise all experts in bf16** (Nemotron-3-Nano-30B-A3B is ~60GB, not 6GB).

## Infrastructure — READ THIS

- **NEVER run unbounded jobs on the DGX Spark (`ssh dgx`).** It is **shared with other
  agents**. Its 119GB is **unified**, so a GPU OOM is a **SYSTEM OOM** — it does not raise
  `CUDA out of memory`, it wedges the host. I did this; it needed a power cycle and destroyed
  other users' work.
- Use `sbatch jobs/jlens.sbatch <script>` (hard `--mem` cap) or Modal.
- `~/Workspace/slurm-add-spark.sh` — **unrun**. Adds the Spark to Slurm with cgroup
  enforcement. Needs the user's sudo.
- **Modal is authenticated on the Spark** (profile `morgan-hough`), not on Legion.
  `modal_app.py` runs there. The Spark only submits. Spent ~$25 so far; ~$16 of it wasted on
  a 6h timeout with no checkpointing (now fixed).

## Open threads

1. **Nemotron fit** — running on Modal, ETA ~2pm 2026-07-14, ~$8. Footnote, not headline.
2. **`jlens-lab` is unpublished.** User has not authorised a public repo. Ask.
3. **Erik Hoel / Bicameral Labs** (`erik@bicameral-labs.org`) — his critique explicitly asks
   for the CKA replication we ran, and his site says "we're assembling the team." He is a
   Tononi-trained IIT theorist attacking a rival theory. **Do not contact without the user's
   say-so.**
4. **Longview Digital Minds RFP closes 2026-07-24** — 10 days. Grants for "applied work"
   accept an *independent effort*, $50K–$2M, 200–1500 word proposal. The user now has
   completed, controlled results rather than a proposal. This is the highest-value action.
5. **A `/lathe` tutorial series** (`devenjarvis/lathe`) was proposed. Part 1 needs **no GPU** —
   the distance-only null runs on published lens files alone.
6. `societies-of-thought` calibration never finished (DGX power loss). Next command is
   `./scripts/run_stages.sh calibrate`.

## What I'd want a fresh agent to attack

- **Is the CKA reconstruction right?** It's the load-bearing finding and it's my invention.
- **Is `excess_over_null` the right statistic?** Would a different null (e.g. permuting layers)
  say something different?
- **Does the emergence threshold survive a second model family?** It's one family (Qwen3),
  and Qwen3.5-27B is architecturally different from Qwen3-14B (hybrid vs dense) — so
  "emergence at 27B" may be confounded with "hybrid architecture". **This is the biggest
  hole in the story and I have not closed it.** Test: run the ASCII-face demo on
  `qwen3-32b` (dense, 615-prompt lens published). If "nose" surfaces there, it's scale. If it
  doesn't, it's the architecture — and that changes the headline.
