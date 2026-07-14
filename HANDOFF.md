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

### 3. The flagship demo reproduces, and emerges sharply
ASCII-art face, lens read at the `^`. `rank(nose)`: **580 → 215 → 290 → 164 → 2** across
Qwen3 1.7B → 27B. Absent below 27B. Turns their anecdote into an emergence threshold.
**This validated the harness** — everything downstream is interpretable because of it.

### 4. The randomization control PASSES — a point *for* Anthropic
Randomize blocks, keep trained embed/unembed: the lens reads out **nothing**
(next_acc 0.0003, echo 0.0016). Its structure requires learned weights. **Anthropic never
ran this control.** It refutes the strong form of the popular "it's just backprop" critique
(which is also about the wrong derivative — see `jlens-lab/docs/01`).

### 5. The workspace is clearest in models that are barely transformers
**Qwen3.5-27B is 48-of-64 linear attention** and is where the phenomenon emerges. This
undercuts the architectural objection (Butlin & Long: "no obviously separable input
processors"; Hoel: LLMs "flatly lack" modularity/reentrant dynamics) **for free**, with no
extra experiment.

### 6. (footnote) Nemotron-H — I oversold this
Running now on Modal. **It is less important than I claimed.** It cannot show "attention is
unnecessary" (it still has 4 attention layers), it's incremental over the Qwen3.5
observation, and — fatally — it does **not** address the strongest deflationary objection
(*the J-space is a property of the residual stream*), because Mamba hybrids have residual
streams too. That objection was already answered by finding #4.

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
