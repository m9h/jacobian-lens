# A covert error-monitoring signal in the open base model's workspace

**Run 2026-07-22. OLMo-3-1025-7B (base), published 31-layer Jacobian lens, 200 TriviaQA
questions + 10 unanswerable controls. Pure readout, no training.**

This began as the reviewers' **C2 (self-monitoring)** test — Dehaene & Naccache asked
whether the J-space "encodes the model's confidence, error detection, and its representation
of the boundary between what it knows and what it does not." But the result sits more
squarely in the **LLM-metacognition** literature than in global-workspace theory, so we
frame it there (survey: Liu et al., *Metacognition in LLMs*, arXiv 2607.11881, 2026).

## The finding

Reading the base model's workspace (the Jacobian lens at mid layers) recovers an
**uncertainty signal that predicts whether the model's own answer is wrong** — and it does so
**beyond what the model's output distribution reveals**. In signal-detection terms it is a
genuine metacognitive-sensitivity signal, and it is *covert*: present in the workspace even
when the output looks confident.

The model answered 78/200 correctly (39% — a real difficulty spread for a 7B base model).

### Structure — the signal emerges in the workspace band

Per-layer AUROC of the workspace uncertainty signal predicting an error:

```
early  L0–L12 :  0.46 – 0.57   (near chance — no signal)
workspace band:  L14 0.57 → L16 0.68 → L18 0.69 → L22 0.68   (sharp emergence)
late   L24–L30:  0.67 – 0.72   (peaks output-adjacent at L30)
```

The metacognitive signal is **absent early and switches on across the workspace band**,
strengthening toward the output. It is a workspace/late-layer phenomenon, cleanly localized
— echoing the interpretability-based "metacognitive space" that prior work reports lives in a
low-dimensional slice of the representation (Li et al., cited in the survey §4.1).

### The decisive control — covert, beyond output confidence

The model's own **output entropy** predicts its errors well (AUROC **0.803**) — better than
the workspace signal overall (0.69). So the workspace signal is *not* the best predictor
alone. The question the survey says the field has **not** answered is whether an internal
signal predicts correctness *independently of output probabilities*. Ours does: within
output-entropy terciles (matched output confidence),

```
low-entropy  (output looks confident):  workspace AUROC 0.643
mid-entropy:                             0.588
high-entropy:                            0.572
```

**Even when the output distribution is confident, the workspace separates right from wrong
at 0.64.** The workspace carries error information the output does not surface — strongest
exactly where it matters, on confident-looking answers. This is the "knows more than it says"
result, and the direct internal-vs-output comparison the survey (§4.2) notes is missing.

### Calibration — clean and monotonic

p(correct) by workspace uncertainty-signal quartile (low → high uncertainty):

```
Q1 low-unc  0.66     Q2  0.38     Q3  0.30     Q4 high-unc  0.22
```

### Known vs unknown — the workspace flags nonsense highest

Mean workspace uncertainty: **correct +3.48 < wrong +4.72 < unanswerable +5.03**. On 10
nonsense questions (no fact to retrieve — "capital of Zorblaxia?"), the workspace is *most*
uncertain — the unanswerable-detection signature that benchmarks like MetaMedQA target.

## Why this is a contribution (positioning against the survey)

The survey maps the field and, read against it, this result is novel on three axes at once:

1. **A workspace / Jacobian-lens latent-confidence probe.** The survey notes zero use of
   global-workspace or Jacobian methods, and no formal internal-vs-output contrast. The
   J-lens — a principled readout of *what the model is poised to report* — is a new instrument
   for latent metacognition.
2. **Internal signal beyond output probabilities.** The survey states prior work does "not
   directly compare whether internal hidden-state signals predict correctness independently
   of output probabilities." The within-entropy-tercile control does exactly that. (Compare
   Wang et al.'s DMC M-ratios: ~0.85–1.05 for token log-probs vs ~0.62–0.92 for self-reports
   — our output-entropy baseline is the analog, and the workspace adds signal on top.)
3. **In a base model.** The survey reports "no systematic characterization of when
   metacognitive abilities emerge during pretraining." This signal is in the *pretrained*
   base — before any Assistant identity. (See the ladder result below for how it changes
   across post-training; cf. Cacioli's base-vs-instruct comparison and the survey's note that
   "RLHF may specifically degrade metacognitive efficiency on STEM tasks.")

## Caveats (do not drop)

- **Output entropy is the stronger *overall* predictor** (0.80 vs 0.69). The workspace's
  value is the **residual/covert** part — information beyond output confidence — not a better
  standalone error detector. Framed accordingly.
- The peak is at L30 (output-adjacent), so it is "workspace **and** late," not uniquely
  mid-band; the signal accumulates toward the readout.
- The within-tercile control is robust and non-parametric; a 2-feature logistic fit and a
  proper meta-d′/M-ratio would tie it more tightly to the field's metrics (planned).
- n = 200, one base model. The "uncertainty signal" reads uncertainty-*word* logits; a
  supervised error direction may do better (a cross-validated direction probe is the natural
  next step).
- Not peer-reviewed; open-weights only.

## Across the post-training ladder

*(filled from `metacog_ladder` — base → Instruct → Think → RL-Zero, each with its own lens,
with an elicited-confidence P(True) control. Tests whether covert metacognition strengthens
or degrades with post-training — the survey's open emergence question.)*

Reproduce: `modal run modal_metacog.py::run` (base v2) and `::ladder` (across arms) in
`github.com/m9h/jacobian-lens`; raw on the Modal `jlens-out` volume, `metacog/`.
