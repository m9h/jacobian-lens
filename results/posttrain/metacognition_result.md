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

### Calibration and standard metrics

p(correct) by workspace uncertainty-signal quartile (low → high uncertainty) — monotonic:

```
Q1 low-unc  0.66     Q2  0.38     Q3  0.30     Q4 high-unc  0.22
```

Field-standard metrics, to connect to the metacognition literature (all on the base model,
n=200): the model's **output** confidence is moderately miscalibrated (**ECE 0.113**, Brier
0.195) and slightly *under*confident (mean top-1 prob 0.30 vs accuracy 0.39). Meta-sensitivity
as type-2 ROC (the ROC form of meta-d′): **workspace 0.69, output-entropy 0.80** — with
√2·z(AUROC) meta-d′ proxies of **0.71** and **1.21** respectively (cf. Wang et al.'s DMC
M-ratios ~0.85–1.05 for log-probs). The output distribution is the stronger overall predictor;
the workspace's contribution is the covert residual it adds beyond it (the within-entropy-tercile
result above).

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
   base — before any Assistant identity.

It also speaks to the **introspection debate** one level up (see `docs/technique-lineage.md`).
Singh, Linzen & Ravfogel's *"Can LLMs Introspect? A Reality Check"* argues the field lacks
"mechanistic evidence of a **dissociable second-order process**." The covert-vs-reportable
split here — internal error signal present in the base, verbal report only after post-training
— *is* that dissociation, on open weights. The companion concept-injection demonstration
(`modal_introspection.py`) adds the manipulation-check the Introspection paper (Lindsey, 2025)
lacks: reading the workspace through the lens to confirm an injected concept is genuinely
present rather than confabulated. (See the ladder result below for how it changes
   across post-training; cf. Cacioli's base-vs-instruct comparison and the survey's note that
   "RLHF may specifically degrade metacognitive efficiency on STEM tasks.")

## Caveats (do not drop)

- **Output entropy is the stronger *overall* predictor** (0.80 vs 0.69). The workspace's
  value is the **residual/covert** part — information beyond output confidence — not a better
  standalone error detector. Framed accordingly.
- The peak is at L30 (output-adjacent), so it is "workspace **and** late," not uniquely
  mid-band; the signal accumulates toward the readout.
- The within-tercile control is robust and non-parametric; the type-2 AUROC / meta-d′-proxy
  above ties it to the field's metrics, and a full Maniscalco–Lau meta-d′ fit would refine it.
- n = 200, one base model. The "uncertainty signal" reads uncertainty-*word* logits; a
  supervised error direction may do better (a cross-validated direction probe is the natural
  next step).
- Not peer-reviewed; open-weights only.

## Across the post-training ladder — covert signal in the base, reportability from post-training

Each arm answered 150 TriviaQA questions through **its own** lens; we measured the workspace
uncertainty signal, output entropy, and the model's **elicited self-evaluation** (P(True):
"Is this answer correct? Yes/No"), each against that arm's own correctness.

| arm | acc | workspace covert (low-entropy tercile) | verbal self-eval **P(True)** AUROC |
|---|---|---|---|
| **base** | 0.39 | **0.66** | **0.51 (chance)** |
| Instruct | 0.28 | 0.34 | **0.78** |
| Think | 0.23 | 0.61 | **0.72** |
| RL-Zero-Math | 0.39 | 0.52 | 0.52 |
| RL-Zero-General | 0.39 | 0.47 | 0.55 |

Two things, one clean and one caveated:

**Clean, readout-independent — post-training installs *reportability*.** The base model's
verbal self-evaluation is at **chance (0.51)** — it cannot tell you whether its own answer is
right — yet its workspace covertly tracks errors at **0.66**. Post-training raises verbal
self-evaluation to **0.78 (Instruct) / 0.72 (Think)**, but only the full SFT+DPO instruction/
CoT pipelines do it; **RL-Zero stays at chance (0.52–0.55)**. Same method-not-domain signature
as the geometry result: reportable self-monitoring is a product of instruction tuning, not
RLVR.

**The mechanistic upshot for the reviewers' C2 claim.** Dehaene & Naccache said the base model
has global availability (C1) but "does not seem to be imbued with self-monitoring (C2)," which
post-training installs. This sharpens it: the base model *already has* the self-monitoring
information — covert, in the workspace — and what post-training installs is **reportability**
of it. Since reportability is their operational criterion for conscious access, post-training
does not *create* the error signal; it makes a pre-existing covert signal verbalizable.

**The covert trajectory, de-confounded (v4 supervised probe).** The uncertainty-*word* readout
appeared to *weaken* after SFT (base 0.66 → Instruct 0.34), but that readout is calibrated on
the base workspace and the post-trained workspaces are reshaped, so the drop could be a transfer
artifact. Fitting a **correctness direction in each stage's *own* L18 workspace residual**
(cross-validated difference-in-means, `metacog_probe`) removes the confound:

| stage | supervised covert AUROC | | stage | supervised covert AUROC |
|---|---|---|---|---|
| base | 0.68 | | | |
| Instruct-SFT | **0.76** | | Think-SFT | 0.67 |
| Instruct-DPO | 0.76 | | Think-DPO | 0.72 |
| Instruct | **0.77** | | Think | 0.72 |

**The covert error signal does not weaken — it persists across every post-training stage, and
strengthens in the Instruct family (0.68 → 0.77).** The apparent drop was entirely the readout
not transferring. This flips the reading for the better: post-training does *not* trade a covert
signal for a reportable one. **The covert self-monitoring signal is present throughout (base and
all post-trained stages); SFT adds a verbalizable report *on top of* it.** The internal state
and its reportability are dissociable and independently present — the base model already
monitors its own errors; supervised fine-tuning grants the ability to say so.

### Stage-resolved emergence — reportability switches on at SFT

Sweeping every post-training stage (base → SFT → DPO → final) locates *where* reportable
self-monitoring appears. Verbal self-evaluation P(True), both families:

| stage | P(True) AUROC | | stage | P(True) AUROC |
|---|---|---|---|---|
| **base** | 0.51 (chance) | | | |
| **Instruct-SFT** | **0.71** (+0.19) | | **Think-SFT** | **0.70** (+0.18) |
| Instruct-DPO | 0.70 | | Think-DPO | 0.75 |
| Instruct (final) | 0.78 | | Think (final) | 0.72 |

**Reportable self-monitoring switches on at the SFT stage — the first post-training step — in
both families.** It is not gradual and does not wait for DPO: supervised fine-tuning installs
the bulk of it (+0.18–0.19 over base), and later stages only refine it. Combined with the
covert base-model signal, the developmental claim is precise: **the base model already
covertly tracks its own errors but cannot report them; SFT is the step that makes the
pre-existing covert signal reportable.** As far as we can find, this is the first developmental
localization of a consciousness-indicator property to a specific training stage, on open
weights — and the first live "emergence" cell of the open scorecard this program is building.
(The covert-readout *drop* after SFT is the confounded trajectory above; the P(True) curve is
behavioral and clean.)

Reproduce: `modal run modal_metacog.py::run` (base v2) and `::ladder` (across arms) in
`github.com/m9h/jacobian-lens`; raw on the Modal `jlens-out` volume, `metacog/`.
