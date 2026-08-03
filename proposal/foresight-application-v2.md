# What Actually Moved the Score? Construct Validity for Reasoning Benchmarks

**Foresight Institute — AI for Science & Safety Nodes** · focus: **AI for Science & Epistemics**
· Morgan Hough, Orthogonal Research and Education Lab (OREL) · San Francisco hub, weekly in
person · **DRAFT v2 — reframed around reasoning models and benchmark attribution**

---

## The problem: we forecast from numbers whose causes we do not measure

AGI timelines are argued from benchmark movement, and ARC-AGI is the benchmark that argument
runs on. But the recent movement was **not primarily model capability**:

- A **7-million-parameter** Tiny Recursive Model reaches ~45% on ARC-AGI-1 — essentially all
  task-specific learning happening at *test time*.
- NVIDIA's winning 2025 entry used a **4B** model with synthetic data and test-time training.
- Test-time training reached 53.5% on the private ARC-AGI-1 set; evolutionary program synthesis
  in *natural-language* program space is a separate line again.
- The ARC Prize 2025 technical report's own summary: **"test-time adaptation and refinement loops
  emerge as critical success factors."**

So a score moved. Five things could have moved it: **the model, the harness, test-time compute,
synthetic data, or contamination.** These have completely different implications for what comes
next, and the public conversation routinely attributes all of it to the first.

**This is an epistemics problem before it is a capabilities problem**, and it is squarely in
Foresight's stated interest in forecasting and short-timeline feasibility.

## Why cognitive science, and why this is NeuroAI

ARC-AGI is not an arbitrary benchmark. Chollet built it on **psychometrics** — the measurement
theory psychology developed precisely because "this test score went up" is not the same as "this
ability improved" — and on Spelke's core-knowledge priors. The discipline that owns this problem
calls it **construct validity**, and it has a century of methods for it: does the instrument
measure the ability it names, or something correlated and cheaper?

That places this work directly in **path 4 of Mineault et al., *NeuroAI for AI Safety***
(arXiv:2411.18526) — *"interpretability advancement using neuroscience methods"* — and extends
it where the white paper is explicitly thin, since it "emphasizes neuroscience-driven approaches
rather than traditional benchmarking." **The neuroscience contribution here is methodological,
not architectural: importing measurement standards, not brain structure.**

Two examples of what that buys, both already run:

- Psychology measures metacognition with **meta-d′ and type-2 ROC** because "the model's
  confidence tracks its accuracy" is too loose to test. Most AI metacognition work reinvents a
  weaker version.
- Psychophysics insists an effect be shown **within item**, not only between items. Applying that
  to our own results is an open exposure we have published rather than hidden.

## What has been built, on ~$250 of compute

Every artifact public, open weights only, and the record includes the failures because for an
epistemics proposal those are the evidence.

**We have already done benchmark attribution once, and the answer reversed.** A 2026 paper
reported that steering a single feature produced a "society of thought" that improved reasoning
accuracy. Reproduced on the paper's own benchmarks, the gain **inverts** — +10 points on one,
−22 on another. The effect was specific to the benchmark it was measured on. The original had no
public code or data; ours does.

**We can separate what post-training changes from what it does not.** Across the fully open
OLMo-3 family, instruction tuning moves the model's internal representation ~10× more than
reinforcement learning with verifiable rewards, while task capability stays flat — and varying
the *quantity* of tuning data across a 300× sweep changes nothing measurable. Method matters;
scale of data does not. That is the same class of question as "was it the model or the harness."

**Two published retractions of our own.** A false negative about another lab's method, traced to
two bugs of ours. A sparsity claim that was mostly model family. Both public, with the corrected
analyses.

**A correction that strengthened our own headline.** Our published effect pooled measurements
whose noise floors differed by an order of magnitude. Correcting it halved one number and doubled
another.

**Artifacts:** 4 repositories, 4 datasets, Apache-2.0 — including the first public per-head
scores across a full training checkpoint sequence for a flagship 2022 result whose own models
were never released.

## Proposed work

**1. An attribution harness for reasoning benchmarks.** For a given score movement, decompose it:
model swap, harness swap, test-time compute budget, data. ARC Prize already publishes
**cost-per-task** alongside score and separates base LLMs / reasoning systems / compute-capped
Kaggle entries — the apparatus exists and is underused. Deliverable: a public, re-runnable
attribution for a set of headline results, on open models and open harnesses.

**2. Construct-validity tests, imported from psychometrics.** Does an ARC-style score predict the
ability it names, or the harness that produced it? Concretely: within-item controls, transfer to
held-out task families, and sensitivity to test-time budget at fixed model. **ARC-AGI-3 is
interactive and agentic**, which makes this sharper — an agent's score confounds reasoning with
exploration policy, and separating them is exactly a psychophysics design problem.

**3. Contamination and provenance checks**, which the benchmark community treats as a footnote
and psychometrics treats as disqualifying.

**4. The reproduction group at the hub** (below), which produces the implementations these
analyses run on.

## Why this reduces risk

Timeline estimates drive policy, investment, and safety prioritisation. If a jump attributed to
model capability was mostly a scaffold that will not generalise — or mostly contamination — then
the forecast built on it is wrong in a direction that matters. **Attribution is upstream of
forecasting, and forecasting is upstream of nearly every safety decision.**

It is also upstream of evaluation integrity generally: our interpretability work has already
surfaced a model's internal representation *that it was being evaluated*, whose ablation
increased dishonest behaviour.

## The in-person activity

A **reproduction group** at the hub: exciting papers with incomplete artifacts get a public
open-source implementation, training documentation, and the control the original omitted. It
continues an existing San Francisco cognitive-science reading group with a thirty-session,
three-year record, and is organised on the 1978 cognitive-science hexagon — because most of these
claims import a construct from a discipline that is not in the room. Five papers have already
been through the process, two of which produced a refutation or a correction.

## Budget

**$48,000 over 12 months** ($4,000/month): **$43,000** researcher time, **$5,000** compute. About
$250 of compute produced everything above, so this is 20× headroom; hardware is already in place.
**The ask is time, not resources.**

## Qualifications

Systems engineering, computational neuroscience, and scientific-software maintenance (I maintain
ACT-R, Soar and Nengo packages for Fedora). The work engages the neuroscience the AI claims
borrow from — including the COGITATE adversarial collaboration, whose own preregistered
prediction was *not* confirmed in humans, which is the sort of thing an attribution project
should be willing to say out loud about its own source disciplines.

---

*github.com/m9h/{jacobian-lens, spinning-up-in-mech-interp, tri-lens, controls-and-trajectories}
· huggingface.co/mhough. Every result, control and retraction reproducible from published open
artifacts.*
