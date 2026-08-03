# What Actually Moved the Score? Construct Validity for Reasoning Benchmarks

**Foresight Institute — AI for Science & Safety Nodes** · focus: **AI for Science & Epistemics**
· Morgan Hough, Orthogonal Research and Education Lab (OREL) · San Francisco hub, weekly in
person · **DRAFT v2 — reframed around reasoning models and benchmark attribution**

---

## The problem: we forecast from numbers whose causes we do not measure

AGI timelines are argued from benchmark movement, and ARC-AGI is where that argument runs. The
decomposition has been done, and it is lopsided:

| study | harness effect | base-model effect |
|---|---|---|
| Product of Experts ([2505.07859](https://arxiv.org/abs/2505.07859)) | **+53.3 pts** on a *frozen* 8B model | **+3.4 pts** (3B → 8B) |
| Cost-Effective Harnesses ([2607.06764](https://arxiv.org/abs/2607.06764)) | **+51.75 pts**, and it replicates across base models | — |
| Berman, evolutionary search in natural language | **+12.9 / +13.4 pts** on a fixed base model | — |

A 7-million-parameter model reaches ~45% on ARC-AGI-1; the 2025 Kaggle winner used a **4B** base.
ARC Prize's own 2024 report concludes that *"there does not exist any static inference-style
transduction solution that scores above 10%"* — i.e. essentially all reported performance is
harness.

**So the attribution question is settled for ARC and unasked everywhere else.** The gap this
project addresses is not "prove harness matters on ARC" — that is done. It is that **no reusable
method exists for asking it of the next benchmark**, and the field keeps reporting scores as
capability. Three specific consequences:

- The **API track and the Kaggle track differ by ~68 points** on ARC-AGI-2 (92.5% vs ~24%) under
  different compute constraints. Conflating them is the most common error in secondary coverage,
  including in timeline arguments.
- **`arc_challenge` in mainstream evaluation harnesses is a different benchmark entirely** — the
  2018 AI2 Reasoning Challenge, grade-school science multiple choice. Headline claims routinely
  cite it as ARC-AGI. There is no ARC-AGI evaluation in any mainstream harness.
- Published results exist whose post-method numbers look like contamination and whose baselines
  are sound. Distinguishing those requires the artifacts, not the paper.

## Why open artifacts, and what cognitive science is actually for

The reason to insist on open code, models and datasets is not that these systems resemble
brains. It is that **they are computational objects we do not understand, and understanding them
requires continuing to probe them.** A closed model can be described; it cannot be interrogated.
A published number without the artifact behind it can be believed or doubted, but not checked.

Attribution makes this concrete. To decide whether a score moved because of the model or the
harness, you have to **hold one fixed and swap the other** — which requires both to be open, and
requires the evaluation harness to be re-runnable. Almost no headline result on a reasoning
benchmark currently permits that. That is the gap, and it is a tooling and artifact gap, not a
theoretical one.

**Where cognitive science comes in — and where it does not.** It is not a claim that brain-like
systems are safer, and this proposal explicitly declines the architectural version of that
argument. The contribution is narrower and entirely methodological:

- **Psychometrics** already has the machinery for "did the score move for the reason claimed" —
  construct validity, developed because IQ testing failed in exactly this way. ARC-AGI is itself
  built on that tradition.
- **Signal detection theory** (meta-d′, type-2 ROC) already formalises "does the system know when
  it is wrong". AI work repeatedly reinvents a weaker version.
- **Psychophysics** insists effects be shown *within item*, not only between items — a control
  that has already made a collaborator's effect disappear, and one of our own open exposures.
- **Adversarial collaboration and preregistration** are how disputes between labs get settled.
  These fields built that machinery after a replication crisis; AI evaluation is pre-crisis.

So the relationship to Mineault et al.'s *NeuroAI for AI Safety* (arXiv:2411.18526) — Foresight's
reference — is partial and worth stating honestly. Of its five paths, four are architectural
bets that brain-likeness confers safety. **We are not making that bet.** Only path 4,
"interpretability advancement using neuroscience methods", is load-bearing here, and even there
the useful import is *measurement discipline*, not brain structure. The white paper itself notes
it emphasises neuroscience-driven approaches over benchmarking; this fills that gap without
requiring its central premise.

The connection to safety needs no analogy at all: **safety cases rest on evaluations.** If a
score reflects a harness, contamination or test-time compute rather than the capability it names,
the safety case built on it is unfounded. Evaluation integrity is upstream of every safety
argument, and attribution is upstream of evaluation integrity.

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

**1. A reusable attribution harness — the thing that does not exist.** ARC's decomposition took
the field several years and several groups. Deliverable: a public tool that performs the same
decomposition on *any* benchmark — model swap, harness swap, test-time budget, at matched
cost-per-task — so the next benchmark does not need its own multi-year effort. ARC Prize already
publishes cost-per-task and separates base / reasoning / compute-capped tracks; that apparatus
should be portable and is not.

**2. Sample self-agreement, which is genuinely unmeasured.** How often do independent samples
from the same model produce *identical* structured outputs? H-ARC defines the metric and
explicitly declines to compute it for models. It matters because it determines whether voting is
even a coherent selection rule — and the indirect evidence points the counterintuitive way:
models agree with themselves far more than assumed, so voting selects confident wrong answers
rather than aggregating diverse ones. We are already measuring this (a pilot shows samples
converging on a single wrong grid roughly 6 times in 16). Cheap, and nobody has published it.

**3. Re-analysis of artifacts already public.** ARC Prize releases per-task pass/fail with costs
for **77 models × 400 tasks**. Much of the model-axis half of any attribution is sitting there
unanalysed. First deliverable, and it costs no GPU time.

**4. The reproduction group at the hub**, which produces the implementations the above run on.

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
