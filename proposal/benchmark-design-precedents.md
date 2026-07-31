# Design precedents for the Scorecard — what RewardBench teaches

**2026-07-31.** The Scorecard proposes a controls-first, open-model benchmark with a leaderboard
and a claim registry. AI2 has already built something of that exact shape, twice, and the
second version exists because the first did not do what people assumed. That is the most
useful evidence available about our chief risk.

## The record

| | RewardBench v1 | RewardBench 2 |
|---|---|---|
| paper | [arXiv 2403.13787](https://arxiv.org/abs/2403.13787), Mar 2024 | [arXiv 2506.01937](https://arxiv.org/abs/2506.01937), Jun 2025 · **ICLR 2026** |
| framing | "the first evaluation tool for reward models" | "advancing reward model evaluation" |
| status | superseded in ~15 months | current |

Authors incl. Nathan Lambert, Valentina Pyatkin, Noah A. Smith, Hannaneh Hajishirzi (AI2).
Apache-2.0, PyPI `rewardbench`, [leaderboard Space](https://huggingface.co/spaces/allenai/reward-bench),
released eval dataset, results dataset, and trained-model collection.

## Why v2 was needed

In the authors' own words, *"progress in evaluation has not been mirrored by the effectiveness
of reward models in downstream tasks."* v1 established evaluation practice and was widely
adopted — and scores on it did not track what practitioners were using them to predict.

**A benchmark can be adopted and still not measure the thing.** Our proposal names this as the
chief risk ("a poorly constructed test is worse than none"). RewardBench turns that from a
caveat into a documented failure mode, from a well-resourced group, on a benchmark far simpler
than ours.

## Four decisions to copy

**1. ★ Make the benchmark prove it predicts something.** RewardBench 2's headline claim is not
that it is harder but that it is *"highly correlated with downstream performance"* — validated
against two independent downstream uses (best-of-N sampling and PPO training).

This is the hardest and most important question for us, and it splits the Scorecard cleanly:

- **Consciousness indicators have no agreed downstream.** Nothing external says whether an
  ignition signature "worked". This half can offer *reproducibility* and *controls*, but it
  cannot offer predictive validity, and we should not pretend otherwise.
- **Metacognitive/reliability properties do.** Calibration error, selective prediction,
  refusal quality, hallucination rate under uncertainty. A workspace error-monitoring AUROC
  that predicts none of those is a number, not an instrument.

That asymmetry is a reason to lead publicly with the reliability framing — not for
palatability, but because it is the half that can be *validated* in RewardBench 2's sense.
**Concrete deliverable:** for each metacognitive cell, report its correlation with at least one
downstream reliability measure, and publish the correlation even when it is weak.

**2. Source new prompts.** v2 commissioned new human prompts rather than reusing prompts from
existing downstream evaluations, explicitly to reduce contamination. Our metacognition work
currently uses TriviaQA; the emergence results use wikitext. Both are heavily trained on. A
Scorecard cell whose prompts appear in pretraining corpora measures memorisation as much as
metacognition.

**3. Ship the apparatus, not just the numbers.** Leaderboard Space + eval dataset + results
dataset + trained models + a pip-installable package. We have the substrate — three HF datasets
— and none of the adoption machinery: no leaderboard, no registry, no package. That gap is the
difference between an artifact and a benchmark.

**4. Build headroom, and version from day one.** Models score ~20 points lower on v2 than v1;
benchmarks saturate. Assume a v2 and design the schema so a v2 does not orphan v1 results.

## A caution this precedent also supplies

RewardBench v1 was *not* a bad benchmark — it was a good one whose validity assumptions went
untested for over a year, in a field with tight feedback loops and an obvious downstream
signal. The Scorecard has neither. Our compensating strength has to be the controls: every cell
carrying an explicit null and a reproduction, so that when a cell turns out not to measure what
we thought, that is visible in the artifact rather than discovered fifteen months later.

## Direct technical overlap

RewardBench evaluates **DPO models via implicit rewards** (`scripts/run_dpo.py`) — the same
post-training stage our J-space result singles out as doing most of the work (SFT+DPO moves the
J-space ~5× more than RLVR). If a J-space measurement on the DPO checkpoints tracked reward-model
quality, that would be exactly the downstream validation point (1) asks for. Untested, and
cheap to test on the OLMo ladder we already have.
