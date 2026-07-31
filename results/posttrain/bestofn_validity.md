# The covert workspace signal does NOT improve best-of-N selection

**2026-07-31** · `allenai/Olmo-3-1025-7B`, layer 18 · 250 TriviaQA questions × 8 samples
(T=0.8) · cross-validated difference-in-means probe, split **by question** so no leakage ·
Modal H100.

## Why this was run

Our metacognition result is a *description*: the base model's workspace predicts its own errors
at AUROC 0.69, beyond output confidence. RewardBench 2's lesson is that a benchmark must prove
it predicts a downstream outcome — and it validated itself partly via best-of-N. This is the
same test applied to our signal: **can it pick better answers than the model's own confidence?**

## Result: no

| ranker | best-of-8 accuracy | headroom captured |
|---|---|---|
| random (floor) | 0.344 | — |
| mean logprob (baseline) | 0.356 | 3.7% |
| **workspace probe** | **0.364** | **6.3%** |
| oracle (ceiling) | 0.656 | — |

Gain over the baseline is **+0.008** — noise at n=250. And per-sample discrimination is
*worse* than the baseline: probe AUROC **0.556** vs logprob **0.584**.

Both rankers are close to useless here: the oracle sits at 0.656 against a 0.344 floor, so
~31 points of headroom exist and neither ranker captures more than 6% of it.

## The interpretation that matters

**Question-level and sample-level metacognition are different tasks, and our signal only has
the first.** Our AUROC 0.69 asks *"is this model likely to be wrong about this question?"* —
discriminating across questions. Best-of-N asks *"which of these 8 attempts at the same
question is right?"* — discriminating within a question. The covert signal does the first and
not the second.

That is a real bound on the practical claim. "The workspace covertly encodes whether its own
answer is wrong" remains supported; **"and this is useful for selecting answers" is not.** We
should not have assumed the second followed from the first, and the Scorecard should record the
distinction rather than let a metacognition cell imply utility it has not demonstrated.

## Limits, and one refinement that could change it

- **n = 250 questions**; +0.008 is well inside noise. A larger run would tighten the bound but
  is unlikely to reverse a difference this small.
- **TriviaQA contamination.** Heavily represented in pretraining — exactly the reuse problem
  RewardBench 2 avoided by commissioning new prompts. Affects both rankers equally, so the
  *comparison* survives, but absolute numbers should not be read as capability.
- ★ **The probe may be fitting the wrong thing.** It was fit on all samples pooled, so a
  difference-in-means between correct and incorrect samples partly captures *question
  difficulty* rather than *sample correctness*. Centring residuals within question before
  fitting would isolate the within-question signal, which is what best-of-N actually needs.
  **That is the honest next test, not an excuse** — the pooled probe is what our AUROC 0.69
  claim was built on, and at the sample level it underperforms logprob.

## Consequence for the proposal

The metacognitive cells cannot yet claim downstream validity in RewardBench 2's sense. The
remaining validation routes that do not depend on within-question ranking are **calibration**
(ECE/Brier, already measured at 0.113) and **selective prediction** (risk–coverage / AURC),
both of which are question-level and therefore matched to the signal we actually have. Those
should be run before any cell is described as reliability-relevant.
