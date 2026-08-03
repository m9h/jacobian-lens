# ARC attribution pilot — design post-mortem

**2026-08-02.** The pilot was designed and launched *before* a literature review. The review
found both axes broken, one of them in a way worse than a null result.

## What was wrong

**1. Model axis had no signal.** Direct-prompted Qwen3-4B scores **0.63%** on the ARC-AGI-1
public evaluation set ([DiARC, 2606.26530](https://arxiv.org/abs/2606.26530)). At n=40 tasks that
is an expected **0.2 solves**; P(zero solves) ≈ 0.82. Reaching p≈0.055 needs a 0-vs-5 split.
Realistic outcome for 4B vs 8B: **1 vs 2**. Restricting to small grids does not rescue it —
MiniARC (5×5) gives Qwen3-4B **0.67%**.

Our own run: Qwen3-4B finished at **4/40 solved at pass@16** (618/640 grids parsed). Higher than
the literature's 0.63% because ARC's *training* set is easier than the evaluation set and pass@16
gives 16 attempts, not the official 2. An early check at 18 tasks read 0/40 and I over-read it as
a hard zero — the floor is low, not absent. It makes no difference to the power problem: 4 vs a
plausible 6 for the 8B arm is Fisher p≈0.7.

**2. ★ Harness axis was backwards.** Majority voting over *temperature* samples is documented as
neutral-to-harmful on ARC. Akyürek et al. state it "is not viable"; the ARChitects measured
stochastic sampling as **worse than greedy** (50.5 vs 51.5 top-2). The k=1→k=16 delta would have
been ≈0, supporting the conclusion **"harness does not matter"** — the opposite of the truth, and
the opposite of our own thesis.

**A null would have been harmless. This would have been a confident false finding.**

What actually works is voting over **augmentation**-induced diversity, not temperature: TRM
29.25%→40.00% with 1000 augmentations; MindsAI AIRV 5→13 tasks; ARChitects 63.5%→71.6% via
product-of-experts over 16 augmentations. Or program synthesis with **execution filtering** —
selection by running candidate programs against the training pairs, which is verifiable rather
than democratic.

**3. Smaller errors:** scored pass@1 majority vote rather than the official **pass@2**;
`temperature=0.8` is precisely the setting the literature avoids for grid output (BARC uses t=0 +
beam for grids, t=0.8 only for *programs*); used 40 of 238 eligible tasks; grid-size filter bound
on output only, so 30×30 inputs were admitted; no chat template on instruct models.

## ★ And the framing was wrong too — 2026 moved the model axis

"Harness dominates, the model barely matters" is true of **small models under a compute cap** and
false of the frontier. Verified July 2026: ARC-AGI-1 **saturated at 98% for $0.52/task against a
$17 human baseline** — 33× cheaper than people; ARC-AGI-2 API **92.5%**, from near-zero at its
March 2025 launch; ARC-AGI-3 **<1% in March 2026 → ~30% by July**.

Generalizing a Kaggle-track result into a claim about capability trends is precisely the
conflation this project exists to correct. The useful fact is that the API and Kaggle tracks run
**identical tasks** and differ by **~68 points** on compute budget and harness alone — the
cleanest published capability-vs-scaffold decomposition in AI evaluation, and unanalysed as one.

## Why running it at all was the wrong call

The decomposition is **already published, cleanly, several times**: Product of Experts shows
+53.3 pts from harness on a frozen 8B against +3.4 pts for a 3B→8B swap; Moghe & Chin show
+51.75 pts from harness alone, replicated across base models. ARC Prize's 2024 report states that
no static transduction solution exceeds 10%.

We set out to demonstrate something demonstrated. The proposal now **cites** it and moves the
contribution to what is actually missing.

## What survives, and one thing worth keeping

- **A measured floor**: direct-prompted Qwen3-4B, 4/40 at pass@16 on easy training tasks — a low
  floor, measured rather than cited, but far too low to amplify.
- **★ An unoccupied measurement we stumbled into.** No published work measures the exact-output
  agreement rate between independent LLM samples on ARC — H-ARC defines the metric and declines
  to compute it for models. Our run has it: **mean top-vote 6.3 of 16 samples**, i.e. models
  converge on a single answer far more than "sampling gives diversity" assumes. That is why
  voting fails here: it aggregates confident agreement on wrong answers. Cheap, unclaimed, and a
  better question than the one we planned.

## The process lesson

Do the literature review **before** designing, not after launching. The empirical check (are
grids parsing? is anything solved?) caught the null within 20 minutes — but only the review
caught that the harness axis pointed the wrong way, and that is the error that would have
survived into a claim.

Filed alongside `PITFALLS.md` #13 (a null with no positive control) as its sibling: **a design
with no literature check can produce a confident result in the wrong direction.**
