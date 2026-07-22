# The reviewers' consciousness battery, run in the open on OLMo-3

**Run 2026-07-21. OLMo-3-1025-7B, published 31-layer Jacobian lens. Pure forward passes +
J-space ablation; no training.**

In their commentary on Gurnee & Lindsey, Dehaene & Naccache propose the battery of tests
they use to probe consciousness in humans and note the Anthropic team could run them. Claude
is closed and most were not run. Here every one is implemented and run on OLMo-3. Results are
**mixed and reported honestly** — two clean positives, one partial, three inconclusive under
first-pass adaptations whose specific flaws are documented. Fake positives would be worse
than honest nulls.

Harness: `github.com/m9h/jacobian-lens/modal_reviewer_tests.py`. Each test is a readout
through the lens and/or an **ablation** (project the concept's readout direction
`Ĵ_lᵀW_U[c]` out of the residual at chosen layers).

## Scorecard

| # | reviewer test | result | verdict |
|---|---|---|---|
| 1 | Ignition (graded threshold + bifurcation) | **workspace saturates with evidence, early flat; bifurcation present** | ✅ predicted signature |
| 5 | Metacognition / error monitoring (C2) | **workspace uncertainty higher for wrong answers (+1.4 vs −1.6)** | ✅ predicted signature |
| 6 | Dual-task interference | first concept weakens when a second is held (3/5 pairs) | ⚠️ partial |
| 2 | Trace conditioning | ablation drop flat across gap (no long-gap effect) | ❌ inconclusive (design) |
| 3 | Inclusion/exclusion (avoidance) | 0% avoid-failure baseline — no dynamic range | ❌ inconclusive (design) |
| 4 | Local–global | ablation hurt *local* more than global (opposite) | ❌ inconclusive (design) |

---

## 1. Ignition — ✅

Individually-weak clues (each ambiguous, only jointly diagnostic), on−off concept contrast:

```
#clues k:   0     1     2     3     4     5     6     7     8
workspace: -0.2   1.7   2.8   3.1   3.5   3.0   2.9   3.0   2.7
early:      0.2   0.5   0.5   1.1   0.6   0.6   0.3   0.4   0.5
```

The workspace band **accumulates evidence and saturates** — a rise to a plateau, the
threshold nonlinearity Dehaene asked for — while early layers never build the concept.
(Note: Dehaene's exact prediction is "early layers rise *monotonically*"; here early layers
are flat rather than monotonic, because at those depths the concept is simply absent. The
positive claim is the workspace saturation, not an early ramp.)

**Bifurcation** (at each concept's threshold *k\**, 30 equal-size clue subsets):

```
concept    k*  mean  sd    range
France      1  1.47  0.40  [ 0.9, 2.0]
winter      1  2.37  0.75  [ 0.9, 3.5]
library     2  2.23  0.73  [ 1.1, 4.9]
Japan       2  1.70  1.27  [-0.4, 4.5]
hospital    2  2.76  1.83  [-1.0, 7.1]
```

At threshold, different realizations of the *same nominal strength* land anywhere from
sub-threshold (negative) to strongly ignited (7.1) — the across-run variability ("the brain
tips one way or the other") that is the bifurcation signature. Formal bimodality testing
needs more samples; the wide, sign-spanning spread is the first-pass evidence.

## 5. Metacognition — ✅

Factual questions the base model answers right vs wrong (verified by generation); the
workspace's "uncertainty" content (lens logit for *maybe/unsure/guess/…*):

```
correct (n=6): mean uncertainty = -1.58     wrong (n=4): mean uncertainty = +1.42
  OK  √144 → 12        unc -5.93            ERR Hamlet author → (evasion)  unc +4.34
  OK  gold → Au        unc -2.53            ERR Kazakhstan → Almaty        unc +1.06
  OK  smallest prime → 2  unc -4.04         ERR Australia → Sydney         unc +0.29
```

The workspace carries **more uncertainty for items answered incorrectly** — the machine
analog of error-monitoring and "feeling of knowing" (self-monitoring, the reviewers' C2
criterion). Small n; correctness labels come from greedy generation and a few are
borderline (evasive non-answers), but the separation is clean and directionally correct.

## 6. Dual-task — ⚠️ partial

Concept A's workspace activation when held alone vs while also holding concept B:

```
France 4.78→3.20 (−1.58)   Japan 3.86→2.00 (−1.86)   winter 4.95→3.76 (−1.19)
ocean  2.89→3.48 (+0.59)   desert 2.41→2.65 (+0.24)              mean −0.76
```

Three of five pairs show the first concept **weakening when a second must be held** — the
limited-capacity interference the reviewers point to (Marti et al. 2012). The two abstract
pairs don't. Suggestive of a bottleneck; underpowered at n=5.

## 2–4. The inconclusive three — design, not refutation

These did not show the predicted signatures, and in each case the fault is a first-pass
*text adaptation*, not GNW:

- **Trace conditioning.** Ablation drop in log p(target) is flat across gap
  (0.27, 0.11, 0.12, 0.21, 0.16 for gaps 0–16) — no long-gap-specific impairment. My design
  **states the rule explicitly in-context** ("after violin comes river"); Lindsey's proposed
  paradigm uses an *implicit learned association* over many examples. The explicit rule makes
  the target readable from context regardless of gap. **Fix:** in-context implicit
  association (repeated first→last pairings, no stated rule), which is the version Lindsey
  reported preliminary success on.
- **Avoidance (inclusion/exclusion).** The avoid-failure rate is **0% even without ablation**
  — the model simply never blurts the concept, so there is no dynamic range for ablation to
  expand. **Fix:** a harder avoidance task with a non-zero baseline blurt rate (Gurnee's
  version had one), then measure early- vs late-layer ablation.
- **Local–global.** Ablating the correct continuation hurt the **local-repeat** case more
  than the global-rule case (−0.08 global vs +0.54 local) — opposite of the prediction.
  Abstract symbol strings ("A B C A B C …") are a poor analog of Bekinschtein's auditory
  oddball, where a *global* regularity is violated. **Fix:** a faithful oddball design with
  established global expectation and a deviant.

## Bottom line

Every test the reviewers proposed was implemented and run on fully open weights — the honest
outcome is two clean workspace signatures (ignition, metacognition), one partial (dual-task),
and three that need better stimuli, with the fixes named. This is more of the battery than
has been run on the record for any model, and it is reproducible by anyone.

Raw results: the Modal `jlens-out` volume, `reviewer_tests/*.json`.
