# Restating the post-training result as excess over a per-layer refit floor

**2026-08-02.** The published headline — `cos(J_base, J_instruct) = 0.69`, "~31% move" — is a
**mean over 11 layers whose independent-refit floors differ by an order of magnitude.** Pooling
them means the same cosine carries very different weight at either end. This corrects it. No GPU
required: it reads published lenses.

## The floor

Cosine between our lens and Neuronpedia's **independently fitted** lens for the same model, same
corpus, same estimator — i.e. how far apart two honest fits already sit:

| layer | 0 | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| floor | 0.884 | 0.899 | 0.944 | 0.956 | 0.976 | 0.986 | 0.993 | 0.996 | 0.997 | 0.999 | 1.000 |

## The corrected result

`excess = (floor_l − cos_l) / floor_l` — movement *beyond* what two fits of the same model
already differ by, computed per layer.

| arm | pooled cos | pooled move (published) | excess, mean | early L0–9 | late L21–30 |
|---|---|---|---|---|---|
| **Instruct** | 0.689 | 31.1% | **29.4%** | **49.9%** | 10.6% |
| **Think** | 0.731 | 26.9% | **25.1%** | **44.6%** | 7.8% |
| RL-Zero math | 0.939 | 6.1% | **2.9%** | 3.1% | 1.7% |
| RL-Zero code | 0.939 | 6.1% | **2.8%** | 3.0% | 1.6% |

## Three consequences

**1. The core claim survives and strengthens.** The method ratio — instruction/CoT tuning versus
RLVR — was published as **~5×** (31.1 / 6.1). Corrected for the floor it is **~10×**
(29.4 / 2.9). Accounting for measurement noise makes the effect *larger*, because RLVR's apparent
movement was mostly noise and Instruct's was not.

**2. RLVR's effect halves, and should be restated.** "RLVR moves the J-space ~6%" becomes
**~3% beyond the refit floor**. Roughly half the published RLVR number was two fits disagreeing
with each other. Any claim resting on the 6% figure needs updating — including our own
comparisons against the new agentic ladder, where the RL step measured 2.5% at a *reduced* budget
and therefore also needs its own floor before the two are compared.

**3. ★ The movement is early-layer concentrated, and this is new.** Instruct moves **49.9%**
beyond floor in layers 0–9 but only **10.6%** in layers 21–30. The pooled statistic hides this
entirely. Note the direction: early layers have the *worst* floor (0.884), so this is a large
effect surviving the noisiest region, not an artifact of it. What post-training changes most is
what the *early* residual stream is disposed to say.

## Limits

- **The floor is n = 1.** It is one comparison — our fit against Neuronpedia's — not a
  distribution over refits. A second independent fit would turn it into an interval. Until then,
  treat "excess over floor" as a first-order correction, not a significance test.
- Floors are measured on the **base** model. If post-trained arms have systematically noisier
  fits, their floors would differ, and we have not measured that.
- The early/late split (L0–9, L21–30) is a post-hoc partition chosen for readability, not a
  pre-registered contrast.

## Reproduce

`analyze_perlayer_floor.py` — downloads published lenses, no GPU, runs in minutes.
