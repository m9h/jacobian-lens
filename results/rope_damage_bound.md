# Bounding the RoPE damage, for free: it sits inside the refit floor

**2026-09-22.** Before spending GPU hours on a refit, we measured what the bug actually cost —
using an artifact that did not exist a week ago.

## The measurement

Neuronpedia **refit their OLMo-3-7B lens on 2026-09-21** under transformers 5.17.0, i.e. the
corrected RoPE ([issue #235](https://github.com/hijohnnylin/neuronpedia/issues/235), closed). That
gives us a public lens for the same base model on the **correct** forward. Ours encodes the buggy
one. So `cos(ours_buggy, theirs_corrected)`, per layer, bounds the damage from above — it contains
the RoPE effect *plus* sampling noise *plus* any protocol difference.

```
OURS   : n_prompts=616, source_layers=[0,3,6,...,30]   (11 layers, subsampled)
THEIRS : n_prompts=568, source_layers=[0..30]          (31 layers, all)
```

| layer | 0 | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **published refit floor** | 0.884 | 0.899 | 0.944 | 0.956 | 0.976 | 0.986 | 0.993 | 0.996 | 0.997 | 0.999 | 1.000 |
| **ours(buggy) vs theirs(fixed)** | **0.897** | 0.906 | 0.940 | 0.954 | 0.977 | 0.986 | 0.991 | 0.993 | 0.996 | 0.998 | **1.000** |
| difference | +0.013 | +0.007 | −0.004 | −0.002 | +0.001 | +0.000 | −0.002 | −0.003 | −0.001 | −0.001 | −0.000 |

Mean cos **0.967**; worst layer **L0 at 0.897**.

## What this means

**The buggy lens and the corrected lens differ by about as much as two honest refits of the same
model differ from each other.** Every layer lands within ±0.013 of the refit floor, and most within
±0.003. At L0 the old-vs-corrected distance is *smaller* than the floor — which mostly tells you the
floor estimate is itself noisy (it is n=1).

So, for our published work:

- **The ladder conclusions survive.** Our headline signal is `cos(base, instruct) = 0.69`, against a
  floor of 0.884–1.000. A 0.69 sits **far** below both the floor and this old-vs-corrected distance.
  Whatever the bug did, it is an order of magnitude smaller than the effect we reported.
- **The refit is still correct to do, but it is not urgent**, and it does not invalidate the
  qualitative picture. This matches the reporter's own estimate (per-prompt readout correlations
  0.96–0.99, layer rank statistics shifting by up to 0.06).
- **The per-layer floor still has to be recomputed**, because its baseline — Neuronpedia's *old*
  lens — no longer exists. That is a separate defect from the RoPE bug and is not addressed by this
  measurement.

## Independent confirmation of the reporter's number

@venvoo predicted the corrected forward would put identity distance **"4 to 6% above"** the old
convergence curve. Neuronpedia's corrected lens converges at `identity_distance = 0.232001`; our
verified pre-fix anchor was **0.220851**. That is **+5.05%** — inside the predicted band, from an
artifact neither of us had when the prediction was made.

## ⚠️ What this measurement is not

- **Not a clean isolation of the RoPE effect.** The two fits differ in prompt count (616 vs 568),
  layer subsampling, and corpus sampling. It is an upper bound on the total discrepancy, not an
  estimate of the bug alone.
- **Not a check of the downstream readouts.** Cosine between Jacobians is not the same as agreement
  between what the lenses *say*. The reporter measured that separately and got 0.96–0.99.
- **n = 1 again.** One lens pair, one model. The same weakness as the floor it is compared against.

## Consequence for the refit

Cost, from Neuronpedia's published convergence log: **568 prompts × ~24 s ≈ 3.8 GPU-hours per
lens** on an RTX PRO 6000 Blackwell, so ~42 GPU-hours for all 11. That is a real spend against a
project total of ~$250 to date, and this measurement says it buys correctness rather than a changed
conclusion. **Recommendation: refit the base lens first** — it is the one that anchors the floor
and the only one with a public corrected counterpart to check against — then decide on the
remaining ten.
