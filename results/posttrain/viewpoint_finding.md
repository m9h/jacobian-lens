> ## ⚠️ DEPTH-PROFILE CAVEAT (2026-10-22)
> "Early layers move most" is an absolute-movement statement (‖ΔJ‖² is largest at L0–9 because
> ‖J − I‖ is largest there). Relative to the off-identity part of the transport, the base→Instruct
> change is *smallest* at L0 (off-identity cos 0.84 R-lens / 0.67 J-lens) and *largest* at L18–24
> (0.72–0.74 under both estimators), and the Instruct lens sits 15–40% further from the identity
> than the base lens at L15–27. Brauer's final-block objection was also checked: 16–23% of the L0
> movement lies in the last block's top output directions; projecting them out leaves cos 0.29.
> See `results/final_block_control/final_block_control.md`.

> ## ⚠️ ESTIMATOR CAVEAT (2026-10-15)
> The numbers below are J-lens numbers. Refit under the R-lens (LRP backward pass, same prompts, same
> protocol) gives cos(base, Instruct) **0.784** instead of 0.689, and the early-layer concentration
> drops from 4.9× to 3.5× (early/late mean 1−cos); the J-lens is noisier on the Instruct model than on
> base at layer 0 (cross-estimator cos 0.34 vs 0.66). The claim survives, smaller and flatter. See
> `estimator_control_rlens.md`.
>
> ## ✅ ERRATUM RETRACTED FOR THESE LENSES (2026-10-15)
> The banner below was wrong about **our** lenses. They were fitted on Modal under transformers
> **5.14.1**, which already has the per-layer RoPE fix; the 5.9.0 figure came from `uv.lock`, which the
> Modal image never used (verified by probing the cached image). The numbers here were never affected
> by the bug. Neuronpedia's June lens was. See ROPE_ERRATUM.md §"Retraction, 2026-10-15". The
> original banner is kept for the record:
>
> ## ⚠️ ERRATUM (2026-09-05)
> These results were computed from lenses fitted under **transformers 5.9.0**, which applied YaRN
> `rope_scaling` to **all 32 OLMo-3 layers** instead of the 8 full-attention ones — so **24 of 32
> layers ran the wrong RoPE**. Numbers here will move on refit. See
> [ROPE_ERRATUM.md](../ROPE_ERRATUM.md). Reported externally by @venvoo.

# Claim 6 (post-training shapes the J-space "toward a point of view") — free scoping

Run on Anthropic's published base/-it lens pairs, no fitting, no GPU. Two measures.

## 1. Magnitude — post-training DOES move the J-space (holds)
`mean cos(base, instruct) = 0.76` over 8 pairs, vs a ~0.96 same-model refit floor. The
workspace moves well beyond fitting noise. First outside quantification of the claim.
But cosine cannot say whether the move is a structured viewpoint shift or diffuse drift.

## 2. Structure — the low-rank result is CONFOUNDED (does not hold as stated)
Effective rank of dJ = J_instruct − J_base looked strongly structured (rank fraction
0.003–0.02, vs 0.50 for iid drift). **But J itself is already low-rank**, and for 2 of 3
models dJ has essentially the same rank as J:

    model          rank(J_base)   rank(dJ)   ratio
    gemma-3-270m       0.048       0.022     0.45x
    gemma-3-1b         0.019       0.019     0.98x
    gemma-2-2b         0.119       0.003     0.02x

So dJ's low rank is mostly inherited, not evidence of a structured shift. Only gemma-2-2b
shows a genuinely concentrated change. **The naive effective-rank result must not be
reported alone.**

## 3. Content — the workspace holds discourse markers over output (separate, holds)
Norm-corrected (dividing out embedding norm via the motor band), OLMo-3-7B's workspace
preferentially expresses `instead / either / similarly / than / themselves` (discourse,
contrastive, reflexive) while the motor layers push `the / that / for` (function words,
literal next tokens). Un-confounded, but n=1, hand-picked layers, no statistics.

## What this means for the $360 OLMo ladder
Do NOT fit it to measure effective rank — that measure is confounded and the confound is
free to see. If the ladder is worth anything, it is for the measures that NEED training
data or checkpoints (feature→data attribution; workspace formation over training), not for
a cosine or a rank the published lenses already answer. The single well-motivated cheap
spend is the refit-noise null: fit ONE model twice (~$3) to calibrate measure 2.
