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
