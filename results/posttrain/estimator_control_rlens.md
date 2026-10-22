> ## ⚠️ DEPTH-PROFILE CAVEAT (2026-10-22)
> "Early layers move most" is an absolute-movement statement (‖ΔJ‖² is largest at L0–9 because
> ‖J − I‖ is largest there). Relative to the off-identity part of the transport, the base→Instruct
> change is *smallest* at L0 (off-identity cos 0.84 R-lens / 0.67 J-lens) and *largest* at L18–24
> (0.72–0.74 under both estimators), and the Instruct lens sits 15–40% further from the identity
> than the base lens at L15–27. Brauer's final-block objection was also checked: 16–23% of the L0
> movement lies in the last block's top output directions; projecting them out leaves cos 0.29.
> See `results/final_block_control/final_block_control.md`.

# Estimator control: the post-training ladder's two endpoints under the R-lens

**2026-10-15.** Script `modal_olmo_rlens.py`; numbers in `results/rlens_control/analysis_rlens.json`
(float64) and `precheck_rlens.json`; lenses on the Modal volume `jlens-out` at
`olmo_rlens/rlens/{olmo-3-1025-7b, olmo-3-7b-instruct}/lens{,_fp32}.pt`, each with a `provenance`
dict. Cost ≈ 6 B200-hours ≈ $35 (estimate from wall time).

## Why

`notes/jpp_lens_2026-10.md` §4: on the same model the J-lens and the LRP-backward lens (R-lens;
J++ without filtering) disagree at early layers more than post-training moves the J-lens. Our
headline — cos(base, Instruct) = 0.689, movement early-layer concentrated — was a claim about what
the J-lens estimator sees until shown under the other estimator.

## What was fitted

The pooled LRP Jacobian through `safety-research/jpp_lens` (commit `93f02cd`, Apache-2.0):
their `LensTrainer` (Anthropic's estimator — one-hot cotangent at every kept position, mean over
kept positions, mean over prompts) with the forward run under `apply_lrp_rules_for_mode("rlens")`:
LN-rule on all 65 `Olmo3RMSNorm`s, identity + half rules on all 32 `Olmo3MLP`s, plain gradient
through attention. No routers, no expert weights, no labelled items. Everything else held to the
ladder's protocol: the same 616 WikiText-103 prompts, layers 0,3,…,30, target = final block,
128 rows per backward pass, max_seq_len 128, skip 16, bf16, B200, transformers 5.14.1 (jpp_lens's
pin; RoPE fix present; same minor version the July J-lenses were fitted under).

**Equivalence check first (precheck, 8 prompts, base model):** the same class with
`lrp_mode="none"` reproduces our pipeline's per-prompt and 8-prompt-mean Jacobians to five digits
(layer-30 identity distance 0.23680 vs 0.23680; layer 0 1.6474 vs 1.6474; prompt 0 0.24290 vs
0.24290). So the only thing that differs between the two columns below is the backward pass.
Timing: 16 s/prompt plain, 14 s/prompt under LRP; peak VRAM 97 GB and 82 GB.

## Result

| layer | cos(base, Instruct) **J-lens** | cos(base, Instruct) **R-lens** | rel. move J | rel. move R | cos(J-lens, R-lens) base | cos(J-lens, R-lens) Instruct |
|---|---|---|---|---|---|---|
| 0 | 0.250 | **0.644** | 1.41 | 0.79 | 0.662 | **0.339** |
| 3 | 0.416 | 0.661 | 1.00 | 0.77 | 0.707 | 0.563 |
| 6 | 0.558 | 0.696 | 0.88 | 0.75 | 0.775 | 0.709 |
| 9 | 0.641 | 0.716 | 0.78 | 0.73 | 0.828 | 0.810 |
| 12 | 0.675 | 0.728 | 0.76 | 0.71 | 0.893 | 0.852 |
| 15 | 0.720 | 0.762 | 0.72 | 0.69 | 0.938 | 0.909 |
| 18 | 0.753 | 0.786 | 0.68 | 0.65 | 0.969 | 0.952 |
| 21 | 0.804 | 0.829 | 0.61 | 0.58 | 0.983 | 0.972 |
| 24 | 0.862 | 0.882 | 0.51 | 0.48 | 0.988 | 0.981 |
| 27 | 0.918 | 0.934 | 0.40 | 0.36 | 0.991 | 0.985 |
| 30 | 0.983 | 0.988 | 0.18 | 0.15 | 0.998 | 0.994 |
| **mean** | **0.689** | **0.784** | | | 0.885 | 0.824 |

| statistic | J-lens | R-lens |
|---|---|---|
| mean (1 − cos), layers 0–9 | 0.534 | 0.321 |
| mean (1 − cos), layers 21–30 | 0.108 | 0.092 |
| early / late ratio | 4.9 | 3.5 |
| share of Σ‖ΔJ‖²_F in L0–9 / L12–18 / L21–30 | 53% / 24% / 23% | 38% / 38% / 24% |
| identity distance at L0, base / Instruct | 1.170 / 1.331 | 1.104 / 1.094 |

## Reading

1. **The magnitude claim survives, smaller.** Post-training moves the lens by 22% under the R-lens
   against 31% under the J-lens (mean cosine 0.784 vs 0.689). At every layer the R-lens sees less
   movement, and the two columns converge from layer 21 down (0.83 vs 0.80 … 0.988 vs 0.983).
2. **The localisation claim survives qualitatively and loses its number.** Early layers still move
   most under the R-lens (cos 0.64–0.72 at L0–9 against 0.83–0.99 at L21–30), but the early/late
   contrast drops from 4.9× to 3.5×, and the early-layer share of squared movement from 53% to 38%
   with the middle layers taking the difference. `perlayer_floor_correction.md`'s "49.9% in L0–9
   vs 10.6% in L21–30" is a J-lens number; under the R-lens the profile is flatter.
3. **Where the J-lens overstated it, and why.** The two estimators agree on the *base* model at
   layer 0 to cos 0.66 but on the *Instruct* model only to 0.34. Under the J-lens the Instruct
   model's layer-0 transport is 14% further from the identity than base's (1.331 vs 1.170); under
   the R-lens the two are equal (1.094 vs 1.104). So part of what the J-lens reported as
   "post-training moved layer 0" was the standard backward pass accumulating more noise on the
   post-trained model than on the base. This is Ayonrinde & Lindsey's "noisy gradients"
   observation, now seen as an *asymmetry across a post-training pair*: the J-lens is least
   trustworthy exactly where our effect was largest, and more so on the arm that moved.
4. **The method-ratio claim is untested here.** The RL-Zero arms (J-lens cos 0.94, ~3% excess
   over floor) were not refit. If the R-lens also shrinks their movement the ~10× SFT-vs-RLVR
   ratio could go either way. Four more arms ≈ $70.

## Caveats

- **No R-lens refit floor.** The J-lens floor (ours vs Neuronpedia's independent fit, 0.884 at L0)
  does not transfer; an R-lens floor needs an independent LRP fit of the same model. The R-lens
  early-layer cosines (0.64) are far below any plausible floor — an LRP estimator should have a
  *tighter* floor than the J-lens, not a looser one — but "excess over floor" cannot be computed.
- One family, two arms, one prompt set, one LRP preset (`rlens`; the J++ filtering step is
  deliberately excluded because its expert weights are supervised and per-model).
- The analysis compares the July J-lenses (transformers 5.14.1) with R-lenses fitted today under
  the same minor version; the 5.14.1→5.19 drift measured on 8 Oct (0.2% at L0) is not in play.
- The cross-estimator cosine on OLMo-3-7B (0.66 at L0 for base) is milder than the Qwen3-4B J++
  number (0.26) in `notes/jpp_lens_2026-10.md` §4. That table compared against a *filtered* J++
  lens; this one is the pooled LRP lens, which is the right comparison for geometry.

## What changes in the published record

- `perlayer_floor_correction.md` and `viewpoint_finding.md` carry an estimator banner pointing
  here. The cos 0.69 / 31% figure should be quoted as "0.69 (J-lens) / 0.78 (R-lens)".
- The Foresight application should cite the R-lens number, not the J-lens one, and can cite this
  file as the worked example of the adjudication function: a published number, a published
  alternative estimator, $35, and the claim is narrower and still standing.
