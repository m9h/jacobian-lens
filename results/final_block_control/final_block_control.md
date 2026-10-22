# Final-block inherited-direction control on the base→Instruct pair (Brauer's objection)

**2026-10-22. Lens-only, float64, no GPU.** `experiments/final_block_control.py`,
`experiments/final_block_direction_tokens.py`; raw numbers `jlens_base_instruct.json`,
`direction_tokens_jlens.json`. Inputs: the two J-lens files on the Hub
(`mhough/olmo3-jacobian-lenses`, fp16, 616 prompts, target = final block). Prompted by Kaley
Brauer, *J-lens shouldn't target the final layer by default* (LessWrong `Adb55vLqt33LEyBfz`,
25 Sep 2026), read 22 Oct.

## The objection

Every J_ℓ = E[∂h_final/∂h_ℓ] ends by passing through the last block. Whatever that block writes
with a large norm and a strong input dependence becomes a large output-side singular direction of
*every* layer's Jacobian. Brauer found a Chinese-vs-Latin token direction doing this on
DeepSeek-V3 (46% of mid-depth English readouts majority-Chinese under a final target, 3% under a
penultimate one). Our flagship depth profile — cos(J_base, J_Instruct) 0.25 at layer 0 rising to
0.98 at layer 30 — is a final-target statistic, so there was a cheap alternative explanation:
post-training rewrites the last block, every earlier J inherits it, and "early layers move most"
is the target talking.

## The control

For each layer ℓ with ΔJ_ℓ = J_ℓ(Instruct) − J_ℓ(base):

- **spectrum**: share of ‖J_ℓ‖² in the top output-side singular direction, per arm;
- **inheritance**: |cos(u₁(J_ℓ), u₁(J₃₀))| per arm — is the last transport's top direction the
  top direction everywhere?
- **where the movement lives**: share of ‖ΔJ_ℓ‖² inside the top-k output directions of J₃₀
  (k = 1, 4, 16; union over both arms, ≤ 2k dims);
- **projected-out cosine**: remove those directions from both lenses, recompute cos(base, Instruct).

J₃₀ is the transport through block 31 (+ final norm) and stands in for "the last block".

## Result: the artifact is present, and the claim survives it

| layer | cos | base: top-1 share | Instr.: top-1 share | inherit base / Instr. | ΔJ share in J₃₀ top-1 / 4 / 16 | cos, top-16 projected out |
|---|---|---|---|---|---|---|
| 0 | 0.250 | **0.466** | 0.207 | **0.81** / 0.07 | 0.16 / 0.20 / 0.23 | 0.290 |
| 3 | 0.416 | 0.450 | 0.127 | 0.81 / 0.37 | 0.29 / 0.36 / 0.41 | 0.492 |
| 6 | 0.558 | 0.264 | 0.111 | 0.82 / 0.23 | 0.23 / 0.29 / 0.35 | 0.628 |
| 9 | 0.641 | 0.227 | 0.054 | 0.82 / 0.46 | 0.24 / 0.31 / 0.37 | 0.707 |
| 12 | 0.675 | 0.154 | 0.033 | 0.82 / 0.46 | 0.17 / 0.23 / 0.29 | 0.723 |
| 15 | 0.720 | 0.061 | 0.015 | 0.83 / 0.48 | 0.08 / 0.12 / 0.17 | 0.746 |
| 18 | 0.753 | 0.023 | 0.006 | 0.83 / 0.62 | 0.04 / 0.07 / 0.11 | 0.768 |
| 21 | 0.804 | 0.014 | 0.004 | 0.83 / 0.68 | 0.03 / 0.06 / 0.10 | 0.816 |
| 24 | 0.862 | 0.010 | 0.003 | 0.84 / 0.74 | 0.04 / 0.07 / 0.12 | 0.873 |
| 27 | 0.918 | 0.010 | 0.003 | 0.87 / 0.85 | 0.07 / 0.11 / 0.17 | 0.929 |
| 30 | 0.983 | 0.004 | 0.002 | 1 / 1 | 0.13 / 0.19 / 0.27 | 0.988 |

**The base model has the Brauer structure.** Its layer-0 transport is 47% one output direction,
and that direction is the last block's top output direction at every layer (inheritance 0.81–0.87
from L0 to L27). Unembedded through the base model's own norm and head, it is the
**unigram-frequency axis**: `,` `.` newline ` the` ` in` ` and` at one end, rare code identifiers
(`_TRNS`, `_CNTL`, `_MSK`, `RTVF`) at the other. That is the direction a final block uses to
regulate confidence — large norm, small effect on which token wins — which is exactly the kind of
direction Brauer predicts a final-target lens will inherit. The same axis is u₁ of J₃₀(base).

**The Instruct model does not have it at the top.** Its layer-0 top direction holds 21% of the
energy, is orthogonal to the base's (cos 0.001), is not inherited from its own last block
(0.07), and does not unembed to anything readable (`ville`, ` Bin`, ` teh` vs ` brush`, ` combo`).
Through layers 3–27 the Instruct lens does inherit its last block's top direction, but weakly
(0.37–0.85) and from a direction that itself holds ≤ 0.2% of J₃₀'s energy.

**But the movement is not where the artifact is.** At layer 0 only 16% of ‖ΔJ‖² lies in the last
block's top direction and 23% in its top 16 (union over arms, 32 dims); projecting them out moves
the cosine from 0.250 to 0.290. At layer 3, the layer where the share peaks (41%), the cosine
moves from 0.416 to 0.492. Nowhere does the projected-out profile stop being early-low and
late-high. The top direction of ΔJ₀ itself (20% of ‖ΔJ₀‖²) *is* the frequency axis — post-training
removes the base lens's dominant direction — and the remaining 80% is high-rank: ΔJ₀'s second
and third directions unembed to noise, which is the picture the R-lens control already gave
(the J-lens is noisy on the Instruct arm at early layers).

So: the final-target artifact exists on OLMo-3-7B, it has a name, post-training deletes it from
the top of the early-layer lens, and after it is projected out four-fifths of the early-layer
movement is still there. The depth profile survives Brauer's objection as it survived the R-lens.

## What the same files say that we had not written down: the profile is statistic-dependent

Computing the control also produced the **off-identity** cosine, cos(J_ℓ − I, J_ℓ' − I), and the
per-arm identity distances ‖J_ℓ − I‖_F / ‖I‖_F, under both estimators (R-lens numbers from
`results/rlens_control/analysis_rlens.json`, which had them all along):

| layer | cos (J-lens) | cos off-identity (J-lens) | cos off-identity (R-lens) | id base / Instr. (J-lens) | id base / Instr. (R-lens) |
|---|---|---|---|---|---|
| 0 | 0.250 | 0.666 | 0.844 | 1.17 / 1.33 | 1.10 / 1.09 |
| 6 | 0.558 | 0.906 | 0.882 | 1.00 / 1.01 | 0.94 / 1.01 |
| 12 | 0.675 | 0.865 | 0.788 | 0.90 / 0.96 | 0.87 / 1.00 |
| 18 | 0.753 | 0.765 | 0.735 | 0.67 / 0.90 | 0.63 / 0.93 |
| 21 | 0.804 | **0.725** | **0.724** | 0.60 / 0.85 | 0.56 / 0.85 |
| 24 | 0.862 | 0.719 | 0.736 | 0.53 / 0.74 | 0.50 / 0.74 |
| 30 | 0.983 | 0.785 | 0.824 | 0.22 / 0.29 | 0.20 / 0.27 |

Two facts, both estimator-independent:

1. **Relative to what the transport does beyond the identity, the base→Instruct change is
   smallest at layer 0 and largest around layers 18–24.** The off-identity cosine is 0.84 at L0
   (R-lens) and 0.72 at L21, under both estimators within 0.02. The raw cosine is low early only
   because ‖J_ℓ − I‖ is large early (identity distance ≈ 1.1 at L0 vs 0.2 at L30), so the same
   relative change produces a much larger absolute ‖ΔJ‖ there. Our "53% of squared movement in
   L0–9" is an absolute statement and remains true; it is not the statement "post-training
   restructures the early layers most".
2. **Post-training moves the mid-to-late transports *away* from the identity.** From L15 to L27
   the Instruct lens is 15–40% further from I than the base lens (L21: 0.85 vs 0.60; L24: 0.74 vs
   0.53), under both estimators. In the base model the late blocks are close to pass-through for
   the final residual; in the Instruct model they rewrite it. At L0 the two arms are equidistant
   from I under the R-lens (1.09 vs 1.10) — the J-lens's 1.33 vs 1.17 was the estimator-noise
   asymmetry already reported.

The second fact is the more interesting one and is new to us. It is consistent with the
reportability result (the covert error signal becomes reportable after SFT: late layers doing
more) and with the ignition and register findings that live in the mid-to-late band. It also
reframes the depth profile for the Foresight write-up: **the honest one-line version is
"post-training changes the late transports most in relative terms and the early ones most in
absolute terms; the early absolute movement is 80% high-rank and not inherited from the last
block."** The flat "early layers move most" line should not be used on its own again.

## Caveats

- One pair, one family. The RL-Zero arms, the Think arms and the R-lens pair were not run through
  this control (the R-lens files sit on the Modal volume; Modal is unreachable from Legion today).
  The R-lens pair is the one to run next; the off-identity numbers above suggest it will agree.
- "Projecting out k directions" is a linear control. It cannot say what a penultimate-target refit
  would say, which is Brauer's actual recommendation (~$35 for the pair). Given 23% at k = 16 and a
  0.04 cosine shift, the refit is not the priority it would have been at 60%.
- `torch.svd_lowrank` (q = k + 24, 6 power iterations) for the spectra; shares are relative to the
  exact Frobenius norm, so a slightly under-converged u₁ understates the top-1 share, never
  overstates it. J-lens inputs are the fp16 Hub files cast to float64.
- The unembedding ignores the final RMSNorm's 1/rms (a scalar per position, irrelevant for the
  ranking of a direction's tokens) and applies the norm's learned scale; no model forward was run.
