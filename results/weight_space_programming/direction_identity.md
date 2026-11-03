# "Weight-space programming" (Takehi, Teoh, Zhang, Khattab, 2 Nov 2026): what the Qwen3.5-0.8B edit is

*Lens-only check, CPU, $0, 2 Nov 2026. Source: the released edit (`exp8_math/runs/p50c/edit/{manifest.json,edit.npz}`
in `RikiyaT/lm_understand-logs`, CC BY 4.0) applied to the base tensors of `Qwen/Qwen3.5-0.8B` fetched by HTTP range.
Scripts: `experiments/fetch_qwen_tensors.py`, `experiments/wsp_direction_identity.py`, `experiments/wsp_decode_tokens.py`.
Raw numbers: `direction_identity_qwen35_0p8b.json`; unit directions `u_L15.npy`, `u_L19.npy`; edited columns `cols_L*.npy`.*

## The claim as posted

A frontier LLM ("Astra"), given a Python sandbox and 50 forward passes, edited 0.016% of Qwen3.5-0.8B's weights
(two `down_proj` matrices, 64 columns each) and MATH-500 went 7.9% → 48.1%, GSM8K 13.8% → 67.1%, IFEval 21.0% → 62.3%,
share of answers hitting the 32,768-token limit 93.0% → 33.1%. The post frames this as the model "natively" reading and
writing weights. No control appears in the post or the model card.

## What the edit is, from the weights

| | layer 15 | layer 19 |
|---|---|---|
| edited columns of `down_proj` [1024 × 3584] | 64 | 64 |
| rank-1 share of ΔW (top singular value² / Σ) | 0.999995 | 0.999991 |
| best-fit α in W' = W − α·u(uᵀW) on those columns | 6.0003 | 6.0002 |
| fit residual (relative) | 0.0023 | 0.0029 |
| relative change ‖ΔW‖/‖W‖ | 0.106 | 0.081 |
| edited columns among the top-64 by \|uᵀw_c\| in the base weights | 31 / 64 | 18 / 64 |
| share of ‖W‖² along u (uniform baseline 1/1024 = 0.00098) | 0.00124 | 0.00114 |
| \|cos(u, top-5 left singular vectors of base W)\| | 0.10, 0.26, 0.05, 0.06, 0.02 | 0.31, 0.03, 0.06, 0.04, 0.03 |
| share of u in top-1 / 4 / 16 / 64 PCs of the (centred) tied embedding matrix | 0.006 / 0.020 / 0.046 / 0.085 | 0.006 / 0.018 / 0.059 / 0.109 |
| \|cos(u, mean embedding)\| | 0.047 | 0.040 |

cos(u₁₅, u₁₉) = 0.849. The run's own notes (`runs/p50c/readable/notes.md`) state the formula
`W = W0 - 6*outer(u, mask64*(u@W0))` with "unit repetition directions from four matched productive/repetitive original
contexts at residual outputs 16/20; 64 selected MLP writer neurons per layer, chosen on original fwd4". The weights
agree with the notes to three decimals.

So:

1. **It is directional ablation by weight orthogonalisation** (Arditi et al. 2024, the refusal-direction paper:
   W ← W − r̂r̂ᵀW), restricted to 64 neurons, with α = 6 instead of 1. At α = 6 the 64 neurons' component along u is not
   removed but reversed (×−5): the edit turns them into writers of −u, scaled by their own activations. "Removing an
   obstacle" is the post's description; mechanically it is an activation-gated steering vector against the repetition
   direction, baked into the weights.
2. **The direction came from activations, not weights.** It is a contrast of residual-stream states on matched
   productive vs repetitive contexts at the outputs of blocks 15 and 19; the harness returned hidden states and MLP
   activations on teacher-forced texts. u is not a dominant direction of the weight matrices it was applied to (its
   share of ‖W‖² is at the uniform baseline) and only half (L15) or a quarter (L19) of the chosen neurons are the ones
   whose weights write it most; the neurons were chosen by activation-weighted writing on the sampled texts.
3. **u is not a geometric artefact of the unembedding.** Its share in the embedding matrix's top-16 PCs is 3–4× the
   uniform baseline, not the 47% we saw for OLMo's final-block direction; cos with the mean embedding is 0.04. This is
   a different object from the unigram-frequency axis that dominated our OLMo J₀ (`results/final_block_control/`).

## What u reads as (unembedded through the tied head; sign is unidentifiable from the edit since uuᵀ is even)

One pole is wrap-up / resolution vocabulary, English and Chinese alike: ` first`, ` thereby`, ` thus`, ` rather`,
` otherwise`, ` accordingly`, ` appropriately`, ` yields`, ` result`, ` exact`, ` value`, ` numerical`, ` count`,
`即可` ("and that's it"), `呗` (sentence-final "just…"), `而不` ("rather than"), `以求` ("so as to"), `对` ("right").
The other pole is trailing-markdown and dangling-subordinate fragments: `...*`, `...**`, `-**`, `…*`, `而是因为`
("but rather because"), `可怕的是` ("what's scary is"), `她还是`/`他还是` ("she/he still…"), `各种各样的`, plus
multilingual word-piece noise. Read as a direction, it is "close the thought" vs "keep going"; the edit pushes the
64 neurons toward "close". The run's own behaviour table is consistent: of 579 rescued practice problems, 576 were
originally truncated, and accuracy on originally-finished outputs was unchanged.

## What this does and does not establish

- The headline "6× math" is a loop-termination effect on a model whose own card warns it loops. The authors' Gemma
  null (6.8 → 7.7 at best) and their own rescue accounting say the same. Math ability was not added; completion was.
- The post's novelty is the **agentic** part: an LLM chose the method, found the direction, ran a dose-response and a
  norm-matched orthogonal random control (+1.6 pp, SE 3.5, in `notes.md`), and chose 64 neurons, inside 50 passes.
  That control exists only in a 655 KB trajectory; it is not in the post, the README table, or the model card.
- "Reading weights" is not what happened in experiment 8, and where reading weights alone was actually tested
  (experiment 7, sycophancy) it failed 0/4.

## Controls still owed (GPU; ~$40–60 on Modal, blocked while the Spark key is refused)

1. Norm-matched random direction on the same 64 columns, 3 seeds, full MATH-500 protocol (the agent's version was 64 problems).
2. α sweep (1, 2, 6, 10): is α = 1 (true orthogonalisation) enough, or is the reversal doing the work?
3. Budget-forcing baseline: base model, thinking cut at 8k tokens and answer forced. If this reaches ~40%, the edit is a learned stop token.
4. Base at the vendor's recommended 81,920-token cap and in non-thinking mode (the vendor's own IFEval for this model is 52.1 non-thinking / 44.0 thinking vs the post's 21.0 baseline).
5. J-lens of base vs edited: does a 0.016% rank-1 edit that moves MATH-500 by 40 points register in cos(J_base, J_edited) at all? Direct sensitivity calibration for our Claim-6 numbers.
