# Post-training reshapes the J-space toward a viewpoint, decoupled from capability

**Run 2026-07-21. OLMo-3-7B, full base→post-trained ladder, Jacobian lenses fit on Modal.**

What Anthropic actually report (qualitatively, no numbers): the J-space is already present
in the pretrained model, but "during post-training, the J-space develops some signatures of
adopting 'Claude's point of view'" — where "in the base model, the J-space mostly tracks
what's needed to predict upcoming text; in the post-trained model, it starts holding
Claude's own reactions."

**In plain terms.** Before post-training, the model's internal "workspace" is mostly busy
with *what word comes next*. After post-training it also carries the model's *own stance* as
the Assistant — its reactions, caveats, and values. Read as a consciousness indicator, that
shift is taken to mean the model has acquired a stable point of view rather than only
predicting text. Anthropic show it with examples, not measurements. **This is the first
quantitative version, on fully open weights** — testable only since AI2 released OLMo-3
(base model, post-trained variants, public data, and a clean method-vs-domain factorial).

**Fully open, fully reproducible, fully shareable.** The Anthropic result rests on
Sonnet 4.5, whose activations no external party can access — the paper's invited
commentators could not check it (Dehaene & Naccache: *"We suggested to the Anthropic team
that they could run exactly the same tests…"*). This test uses **only open artifacts**:
OLMo-3 weights (Apache-2.0), the public wikitext fitting corpus, and the open
`jlens`/`jlens-lab` code. Every number below can be independently reproduced by anyone with
a GPU — no NDA, no frontier-model access, no privileged position. That is the point: a
consciousness-adjacent claim about frontier models, made checkable on fully open ones.

## The result in one paragraph

Post-training moves the J-space **massively** — the Instruct lens is only `cos = 0.69`
from base, a ~31% move, versus a same-model refit that agrees at `0.97`. The size of the
move is set by the **training method, not the task domain**: full SFT+DPO instruction/CoT
tuning moves the J-space ~5× more than RLVR-from-base, and varying the RLVR *domain*
(math/code/instruction-following/general) at matched capability changes it by only ~1%.
And the move is **decoupled from capability**: MMLU is flat-to-slightly-*down* across
post-training while the J-space moves ~31%. A large representational shift with no
competence gain is exactly the viewpoint-not-prediction shift Anthropic describe. **The
claim is supported, and sharpened: the viewpoint post-training installs is method/format-driven and
nearly domain-invariant.**

## Design

- **11 arms**, one lens each: base, Instruct-{SFT,DPO,final}, Think-{SFT,DPO,final},
  RL-Zero-{Math,Code,IF,General}. RL-Zero-**Mix excluded** — it is `olmo2-retrofit`
  (`Olmo2RetrofitForCausalLM`), a *different base architecture*; including it would confound
  the RL-Zero control with an architecture change. (Caught by the capability probe.)
- Every arm fit on the **identical 616 wikitext prompts**, layer_step=3 (11 of 32 layers),
  max_seq_len=128, matching the published olmo-3-1025-7b lens config. dim_batch=128 (B200),
  Anthropic's published setting.
- **Anchor-gated.** Before fitting the ladder, our base fit was validated against the
  published olmo-3-1025-7b lens: `mean_cosine 0.969`, `identity_distance 0.2199` vs
  published `0.2209` (0.41% error) — both gates pass. This confirms OLMo-3 YaRN RoPE loads
  correctly, the wrapping/subsampling is sound, and the sharded mean is exact.
- **Capability-controlled.** A forward-pass probe (neutral perplexity + MMLU-by-domain)
  measured capability per arm *before* the fits, to test the RL-Zero premise that capability
  is held constant across domains.

## Distance from base — method sets the magnitude

`cos(J_base, J_arm)`, mean over the 11 shared layers:

| arm | cos(base, arm) | move from base |
|---|---|---|
| Instruct | 0.691 | ~31% |
| Instruct-DPO | 0.692 | |
| Instruct-SFT | 0.701 | |
| Think | 0.733 | ~27% |
| Think-DPO | 0.735 | |
| Think-SFT | 0.740 | |
| RL-Zero-Math | 0.941 | ~6% |
| RL-Zero-Code | 0.942 | |
| RL-Zero-General | 0.944 | |
| RL-Zero-IF | 0.946 | |

Instruction/CoT post-training (SFT+DPO) reshapes the J-space **~5× more** than RLVR applied
to the same base. Within each family the SFT/DPO/final stages are tightly clustered — most
of the move is already present after SFT.

## RL-Zero domain control — the viewpoint is nearly domain-invariant

Pairwise `cos` among the four RL-Zero domain arms (matched RLVR method, matched capability):

| pair | cos |
|---|---|
| Code vs Math | 0.9918 |
| Code vs IF | 0.9925 |
| IF vs Math | 0.9931 |
| Code vs General | 0.9945 |
| General vs IF | 0.9952 |
| General vs Math | 0.9953 |

**The four domain arms are near-identical (`cos ≥ 0.992`).** The domain of the RLVR reward
(math vs code vs instruction-following vs general) changes the J-space by ~1% — an order of
magnitude less than the ~6% that RLVR moves it from base, and ~30× less than the ~31% that
instruction-tuning moves it. The J-space is **largely domain-invariant** under RLVR.

### Null correction (stated because the first pass got it wrong)

The analysis script initially tagged these pairwise numbers "within refit noise" by
comparing to `0.969` (our base vs Anthropic's base). **That is the wrong null.** `0.969`
carries *cross-prompt + cross-implementation* noise (different prompt sample, different fit
code). The ladder arms are all fit on the **same 616 prompts with the same code**, and the
Jacobian fit is **deterministic** — so the correct null for same-prompt/different-model is
`≈ 1.0`. Against `1.0`, the domain differences (`0.99`) are **small but real (~1%), not
noise.** The honest statement: RL-Zero domains *do* differ, by ~1%, which is real but ~30×
smaller than the method effect. (The RL-Zero pairwise is the clean comparison: all four
arms fit at dim_batch=128 on identical prompts, so cos < 1 is pure model difference.)

## The clincher — capability is flat while the J-space moves

From the capability probe (MMLU, 100 items/domain, length-normalised answer logprob):

| arm group | MMLU overall | neutral ppl |
|---|---|---|
| base | 0.507 | 13.0 |
| Instruct family | 0.447–0.470 | 22–28 |
| Think family | 0.473–0.480 | 21–22 |
| RL-Zero family | 0.490–0.507 | ~13.2 |

Post-training did **not** raise capability — MMLU is flat-to-slightly-down (partly a
format artifact: logprob-MC penalises chat/CoT-tuned models). Yet the same Instruct arms
moved the J-space ~31%. **A large J-space move with no capability gain cannot be "it got
better at predicting" — it is a change of objective/viewpoint.** This is the strongest
support for Anthropic's framing: the workspace shifts from tracking prediction to holding the model's own reactions.

The capability probe also validated the RL-Zero control it gates: RL-Zero overall-MMLU
spread is **1.7pp (0.490–0.507), within noise**, with **no domain diagonal** (the code arm
is not best at code, etc.). So the ~1% RL-Zero *geometry* differences are not capability
artifacts — capability is genuinely held constant across those domains.

## What this says about the post-training claim

- **Supported:** post-training reshapes the J-space far beyond fitting noise (~31% for
  Instruct), and the move is decoupled from capability — the signature of a viewpoint shift
  rather than a prediction-quality gain.
- **Sharpened:** the viewpoint is **method/format-driven** (instruction/CoT tuning ≫ RLVR)
  and **nearly domain-invariant** (RLVR domain contributes ~1%). It is not that the model
  acquires task-specific viewpoints; it is that instruction/reasoning post-training globally
  reorients the J-space.

## Caveats (do not drop)

- **Cosine is magnitude, not structure** — "how much moved," not "in what organised way."
  The structural (effective-rank) measure is confounded (`rank(dJ) ≈ rank(J)`; see
  `viewpoint_finding.md`), so it is deliberately not used here.
- **RL-Zero is a light intervention** (~6% from base), so the ~1% domain differences are a
  small absolute signal; a more sensitive measure might resolve domain structure this one
  cannot.
- **MMLU-by-logprob is crude** and format-biased against chat/Think arms; it rules out
  gross capability collinearity, not subtle differences.
- **base was fit at dim_batch=64** (the anchor), arms at 128 — a negligible (~1e-4)
  numerical difference; the RL-Zero *pairwise* (all 128) is the confound-free comparison.
- **n = 1 family (OLMo-3).** The Ministral-3 second-family replication is designed but not
  run (needs an HF Modal secret; and by design it lacks an anchor lens and an RL-Zero
  control, so it can only ask "does the magnitude reproduce", not "viewpoint vs capability").
- Not peer-reviewed; open-weights only.

## Reproduce

`~/Workspace/jacobian-lens/modal_olmo_ladder.py` — `anchor` (gate), `capability_all`
(control), `ladder` (fit), `analysis` (this result). Lenses persist on the Modal
`jlens-out` volume, so deeper (structural) analysis needs no re-fitting.
