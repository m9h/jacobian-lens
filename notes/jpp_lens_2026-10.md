# J++ Lens, R-lens, and the Drift Calibrator — what Neuronpedia's 15 Oct post changes for us

**Literature note, 2026-10-15.** Prompted by
[neuronpedia.org/blog/jpp-drift-ui](https://www.neuronpedia.org/blog/jpp-drift-ui) (Johnny Lin
with Sheer Karny, Anthony Baez, Pat Pataranutaporn, MIT Media Lab; 15 Oct 2026). Everything
below was read in full, not from the summary: the J++ post, the R-lens post, the J++ repo README
and CLI, Neuronpedia's `jpp/` fitting scripts, the Drift Calibrator paper PDF, and the lens
files themselves. One cheap measurement is new (§4).

## 1. The post, in two halves

**J++ Lens** (Ayonrinde & Lindsey, LessWrong
[nc9dHfcB22JdMzGbr](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr), code
[safety-research/jpp_lens](https://github.com/safety-research/jpp_lens) Apache-2.0, lenses
[koayon/jpp-lenses](https://huggingface.co/koayon/jpp-lenses)) is a drop-in replacement for the
J-lens: same object (one linear map per layer, no inference cost), different estimator. Neuronpedia
now serves it beside the J-lens on Qwen3.6-27B and has uploaded J++ files into
`neuronpedia/jacobian-lens` under a new `<model>/jpp/` folder (four conversions of Kola Ayonrinde's
released lenses plus their own fits for small Qwen models).

**Drift Calibrator** (MIT Media Lab guest section; paper *Surfacing Neural Activations Over
Multi-Turn Conversation Improves User Calibration to LLM Behavioral Drift*, Karny, Baez,
Pataranutaporn, acmart, 14 Oct 2026) is a chat UI that shows six trait scores per turn, computed
from difference-in-means persona vectors on **Llama-3.1-8B-Instruct at layer 11**, in a
randomised study with N = 246.

## 2. J++ in one paragraph, and the claims that matter to us

Three changes to the J-lens estimator: **(1) Jacobian Filtering** — k-means the layer's source
activations into E = 8 clusters, average one "expert Jacobian" per cluster, combine experts with
learned signed weights fitted on a small labelled dev set of latent-variable items; experts giving
poor readouts get zero or negative weight. **(2) LRP backward pass** (the R-lens of Blank, Bhatia
& Nanda, Aug 2026): LN-rule on the RMSNorms, identity and half rules on the gated MLP, plain
gradient elsewhere. **(3) Readout Filtering** — drop tokens with no letter or digit from the
ranking. Fit on 64 WikiText-103 sequences of 128 tokens, first 16 positions skipped — the same
corpus and skip as our ladder.

Results they report (Qwen3.6-27B, recall@10 of the intermediate variable over seven layers,
five tasks): J-lens 35.7%, R-lens 37.7%, **J++ 55.2%**. Across six models 9B–284B, median uplift
64% over J-lens. Jacobian Filtering alone is 60% of the gain (removing it: −11.7 pp), LRP −6.6 pp,
Readout Filtering −2.9 pp. Causal swap success 59.7% vs 49.3%. Gains are largest at early
layers.

Four of their findings bear directly on our ladder:

- **"Noisy gradients corrupt the averaged Jacobian."** With the standard backward pass the
  per-position gradient norm "explodes towards early layers"; LRP grows 5× slower. They read
  this as accumulated noise, not signal. Our ladder's headline is that post-training movement
  is *early-layer concentrated* (49.9% of movement in L0–9 vs 10.6% in L21–30). That is exactly
  the region they say the J-lens estimator is least trustworthy.
- **Expert Jacobians disagree in early layers and agree in late ones** (their Figure 17): the
  transport map is far from globally linear early. So an averaged Jacobian at an early layer is a
  mixture over disparate local maps, and its value depends on *which activations were visited*.
  Post-training changes the activations the same prompts produce. A base-vs-instruct cosine at
  layer 0 therefore mixes two things: change in the map, and change in the activation
  distribution the average was taken over. Our refit floor controls for prompt sampling, not for
  this.
- **J++ is no longer unsupervised.** The expert weights are fitted on labelled readout items. For
  a cross-model comparison (base vs post-trained arms) that selection step would be re-run per
  arm and could itself differ across arms. The **R-lens** (LRP, pooled, no supervision; also
  what `jpp_cli.py pooled-lens` writes) is the clean drop-in for a ladder.
- **Null results they report:** affine bias term — mixed; single-token Jacobians — "very
  poorly"; using the final block as an extended unembedding — "does not work for us"; gradient
  clipping — helps the J-lens, not J++. Worth knowing before anyone here tries them.

Their stated limitations: single-token readouts only; tokeniser-dependent (no arithmetic eval on
Qwen); unclear ceiling (models can shortcut around the intermediate variable); the k-means is
"relatively coarse". Note also: a reviewer (Stella Biderman) on the R-lens post raises the
general problem that "what the model thinks after layer K" has no ground truth, so every readout
metric can be gamed by reporting layer K+2 — the probe "upper bounds" in the R-lens post are
their only answer and they themselves do not take them seriously.

## 3. Neuronpedia's artifacts changed shape — and our loader was affected

`neuronpedia/jacobian-lens` now has `<np_id>/jlens/…`, `<np_id>/jpp/…`, and for some models
`<np_id>/oracle/…`. The J++ `.pt` has the same `J`/`source_layers`/`d_model`/`n_prompts` keys plus
a `provenance` dict, with the fitting config in `config.yaml` marked "NOT fitted by
run-all-fit-lens.py". jlens-lab's `verify` / `validate_fit` / `published_targets` matched on the
bare `<np_id>/` prefix and took the first `.pt` — listing-order dependent for the four models with
both kinds. Fixed in jlens-lab `d82e9bc` (`_lens_files`, filters on `<np_id>/jlens/…`; test added).
Our anchor model `olmo-3-1025-7b` has no J++ sibling, so no past gate was affected.

A detail worth recording because I got it wrong first: Neuronpedia's own small-model J++ fits
(`qwen3-4b`, `qwen3.5-2b`) carry `num_clusters: 0` and `checkpoint_name: recipe_lens` in their
provenance. That is **not** evidence that Jacobian Filtering was skipped — `recipe_lens` is the
default name of the `fit-weights` stage (the final, filtered lens), and the `num_clusters: 0` field
is also present in the conversions of Ayonrinde's own lenses. The saved file simply does not
record the expert count or weights. It is a provenance gap, not a protocol deviation.

## 4. New measurement: the two estimators disagree more than post-training moves the lens

Cheapest possible check: Neuronpedia publishes both a J-lens (479 prompts, transformers 5.11.0)
and a J++ lens (96 prompts, their fit of the published recipe) for **Qwen3-4B**, 0.46 GB each.
`experiments/compare_lens_files.py`, float64, all 35 layers; table in
`results/jpp_vs_jlens_qwen3-4b.json`.

| layer | cos(J-lens, J++) | rel. Frobenius | id. dist J-lens | id. dist J++ | cos of (J − I) parts |
|---|---|---|---|---|---|
| 0 | **0.260** | 1.33 | 2.32 | 2.76 | 0.36 |
| 4 | 0.467 | 0.89 | 1.61 | 1.20 | 0.66 |
| 8 | 0.569 | 0.82 | 1.49 | 1.11 | 0.74 |
| 16 | 0.327 | 0.98 | 2.11 | 1.25 | 0.38 |
| 24 | 0.786 | 0.63 | 1.13 | 0.69 | 0.53 |
| 28 | 0.757 | 0.79 | 0.89 | 0.78 | 0.04 |
| 30 | 0.632 | 0.87 | 0.75 | 0.83 | **−0.14** |
| 32 | 0.548 | 0.85 | 0.63 | 0.83 | −0.10 |
| 33 | 0.971 | 0.33 | 0.56 | 0.72 | 0.82 |
| 34 | 0.963 | 0.30 | 0.39 | 0.29 | 0.56 |

Transposition is ruled out (cos against the transpose is lower at every layer). Prompt count
cannot explain it: we measured earlier that a 100-prompt under-fit still scores 0.956–0.972
against the converged J-lens of the same model (jlens-lab `artifacts.py`, Qwen3 1.7B/4B/8B).
So on the same model, same corpus, the J-lens and the LRP-filtered estimator give maps whose
flattened cosine is 0.26–0.84 at every layer but the last two, and whose off-identity parts are
*anti-correlated* at layers 29–32.

Our ladder's effect is cos(base, Instruct) = 0.69, averaged over layers. **The estimator choice
moves the lens more than post-training does.** That does not mean the ladder result is wrong — it
is a difference *between arms under one fixed estimator*, and the estimator is deterministic given
the weights. It means the result is, until checked, a statement about what the *J-lens estimator*
sees change under post-training, and the layers where it sees the most change are the layers
where that estimator is dominated by what Ayonrinde & Lindsey call noisy gradients.

Caveats on this table: one model (4B, where the R-lens post reports *no* R-lens advantage for the
smallest dense model); the J++ file is Neuronpedia's fit, not the authors'; 479 vs 96 prompts;
Qwen3-4B's J-lens was fitted under transformers 5.11.0 (Qwen3 has no per-layer RoPE, so the OLMo
bug does not apply). None of these can produce a cosine of 0.26.


**The free check from §5, done the same afternoon (OLMo-3 32B).** Neuronpedia's J-lens
(456 prompts) vs the converted Ayonrinde J++ lens (64 prompts) for `olmo-3-1125-32b`, 63 layers;
table in `results/jpp_vs_jlens_olmo-3-1125-32b.json`:

| layer | cos(J-lens, J++) | id. dist J-lens | id. dist J++ | cos of (J − I) parts |
|---|---|---|---|---|
| 0 | 0.612 | 1.056 | 1.129 | 0.82 |
| 1 | -0.398 | 1.041 | 1.262 | 0.67 |
| 5 | 0.283 | 1.012 | 1.004 | 0.89 |
| 9 | 0.358 | 0.998 | 1.007 | 0.89 |
| 13 | 0.433 | 0.976 | 1.039 | 0.89 |
| 17 | 0.697 | 0.913 | 0.850 | 0.81 |
| 21 | 0.789 | 0.797 | 0.785 | 0.66 |
| 29 | 0.844 | 0.586 | 0.724 | 0.89 |
| 37 | 0.871 | 0.522 | 0.785 | 0.87 |
| 45 | 0.806 | 0.463 | 0.829 | 0.85 |
| 53 | 0.937 | 0.387 | 0.623 | 0.72 |
| 57 | 0.960 | 0.335 | 0.410 | 0.66 |
| 60 | 0.835 | 0.235 | 0.492 | -0.15 |
| 61 | 0.846 | 0.187 | 0.807 | 0.16 |
| 62 | 0.554 | 0.130 | 0.926 | 0.11 |

Same picture on the OLMo family, with one addition: the J++ lens is nowhere near the identity at
the last layers (0.93 at L62 vs 0.13 for the J-lens), which is what signed expert weights do to
the matrix. So "J++ as drop-in" is a readout claim, not a geometry claim, and the apples-to-apples
estimator for a geometry comparison is the **R-lens** (pooled LRP, no reweighting) — which is
exactly the one the §5 control uses. The −0.40 at layer 1 is not a typo.

## 5. What this means for the roadmap

**Done the same evening** — see
`results/posttrain/estimator_control_rlens.md`. Outcome: the second of the three below. cos(base,
Instruct) 0.689 → **0.784** under the R-lens; early/late contrast 4.9× → 3.5×; the J-lens's extra
early-layer movement was partly the standard backward pass being noisier on the post-trained model
(cross-estimator cos at L0: base 0.66, Instruct 0.34). ≈ $35.

*Original plan, kept for the record:*


**New control, ranked above the Huginn experiment:** refit the two endpoints of the ladder —
`Olmo-3-1025-7B` and `Olmo-3-7B-Instruct` — with the **R-lens estimator** (LRP pooled Jacobian,
`jpp_cli.py fit-shard --lrp-mode rlens` then `merge-experts --pooled-lens-out`; no labelled
items, no expert weights), same 616 prompts, same layers, and recompute (a) cos(base, Instruct)
and (b) the per-layer movement profile. Three outcomes, all informative:

- early-layer concentration survives under LRP → the result is about the model, not the
  estimator; write it up as replicated across estimators;
- the cosine survives but the profile flattens → the *magnitude* claim stands, the
  *localisation* claim was an estimator artifact; correct `viewpoint_finding.md`;
- the cosine collapses toward the refit floor → the headline was estimator noise responding to
  weight change. Publish that.

Cost: the J++ paper reports LRP fitting at ~7% over a J-lens fit. Our base refit was ~2 B200-h
(~$12). Two arms ≈ $25–30. Blocker to check first: `jpp_lens` pins `transformers==5.14.1`
(that version has the OLMo-3 RoPE fix, so fine) and its LRP rules were run by the authors on
Olmo-3-32B, which shares the 7B's block structure. The OLMo-3 model adapter in their
`src/jlens/` is the Anthropic one we already use.

**Second, cheaper check — done (above):** the OLMo-3 32B pair shows the same estimator
disagreement, so the Qwen3-4B number is not a small-model peculiarity.

**Not changed:** the Huginn developmental design (roadmap item 1) — but whichever estimator the
ladder control favours should be the one Huginn is run with, so the control goes first.

## 6. Drift Calibrator — a probe-validity reading

The UI work is good HCI and the study is properly randomised (three conditions, two personas,
counterbalanced; d = −0.36 for any visualisation vs control, d = −0.32 for multi-turn vs static).
The part relevant to the Scorecard is how the scores are made, because it is the same structure
as the three probe-validity papers in `lit_review_roadmap_2026-10.md` §2.3:

- Trait vectors are **difference-in-means between responses generated under positive vs negative
  trait system prompts** (prompts written by Claude Sonnet 4.6 at ten intensity levels; responses
  filtered by an LLM judge). The vector therefore separates "the system prompt said to be X" from
  "the system prompt said not to be X" — and "the response is X" only insofar as those coincide
  on the fitting set. That is PITFALLS #27 (perfect aliasing; arXiv 2609.10739) exactly: the
  fitting labels are the prompt condition, and no disagreement rows exist.
- Validation is an R² ≥ 0.90 regression of score against *prompted* intensity, on data generated
  the same way. The paper's own limitations section says so: "the regression-based validation may
  overestimate the linearity".
- The score is read at **the last token of the user's prompt**, i.e. before the model has
  produced the turn it is scoring. The blog calls this "a causal measure of behaviors"; the paper
  does not test causation (no steering, no ablation, no random-direction or shuffled-label
  control on the scores themselves). It is a prediction of disposition, and the study shows
  users calibrate to it better than to text alone — which is a result about the UI, not about
  the probe.

None of this is a criticism of the study's HCI claim. It is a worked example of why a Scorecard
needs the adjudication layer: a dashboard that surfaces "sycophancy 0.7" to end users inherits
every validity problem of the probe behind it, and this one has the aliasing problem by
construction. Good citation for the Foresight proposal's "who consumes an unvalidated probe"
paragraph.

## 7. Read next (found via this pass, not yet read)

- *WorkspaceBench: Evaluating Interpretability Methods for the Global Workspace* (LessWrong
  `Zeg2JztbdhguL48uH`) — multi-token readout evals; the J++ post cites it as the alternative eval.
- *J-lens shouldn't target the final layer by default* (LessWrong `Adb55vLqt33LEyBfz`) — directly
  about our `target_layer = n−1` choice.
- Anne Halsall's comment on the R-lens post: per-prompt margin gradients on Gemma 2 are mutually
  coherent at early layers (median cosine 0.72–0.80 to the population mean) where J-lens readouts
  are noisiest — "there is a stable direction to find there, the standard readout just doesn't
  see it." Unpublished; a lead, not a citation.
- Neuronpedia's `jpp/` README notes GatedDeltaNet models need the torch path (no `fla` kernels)
  for the LRP backward — irrelevant for OLMo, relevant if Qwen3.5/3.6 ever enter a ladder.
