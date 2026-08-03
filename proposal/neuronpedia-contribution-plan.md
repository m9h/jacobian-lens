# Contributing to Neuronpedia — a concrete plan

**An explicit goal, not an aspiration.** Neuronpedia is MIT-licensed, actively developed
(github.com/hijohnnylin/neuronpedia, ~1.1k stars, pushed within the last month), run by Johnny
Lin at Decode Research, and funded by Open Philanthropy, LTFF, AISTOF, Anthropic and Manifund.
It is the field's de-facto public interpretability substrate.

Every gap below was **verified by inspection**, not inferred — file listings, export schemas,
API probes and their own documentation. Each is something they either lack or have publicly
asked for.

## What they have asked for

Their J-lens blog post carries an open call: *"We're taking contributions for more pre-fitted
J-lenses that we don't have, especially larger models and different model families."*

## Five contributions, ranked by fit

### 1 ★★ Per-feature steering controls — the largest verified gap
Their per-feature export schema contains **no steering measurement at all**. `vectorDefaultSteerStrength`
is a UI slider default; `neg_str`/`neg_values` are down-weighted *logits*, not a negation control.
Nor does the gap close elsewhere: **SAEBench has no steering eval**, **AxBench** (the closest
thing, 500 concepts) has zero occurrences of "random direction", "matched norm" or "null", and
**Gemma Scope runs no steering experiments** and lists comparing SAE steering to steering vectors
as an *open problem in its own paper*.

**Contribution:** per-feature steering effect scored against a **random-direction matched-norm
null** and the feature's **negation**. We have this working (rung 5 of the curriculum: feature
+6.5, random null +1.0, negation −13.5) and validated at scale in the tri-lens Phase 2 result.
This is the "control layer" thesis applied directly to their substrate.

*Honest framing:* individual papers use random-direction nulls ad hoc. Nobody publishes them as
reusable per-feature data for the major SAE releases.

### 2 ★ J-lenses for a post-training ladder — an axis they do not have
Their `neuronpedia/jacobian-lens` repo holds 38 models including `olmo-3-1025-7b` and
`olmo-3-1125-32b` — **base models only, on wikitext**. We hold 11 lenses spanning the OLMo-3
post-training ladder (SFT / DPO / final for both Instruct and Think, plus four RL-Zero arms).
That is the axis their `run-all-fit-lens.py` has no notion of, since it maps one np_model_id to
one hf_model_name.

**Contribution:** the ladder lenses, re-laid-out to `<np_model_id>/jlens/<dataset>/<slug>_jacobian_lens.pt`
with a `config.yaml`. Checkpoint keys are stock `jlens` (`J`, `d_model`, `source_layers`,
`n_prompts`), verified to match their loader.

**Design contribution required alongside it:** their namespace is one-lens-per-model, so eleven
variants of one base model need a naming convention that does not exist yet. Worth proposing
rather than working around.

### 3 The high-sparsity MoE regime, which is empty
Of their 38 J-lens models, **exactly one is a mixture-of-experts** (`gpt-oss-20b`, 4/32 = 12.5%).
Nothing below that sparsity. We have measured the regime (Qwen3-30B-A3B at 6.2%; published
traces for Inkling at 2.3%) and found dispersion varies more by *model family* than by sparsity —
useful precisely because it tells users when a cross-model lens comparison is meaningless.

### 4 ★ A per-layer refit floor for their own lenses
Produced *by* comparing our fit to theirs: two independent fits of the same model, same corpus,
same estimator agree to **cosine 0.9998 at layer 30 but only 0.884 at layer 0**. Anyone using
their lenses to compare models needs this — it says how large a difference must be before it is
distinguishable from fitting noise, per layer. It cost us nothing beyond the comparison and it
makes their artifact more useful.

### 5 Documentation for the J-lens, which has none
`docs.neuronpedia.org` returns **zero hits** for "jacobian", "jlens" or "lens". Their own docs
homepage admits it is stale (last substantive revision ~September 2024) and lists probes,
concepts, transcoders, circuit tracing, Natural Language Autoencoders, the Assistant Axis and
the Jacobian Lens as shipped-but-undocumented. There is no tutorial track — ten reference pages.

We have now used the J-lens deeply enough to write the missing page: what it is, how it differs
from logit and tuned lenses, how to read a convergence trace, and which parameters are
load-bearing.

## How contributions actually land (checked)

There is **no lens upload API** — only `/api/lens/prompt` and `/api/lens/share`, and the latter
stores a *run*, not a lens. Two routes:

1. **PR to `huggingface.co/neuronpedia/jacobian-lens`** — matches their open call.
2. **Self-host their inference server** with `JLENS_HF_REPO` / `JLENS_HF_PATH` pointed at our
   repo — works today with no changes to their code, and is the way to demo before proposing.

For SAE-side contributions their SAE upload flow exists (`/api/model/new`, `/api/source-set/new`,
`/api/features/upload-batch`), so steering-control data has a plausible ingestion path, though it
would likely need a new field.

## Sequencing

**Open with (4)** — the per-layer refit floor. It is small, already computed, immediately useful
to every consumer of their lenses, and it establishes that we are contributing measurements
rather than asking for hosting. **Then (1)**, the steering controls, which is the substantial one.
(2) and (3) follow once the naming-convention question has an answer, and (5) is a natural
by-product of doing any of them.

Contact routes: the J-lens work is an Anthropic collaboration and their UI hardcodes
`jacklindsey@anthropic.com`; Neuronpedia itself runs an `#neuronpedia` channel in the Open Source
Mechanistic Interpretability Slack.
