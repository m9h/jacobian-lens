# Erratum: OLMo-3 lenses were fitted with the wrong RoPE on 24 of 32 layers

**Status: confirmed, unresolved. Refit pending.**
**Reported by [@venvoo](https://huggingface.co/mhough/olmo3-jacobian-lenses/discussions/1)
(Wenbin Wu), 2026-09-05.** First external check of these artifacts by someone outside the project,
and it found a real error.

## The error

Our `uv.lock` resolves **transformers 5.9.0** (verified). In that range the OLMo-3 modeling code
applied the config's YaRN `rope_scaling` to **every** attention layer, a regression introduced by
[huggingface/transformers#39847](https://github.com/huggingface/transformers/pull/39847) (*"[v5]
Refactor RoPE for layer types"*) and fixed in
[#46911](https://github.com/huggingface/transformers/pull/46911) (*"[Olmo3] different RoPE per
layer type"*), released in **v5.13.0**. Correct behaviour — per the OLMo 3 paper §3.6.4 and
OLMo-core's exporter — applies YaRN to **full-attention layers only**.

`OLMo-3-1025-7B/config.json` (checked directly):

```
num_hidden_layers : 32
layer_types       : 24 sliding_attention, 8 full_attention   (every 4th layer is full)
rope_scaling      : yarn, factor 8.0, attention_factor 1.2079, original_max_pos 8192
```

**So 24 of 32 layers — 75% of the model — ran a positional encoding the model was not trained
with.** Our 11 lenses were fitted 2026-07-22 → 08-02, after the fix shipped (2026-07-03) but
against a lock that predates it.

## Measured impact

From the reporter, reproducing Neuronpedia's `olmo-3-1025-7b` lens on its exact prompt set:

- under transformers 5.16.1, running-mean identity distance sits **4–6% above** Neuronpedia's
  `convergence.csv` at every *n*;
- forcing YaRN onto the sliding layers to emulate the old behaviour reproduces their curve to
  **within 2e-4 on all 20 rows** tested.

On readouts through the old lenses: per-prompt scores correlate **0.96–0.99** across the two
forwards; layer-level rank statistics shift by up to **0.06**.

## Why the impact is modest despite affecting 75% of layers — and the part that is not

Our fits used `max_seq_len 128`. YaRN's *frequency* interpolation is position-scaled, so at m ≤ 128
the angular difference is small, which plausibly accounts for the high per-prompt correlation.

**But YaRN also applies `attention_factor: 1.2079`, a position-independent softmax temperature.**
On 24 layers that should not have it, that applies at every position regardless of sequence length.
This is *not* a long-context-only artifact, and it is our best current guess at the residual 4–6%.
Untested — flagged as a hypothesis, not a finding.

## What this does and does not touch

| result | status |
|---|---|
| **Per-layer refit floor** (`perlayer_floor_correction.md`) | valid as a **sampling** floor, measured on the wrong forward |
| **Neuronpedia cross-validation** (`neuronpedia_crossvalidation.md`) | ⚠️ **cannot have caught this** — both lenses share the convention. See below |
| **Post-training ladder** (method ratio, capability-flat, early-layer concentration) | numbers will move; qualitative picture likely survives, per the reporter. **Not re-verified by us.** |
| **Adebayo randomization control** (random blocks read out nothing) | unaffected in kind — a null on scrambled weights does not depend on the RoPE convention |

## The methodological lesson, which is the expensive part

`neuronpedia_crossvalidation.md` calls itself *"the first external check"* of our pipeline. It was
not. **Both lenses were produced by the same library version with the same wrong convention**, so
the comparison could only ever have measured sampling noise. Agreement between two artifacts is
weak evidence when the same tooling produced both.

Recorded as [PITFALLS #26](https://github.com/m9h/spinning-up-in-mech-interp/blob/master/PITFALLS.md):
*a comparison between two implementations tests correctness only if the implementations differ in
the way that could be wrong.* What would have caught it is a cross-**version** check, or validation
against the model's training-time definition — not another consumer of the same library.

## Actions

- [x] Confirm the report (lockfile, both PRs, the config's layer types) — all verified
- [ ] Refit the 11 lenses under transformers ≥ 5.13 and republish
- [ ] Re-run the post-training ladder on the refitted lenses and diff every published number
- [ ] Annotate the HF model card and every affected `results/` file until the refit lands
- [ ] Pin `transformers>=5.13` in `pyproject.toml` (currently `>=5.5`) and re-lock
- [ ] Take up the reporter's offer of their comparison script
