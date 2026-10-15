# Erratum: OLMo-3 lenses were fitted with the wrong RoPE on 24 of 32 layers

**Status (2026-10-15): the bug was real; it never applied to our lenses. See §"Retraction, 2026-10-15" at the end — the sections between were written believing it did.**
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


## Update 2026-09-21: Neuronpedia has refit. Ours is now the only stale one.

Independently filed against them as
[hijohnnylin/neuronpedia#235](https://github.com/hijohnnylin/neuronpedia/issues/235) — *"olmo-3-1025-7b
Jacobian lens was fitted under transformers 5.11.0, which applies YaRN to the wrong layers"* —
**status Closed**. Note their version was **5.11.0** and ours **5.9.0**: different versions, same
bug window, which is further confirmation the report is sound.

Their replacement lens landed **2026-09-21 04:07 UTC**, and their `config.yaml` now records:

```
# Environment (the forward pass this lens encodes depends on these):
#   transformers 5.17.0, torch 2.11.0+cu128, ... jlens_commit 22f0412f...
```

**They have adopted the lesson of PITFALLS #26 in their artifact format** — the environment is now
part of the published lens, because the forward pass it encodes depends on it. We should do the
same.

### ★ This invalidates our refit floor's baseline, not just our lenses

`posttrain/perlayer_floor_correction.md` states plainly: **"The floor is n = 1. It is one
comparison — our fit against Neuronpedia's."** That comparison was against **their old lens, which
no longer exists**. So the published per-layer floor (0.884 at L0 → 1.000 at L30) is now measured
against a withdrawn artifact, on a forward neither party uses.

The upside is large: their full fit protocol is public in that config —
`n_prompts 1000, dim_batch 128, max_seq_len 128, bfloat16, stop_at_delta 0.002`, dataset
`Salesforce/wikitext` (wikitext-103-raw-v1, train). **Refitting to match it gives us a genuine
cross-implementation floor with both sides on the corrected forward** — which is exactly the
cross-version check PITFALLS #26 says would have caught this in the first place.

### ⚠️ Correction to the "refit is cheap now" hope

I relayed Nanda's commentary as showing n=10 suffices. Checked against the paper's §A.7 directly:

> "We observe that J-lens beats the logit lens and tuned lens **baselines** with as few as 10
> prompts, with modest improvements coming from additional data."

That is a claim about **beating baselines**, not about matching an n=1000 lens. **Our measurement is
a difference *between* lenses** (`cos(base, instruct)`, and excess over a refit floor), which is a
stricter requirement than "is this lens better than logit lens" — a noisier lens raises the floor,
and our ladder signals sit close to it at some layers. **Do not refit at n=25 and compare against a
floor measured at n=1000.** Either match Neuronpedia's n=1000 protocol, or measure the floor at
whatever n we choose. Same n on both sides of every comparison.

## Damage bounded (2026-09-22) — the ladder survives

Measured `cos(ours_buggy, theirs_corrected)` against our per-layer refit floor, using Neuronpedia's
21 Sep corrected lens. **Every layer lands within ±0.013 of the floor**, mean cos 0.967, worst layer
L0 at 0.897 against a floor of 0.884. The bug moves our lenses by about as much as two honest refits
disagree. Our ladder signal (`cos(base, instruct) = 0.69`) is far below both, so the published
qualitative conclusions stand. Full numbers and caveats in
[rope_damage_bound.md](rope_damage_bound.md).

This downgrades the refit from *urgent* to *correct to do*, and re-prioritises it: the base lens
first (it anchors the floor and has a public corrected counterpart), then decide on the other ten.

## Actions

- [x] Confirm the report (lockfile, both PRs, the config's layer types) — all verified
- [x] ~~Refit the 11 lenses~~ Base lens refit 2026-10-08: bug effect ≤ 0.2% cosine, 0 at L ≥ 12; the other ten are deliberately not refit (see update)
- [x] Not needed: the measured bug effect (0.002 cos at L0) is two orders below the ladder effect (0.31); documented in the update
- [ ] Annotate the HF model card and every affected `results/` file until the refit lands
- [x] Pin `transformers>=5.13` in `pyproject.toml` — done (modal_olmo_ladder.py still pins >=5.5; the refit script pins >=5.13)
- [ ] Take up the reporter's offer of their comparison script

## Update 2026-10-08: base lens refit under the fix — the bug moved our lens by ≤ 0.2% cosine

Refit the base lens (`allenai/Olmo-3-1025-7B`) on Modal under **transformers 5.19.0 / torch
2.14.1+cu130**, protocol identical to the original fit (616 prompts, same wikitext stream,
layers 0,3,…,30, dim_batch 128, max_seq_len 128, skip_first 16, bf16), written to a separate
volume path so the buggy lens survives as the comparison. Script
[`modal_olmo_refit_tf513.py`](../modal_olmo_refit_tf513.py); environment and every number below in
[`results/rope_refit_tf513/`](rope_refit_tf513/). All statistics recomputed in **float64**
([`modal_refit_check_f64.py`](../modal_refit_check_f64.py)) — see the precision note at the end.

### Buggy vs corrected, same protocol — the direct measurement of the damage

| layer | cos (f64) | rel. Frobenius error |
|---|---|---|
| 0 | 0.99784 | 6.6% |
| 3 | 0.99899 | 4.5% |
| 6 | 0.99935 | 3.6% |
| 9 | 0.99987 | 1.6% |
| 12 | 1.00000 | 0.27% |
| 15–30 | 1.00000 | ≤ 0.07% |

`identity_distance` at layer 30: buggy **0.220148**, corrected **0.220154** — unchanged to five
digits. The wrong YaRN on the 24 sliding-window layers perturbed the **early-layer** transports by a
few percent in Frobenius norm and the deep ones not at all, which is what §"Why the impact is
modest" predicted: at max_seq_len 128 the frequency interpolation is near-identity and only the
position-independent `attention_factor` is live. The ladder's early-layer signal (49.9% of Instruct's
movement in L0–9) sits in the region the bug touched most, and the bug's effect there is 0.002 in
cosine against a ladder effect of 0.31. **The ladder survives by two orders of magnitude**, now
measured rather than bounded (supersedes [`rope_damage_bound.md`](rope_damage_bound.md)).

### Ours vs Neuronpedia's corrected lens — the floor's baseline is back

| layer | buggy ours vs NP-corrected | corrected ours vs NP-corrected |
|---|---|---|
| 0 | 0.89682 | 0.89674 |
| 3 | 0.90632 | 0.90625 |
| 6 | 0.93977 | 0.93980 |
| 9 | 0.95444 | 0.95440 |
| 12 | 0.97725 | 0.97725 |
| 15 | 0.98631 | 0.98631 |
| 18 | 0.99064 | 0.99064 |
| 21 | 0.99334 | 0.99335 |
| 24 | 0.99571 | 0.99571 |
| 27 | 0.99776 | 0.99776 |
| 30 | 0.99964 | 0.99964 |

Identical to four digits whether or not our lens has the bug. So the per-layer refit floor
(0.884 → 1.000 in the original float32 numbers) is **entirely protocol and sampling** — prompt
set, n (616 vs 568), convergence stopping — and none of it was the bug. The anchor gate's cosine
check passes (mean 0.9696 ≥ 0.95).

### ⚠️ New open discrepancy: Neuronpedia's refit moved 5%, ours moved 0.003%

The gate's `identity_distance` check **fails** at 5.19% vs a 5% tolerance: ours 0.22015, theirs
0.232001. Before the fix the published value was **0.220851** — we matched it to 0.3%. Same model,
same fix, and their layer-30 statistic moved **+5.0%** while ours moved **+0.003%**. The fix cannot
be the cause: at layer 30 it changes our transport by 0.01% in Frobenius norm. Something *else*
differs between their old and new fits — prompt sampling, the convergence stop (568 vs their
original count), or a second behavioural change between transformers 5.x versions (attention
backend, sliding-window handling). **Unresolved.** It is not a threat to our ladder (which never
used their lens) but it is exactly the kind of thing PITFALLS #26 says to chase: two refits of the
same model under "the same fix" should not disagree by 5% at the layer the fix does not touch.
Worth raising with Neuronpedia/@venvoo with these numbers; not yet done.

### Precision note — float32 cosine exceeds 1.0 here, and our published floor used it

`torch.nn.functional.cosine_similarity` in float32 on these 4096² ≈ 16.7M-element,
identity-dominated vectors returns **1.0059** at layer 30 for two lenses whose float64 cosine is
1.00000, and inflates every deep-layer value by +0.002 to +0.006 (full table in `check_f64.json`,
column `cos32_repro`). `jlens_lab.artifacts.compare` had this; so did the comparison in this
refit script; so did the floor numbers published in `perlayer_floor_correction.md`. The shape of
the floor is unaffected (early layers are barely touched: 0.898 vs 0.897), but any deep-layer
"cos 1.000" from the float32 path should be read as 0.9996. Fixed in jlens-lab (float64 in
`compare` and `identity_distance`); added as
[PITFALLS #28](https://github.com/m9h/spinning-up-in-mech-interp/blob/master/PITFALLS.md).

### Cost and what remains

One B200 fan-out (8 shards) plus a CPU check: ~23 min wall including image build, ≈ 2 GPU-h,
roughly **$12** (estimate from wall time; the Modal dashboard has the exact figure). The ten
post-trained arms are **not** refit. Given ≤ 0.2% cosine at the worst layer and 0.000 at the layers
that carry most of the ladder's deep-layer statistics, a $120 refit of the other ten would change
no published conclusion; the base-lens measurement is the bound, and it is now a measurement. If
anyone needs the refit arms for a downstream use, the script takes `--arm`.
- [ ] **New (2026-10-08):** raise the 5% identity_distance discrepancy between Neuronpedia's old and new lenses with them — our refit moved 0.003% under the same fix

---

## Retraction, 2026-10-15: our lenses were never fitted with the bug

**Status now: the bug was real (transformers 5.0–5.12, Neuronpedia's June lens). Our eleven
OLMo-3 lenses were not affected. The "measured damage" of 8 Oct was version drift between two
fixed releases.**

### What was wrong in this file

The opening section says *"Our `uv.lock` resolves transformers 5.9.0 (verified)"* and concludes
our lenses were fitted with YaRN on 24 wrong layers. The lockfile does resolve to 5.9.0. The
lenses were fitted on Modal (`modal_olmo_ladder.py`, 2026-07-22 → 08-02), whose image is built
from `pip_install("transformers>=5.5,<6")` at build time and never reads `uv.lock`. The
`pip_install` line was last edited on 2026-07-21 (commit `e62e50d`), the day before the first fit,
so the image was built in the window 21 Jul – 9 Aug, when `>=5.5,<6` resolved to **5.14.1**
(5.13.0 shipped the fix on 3 Jul; 5.15.0 appeared 10 Aug). Confirmed by running a one-line
`env_report` inside the cached image (same definition → same build; Modal does not rebuild
unchanged definitions):

```
OLD-IMAGE ENV {'transformers': '5.14.1', 'torch': '2.13.0+cu130', 'python': '3.11.12', 'jlens_dist': '0.1.0'}
```

The Spark venv used by `jobs/jlens.sbatch` is transformers 5.13.1 (dist-info dated 2026-07-12),
also fixed. No forward pass in this project's OLMo-3 results ran the buggy RoPE. Every results
file carrying the banner *"computed from lenses fitted under transformers 5.9.0"* was wrong about
that; the banners now say so.

### What the 8 Oct "refit under the fix" actually measured

Both fits were on the corrected forward. `cos(ours_July, ours_Oct)` = 0.9978 at L0 → 1.0000 from
L12 is the numerical drift between transformers 5.14.1 / torch 2.13 and 5.19.0 / torch 2.14 (the
rotary code was refactored twice in between: #47598 "Simplify all Rotary modules", #49265 "Do
RoPE with mul instead of matmul"). It is a useful number — it is the version-noise floor — but it
is not the bug's damage, and `rope_damage_bound.md`'s 2.3%-at-L0 bound was a bound on the wrong
quantity for our lenses.

### The four lenses side by side (float64, `results/rope_refit_tf513/fourway_f64.json`)

| layer | cos(ours-Jul, NP-Jun) | cos(ours-Jul, NP-Sep) | cos(NP-Jun, NP-Sep) | id. dist: ours-Jul / ours-Oct / NP-Jun / NP-Sep |
|---|---|---|---|---|
| 0 | 0.8844 | 0.8968 | 0.9537 | 1.1696 / 1.1695 / 1.1643 / 1.1837 |
| 3 | 0.8993 | 0.9063 | 0.9663 | 1.0995 / 1.0993 / 1.0884 / 1.1069 |
| 6 | 0.9439 | 0.9398 | 0.9695 | 0.9946 / 0.9946 / 0.9878 / 1.0095 |
| 9 | 0.9563 | 0.9544 | 0.9757 | 0.9754 / 0.9754 / 0.9591 / 0.9945 |
| 12 | 0.9763 | 0.9773 | 0.9868 | 0.9042 / 0.9042 / 0.8906 / 0.9335 |
| 18 | 0.9928 | 0.9906 | 0.9958 | 0.6743 / 0.6743 / 0.6830 / 0.7277 |
| 24 | 0.9972 | 0.9957 | 0.9981 | 0.5343 / 0.5343 / 0.5429 / 0.5822 |
| 30 | 0.9998 | 0.9996 | 0.99985 | 0.2201 / 0.2202 / 0.2211 / 0.2322 |

Reading: our two (fixed) fits sit within ~1% of Neuronpedia's **June** lens in identity distance
at every layer, and are closer to it than to the September one from L6 down. Neuronpedia's
September lens is 1.5–5% *larger* than both at every layer.

### The bug's actual effect, measured by emulating it (`modal_rope_emulation_test.py`, ~$1)

Same image (transformers 5.19.0), same 8 prompts, same code path as the fits; one model loaded
stock, one with YaRN forced onto the 24 sliding-window layers (`rope_parameters["sliding_attention"]
= rope_parameters["full_attention"]`, verified on the module: `attention_scaling` 1.0 → 1.208,
`inv_freq` equal). Result in `results/rope_refit_tf513/rope_emulation.json`:

| | fixed | emulated bug | fixed vs bug |
|---|---|---|---|
| final hidden state, prompt 0 | | | rel. diff **0.226**, cos 0.974 |
| J₀ (n = 8) | id 1.647 | id 1.659 | cos **0.886**, rel 0.48 |
| J₉ | id 1.094 | id 1.067 | cos 0.962, rel 0.28 |
| J₃₀ | id **0.2368** | id **0.2279** | cos 0.99984, rel 0.018 |

The fix *raises* the layer-30 identity distance by ~4% (7 of 8 prompts individually), and moves
the early-layer transports a lot (n = 8, so the L0 number is noisy; Neuronpedia's converged
June-vs-September cosine at L0 is 0.954). **That is the bug's damage, and it is not "≤ 0.2%".** It
simply never applied to our lenses.

A second test (`modal_loadpath_test.py`) confirms the ladder's real loading path (`_load_lm`,
no explicit config) builds the same rotary embeddings and the same Jacobian as the explicit-config
path: prompt-0 identity distances identical to 15 digits, final hidden state bit-identical.

### So Neuronpedia's refit is right, and the "5% discrepancy" was ours to explain

Prompt 0 of our loader is prompt 0 of theirs (same wikitext stream). Single-prompt layer-30
identity distance:

| | ours, stock (5.19) | ours, emulated bug | Neuronpedia Sept (row 1) | Neuronpedia June (row 1) |
|---|---|---|---|---|
| prompt 0 | **0.24290** | **0.23380** | 0.24264 | 0.23365 |

Our fixed forward reproduces their **September** lens on the shared prompt to 0.1%; our emulated
bug reproduces their **June** lens to 0.1%. Their refit is on the corrected forward; so are both
of ours. The September-vs-June gap (+0.011 from prompt 1, constant to n = 568) *is* the RoPE fix,
as the reporter said. The thing I called "unexplained" on 8 Oct — our fixed lens agreeing with
their June (buggy) lens to 0.5% — is a coincidence: the prompt sets differ after prompt 0 (their
loader keeps records ours filters out), and the prompt-set effect on identity distance happens to
be about the size of the bug's effect and opposite in sign at every layer we fit (ours vs their
September, both fixed: −1.2% at L0, −1.9% at L9, −5.2% at L30). The old per-layer floor already
told us prompt sampling moves the off-identity part of J₃₀ by ~9%; I did not connect the two.

Consequence for the anchor gate: `jlens_lab.artifacts.validate_fit`'s identity-distance check
(±5%) has no power to tell a wrong-RoPE fit from a different-prompt-set fit of the same model.
Our July gate passed against the June lens *because* two ~5% effects cancelled. It is documented
as such in jlens-lab now; a gate that can catch this needs matched prompts on both sides.

Minor and real: their September run stopped at 568 of 1,000 prompts on `stop_at_delta 0.002`
with the identity distance still drifting (−1e-4 per prompt over the last rows). Not an error,
but their June lens had 616.

### What stands from the earlier updates

- The float32 cosine artifact (PITFALLS #28) and the per-layer floor (ours vs Neuronpedia,
  0.884 → 0.9998) are unaffected; the floor is protocol/sampling.
- Neuronpedia's artifact-format lesson (environment in the file) stands and we had not adopted
  it, which is exactly why this took six weeks to notice. Their refit is correct. `modal_rope_emulation_test.py` and
  future fits write a `provenance` dict.
- The 2.3% damage bound is withdrawn as a statement about our lenses; `rope_damage_bound.md` is
  marked.

### Checklist (replaces the one above)

- [x] Determine the environment the lenses were actually fitted in — **5.14.1, fixed**
- [x] Re-examine the 8 Oct refit null as a test of the premise, not a bound — it was the premise
- [x] Correct the thirteen results banners
- [x] The September discrepancy is explained (it is the fix); nothing to raise. A short
  confirmation note is drafted in `notes/neuronpedia_refit_discrepancy_draft.md` — your call
- [ ] Our `.pt` lenses still carry no provenance; add it on the next fit (the R-lens control)

Lesson recorded as PITFALLS #30 and in memory. The honest summary for the README: *we filed an
erratum against ourselves on the strength of a lockfile, spent $12 refitting a lens that was
already right, and found out by probing the image. The erratum was correct about Neuronpedia's
June lens and about the floor; it was wrong about us — and the bug it described was worse than our 8 Oct number said.*
