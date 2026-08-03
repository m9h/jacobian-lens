# ARC-AGI with LLMs — research dive (Aug 2026)

Commissioned *after* the pilot had already launched, which was the error; see
[DESIGN_POSTMORTEM.md](DESIGN_POSTMORTEM.md). Preserved verbatim because the distillation in the
post-mortem drops most of it — the baseline tables, the power analysis, the voting evidence, the
harness inventory, and the report's own account of what it could not verify.

Treat the numbers as sourced-but-unaudited: they are as the report gives them, with its links.

---

All research complete. Here is the report.

---

# ARC-AGI: how it's actually evaluated, and a verdict on your design

## Verdict first

**Your design will not produce literal all-zeros, but it will produce statistically uninformative near-zeros — and the harness axis you chose is the one harness variant the literature has specifically measured and found does *not* work on ARC.** Both of your axes are dead for different reasons:

- **Model axis is dead** because direct-prompted 4B/8B models score **~0.5–5%** on ARC. At n=40 tasks you expect **0–2 solves per arm**. You need a 0-vs-5 split to reach p≈0.055 (Fisher exact, two arms of 40). You will almost certainly get 1 vs 2 and learn nothing.
- **Harness axis is dead** because majority voting over temperature samples is documented to be *neutral-to-harmful* on ARC. Akyürek et al. state it is "not viable"; the ARChitects measured stochastic sampling as **worse than greedy**. Your `k=1 → k=16` delta will be ≈0, and you will mistakenly conclude "harness doesn't matter" — the opposite of the true finding.

The irony is that your underlying thesis is **correct and well-supported** — harness dominates base model on ARC. You've just picked the one harness that doesn't. Details and a fix below.

---

## 1. Scoring protocol

**pass@2 confirmed.** Two attempts per *test input*; the first exact match counts. ([ARC Prize guide](https://arcprize.org/guide/1), [ARC-AGI-2 report](https://arxiv.org/abs/2505.11831))

- **No partial credit.** Every cell *and* both dimensions must match exactly. Off-by-one on grid size = 0.
- **Multi-test tasks:** a task scores 1 only if **all** its test pairs are solved. This can be fractional in some ARC Prize reporting (mean per-task solve rate), but the official harness (`scoring.py`) requires all pairs correct for the task to count.
- Pixel-correctness exists as a documented *secondary indicator*, not the headline metric — useful to you, see §8.
- **Relevant to your script:** 14 of the 400 ARC-AGI-1 training tasks have >1 test input. You only use `test[0]`.

## 2. Prompting / serialization

Your `grid_str` (space-separated digits, row per line) is **exactly ARC Prize's own published o3 prompt format**. That's fine. Two canonical templates, and they disagree with each other:

- [`arc-agi-benchmarking/prompts/system_prompt.txt`](https://github.com/arcprize/arc-agi-benchmarking) — grids as **JSON nested lists** via `json.dumps`.
- [`docs/examples/prompt_example_o3.md`](https://github.com/arcprize/arc-agi-benchmarking/blob/main/docs/examples/prompt_example_o3.md) — **space-separated, row per line**, opening "Find the common rule that maps an input grid to an output grid…"

**Does format matter?** Yes, but mostly at the bad end:

| Finding | Source |
|---|---|
| Dropping the in-row delimiter halves GPT-4 few-shot (11–12 → 5 of 50 tasks) | [Xu et al., TMLR 2024](https://arxiv.org/abs/2305.18354) |
| BIG-Bench's near-zero ARC scores were a *tokenization artifact* — `8686` vs `8 6 8 6` merges cells into single BPE tokens | [Mirchandani et al., CoRL 2023](https://arxiv.org/abs/2307.04721) |
| Color-words vs digits: within noise | Xu et al. |
| Replacing digits with rare symbols: **no benefit** (2.0% → 1.9%) | [ironbar ablation](https://ironbar.github.io/arc24/modeling/Iteration_12_grid_representation/) |
| Stating grid shape + row indices helps (correct_pixels 58→66%) | ironbar |
| Suboptimal format costs ~10% relative | [Land 2026](https://arxiv.org/abs/2606.31543) (prose only, no table) |
| **Images are much worse than text**: o3 75.6% text vs 29.2% image on ConceptARC | [Beger/Mitchell et al.](https://arxiv.org/abs/2510.02125) |

Kaggle winners sidestep the issue by using **custom tokenizers**: the ARChitects cut vocabulary to ≤64 tokens (one per cell); NVARC to 16. Nobody has ablated custom-vs-stock tokenizer.

## 3. ★ Baseline numbers for 4B–8B open models — the number you needed

This is the decisive evidence. From [DiARC (arXiv 2606.26530)](https://arxiv.org/abs/2606.26530), Table 1, **ARC-AGI-1 400-task public evaluation set, direct prompting, two candidates**:

| Model | ARC-AGI-1 | ARC-AGI-2 |
|---|---|---|
| **Qwen3-4B** | **0.63%** | 0.00% |
| Minitron-8B | 1.12% | 0.00% |
| Llama-3.2-3B | 0.50% | 0.00% |

And on **MiniARC (5×5 grids)** — i.e. your "shrink the grids" mitigation, taken further than you planned: **Qwen3-4B 0.67%**, Llama-3.2-3B 1.34%, Minitron-8B 6.04%. **Restricting to small grids does not rescue a 4B model.**

On the *easier* training set, [Wu et al. NAACL 2025 (arXiv 2502.07190)](https://arxiv.org/abs/2502.07190), 100 ARC training tasks:

| Model | Accuracy | **Wrong output shape** |
|---|---|---|
| Mistral-7B-Instruct | 2.00% | **48%** |
| Llama-3-8B-Instruct | 5.00% | **33%** |
| Llama-3-70B-Instruct | 9.00% | 24% |
| GPT-4o | 19.00% | 11% |

And the bluntest statement, from the MIT TTT paper: **"The zero-shot baseline is 0 for ARC"** (raw Llama-3 8B Instruct, no fine-tuning).

⚠️ DiARC's *post-method* numbers (Qwen3-4B → 97% on ARC-AGI-1) are implausible and smell like eval contamination. I would cite only their baseline column.

**Power analysis on your exact design** (n=40, two arms):

```
true rate  0.5% -> expect 0.20 solves;  P(zero solves) = 0.82
true rate  2.0% -> expect 0.80 solves;  P(zero solves) = 0.45
true rate  5.0% -> expect 2.00 solves;  P(zero solves) = 0.13

Fisher two-sided, 40 vs 40:  0 vs 3 -> p=0.24    1 vs 5 -> p=0.20
                             0 vs 5 -> p=0.055   2 vs 9 -> p=0.048
```

Realistic expectation for Qwen3-4B vs Qwen3-8B on 40 small-grid *training* tasks: roughly **1 vs 2 solves**. Unpublishable.

**Also:** only **238 of 400** training tasks have output ≤10×10 (and only **179** have *all* grids ≤10×10). You filter on the output grid only — the largest input among your eligible tasks is **900 cells (30×30)**, so your prompt-length control doesn't actually bind.

## 4. Does majority voting help on ARC? Not the way you're doing it

**Your premise is half-wrong, and the conclusion is worse than you feared.** Fine-tuned ARC models are *highly* self-agreeing — the ARChitects' DFS finds only **~9.3 distinct candidates per task** above a 9% probability threshold. So samples *do* agree; there's simply nothing to vote over. Naive resampling produces near-duplicates.

Measured head-to-head ([ARChitects Table 5](https://da-fr.github.io/arc-prize-2024/the_architects.pdf), 100 tasks):

| Sampling | Llama-mix top-2 | Nemo-mix top-2 | Coverage |
|---|---|---|---|
| Greedy | 51.5 | 59.0 | 65.5 / 75.0 |
| **Stochastic** | **50.5** | **58.5** | 65.5 / **72.0** |
| DFS 10% | 51.5 | 60.5 | **73.0 / 80.0** |

Stochastic is *worse than greedy* at longer runtime. Akyürek et al. verbatim: *"this is not viable in ARC… there is no way to directly enforce diversity across samples while ensuring coherence within samples"* — they used **greedy decoding throughout**.

**What actually works is voting over *augmentation*-induced diversity, not temperature:**

| Method | Gain |
|---|---|
| TRM: 1000 augmentations + majority vote vs single canonical pass | **29.25% → 40.00%** ([arXiv 2512.11847](https://arxiv.org/abs/2512.11847)) |
| MindsAI **AIRV** (augment, infer, reverse-augment, vote), ARC private | **5 → 13 tasks** (+260%) ([arXiv 2506.14276](https://arxiv.org/abs/2506.14276)) |
| ARChitects: product-of-experts over 16 augmentations vs single-perspective argmax | **63.5% → 71.6%** |
| ARChitects: pixelwise-similarity *voting* vs PoE | loses by **5.5–6.5 pts** |
| MIT TTT: augmented inference + hierarchical voting vs greedy-vanilla | ~38% → ~50% |

**The programs-vs-grids crux is real and it is the whole story.** BARC uses *the same model* with completely different inference machinery depending on output type:

| | Induction (programs) | Transduction (grids) |
|---|---|---|
| Temperature | **0.8** | **0** |
| Decoding | stochastic, 2k–20k samples | **beam search** |
| Selection | **execute against train pairs → filter →** majority vote | augmentation reranking |

They turn temperature *on* for programs and *off* for grids. Greenblatt likewise samples 8,000 programs at t=1.0 but selects by **execution correctness on training pairs**, with voting only as a downstream tie-breaker (~9% of train-passing programs are false positives; voting kills about half). SOAR weights execution accuracy over grid vote-count by a factor of **c=1000**.

Additional caution: on hard tasks the majority is *wrong*. Land's ARC-AGI-2 judge beats majority vote by **+7 instances, all of them minority recoveries**; on one task, 29/29 candidates converged on the same wrong grid.

## 5. Solution families and the harness-vs-model evidence

**This is where your thesis is genuinely well-supported — just not by your experiment.**

The cleanest published decomposition, [Product of Experts with LLMs (arXiv 2505.07859)](https://arxiv.org/abs/2505.07859), ARC-AGI-1 400-task public eval:

| Model | Baseline | +TTT | +16×Aug | +PoE | +DFS |
|---|---|---|---|---|---|
| Llama-3.2-3B | 14.9% | 40.9% | 52.9% | 59.5% | 61.4% |
| **NeMo-Minitron-8B** | **18.3%** | 44.5% | 62.5% | 67.6% | **71.6%** |

**Harness on frozen 8B: +53.3 pts. Base-model swap 3B→8B: +3.4 pts at baseline, +10.2 pts at the end.** In the ARChitects' own Kaggle timeline, the single pure base-model swap was worth **+3 pts (44.0 → 47.0)**; everything else was harness.

Second clean decomposition — [Moghe & Chin, arXiv 2607.06764](https://arxiv.org/abs/2607.06764), DeepSeek V3.2, no ARC-specific training:

| Architecture | Score | $/task |
|---|---|---|
| One-shot | 15.50% | $0.002 |
| CoT | 30.00% | $0.004 |
| Explorer–Definer pipeline | 57.50% (pass@2) | $0.25 |
| Reflective Orchestrator | **67.25%** (pass@2) | $0.62 |

**+51.75 pts from harness alone**, and it *replicates on Qwen3-235B* (17.2 → 54.6). Their conclusion: *"evidence that the harness, not the specific model, drives the lift."* Note their selection rule is explicitly **anti-vote** — they *deduplicate* by predicted grid so convergent wrong answers can't crowd out a dissenting correct one.

Other key results:

- **Test-time training** — [Akyürek et al. 2411.07279](https://arxiv.org/abs/2411.07279): Llama-3 8B, fine-tuned 18.3% → **+TTT 47.1%** → 53.0% → 61.9% ensembled with BARC. Code: [ekinakyurek/marc](https://github.com/ekinakyurek/marc).
- **Program synthesis** — Icecuber 2020 hand-built DSL, ~20% (rescored 17.0% by ARC Prize). [Greenblatt/GPT-4o](https://blog.redwoodresearch.org/p/getting-50-sota-on-arc-agi-with-gpt): 50% public eval, **+3% per doubling of samples** (1024→25-30%, 2048→34%, ~8000+revision→50%); ARC Prize verified 42-43%.
- **Berman's evolutionary search** — 2024 Sonnet-3.5 evolving Python: 53.6%. 2025 Grok-4 evolving **English-language instructions**: **79.6% ARC-AGI-1 / 29.4% ARC-AGI-2**, vs bare Grok-4-thinking at 66.7/16.0 — a **+12.9/+13.4 pt pure harness delta on a fixed base model**. Code: [jerber/arc-lang-public](https://github.com/jerber/arc-lang-public).
- **Induction vs transduction** — [BARC, arXiv 2411.02272](https://arxiv.org/abs/2411.02272), Llama-3.1-8B both heads: transduction 19.25% → 43.00% via TTT + reranking; ensemble **56.75%**. The two families solve *different* tasks — that's why ensembling adds +13.75.
- **TRM** — 7M params, 44.6%/7.8%. But the control row in its own Table 5 is damning: *same 27M network without recursion* = **21.0% / 0.0%**. And [ARC Prize's HRM autopsy](https://arcprize.org/blog/hrm-analysis) found a plain transformer comes within **~5 pts** of HRM, the outer refinement loop is worth **+13 pts for one iteration**, and the model *requires the eval puzzle IDs at training time* — it's transductive test-time training wearing an architecture costume.
- **NVARC** (2025 winner, **24.03% ARC-AGI-2 private, $0.20/task**) — [github.com/1ytic/NVARC](https://github.com/1ytic/NVARC). **Base model: Qwen3-4B-Thinking-2507.** Same size class as your "small" arm. 103k synthetic puzzles → 3.2M augmented samples, full fine-tune, per-puzzle LoRA TTT, DFS, augmentation rescoring. Their own conclusion: *"The core idea of our solution is the novel synthetic dataset."* Notable negative: adding TRM to the ensemble **helped nothing** (27.22 with vs without).
- ARC Prize 2024 report's summary line: ***"there does not exist any static inference-style transduction solution that scores above 10%."*** That sentence alone predicts your result.

## 6. Open harnesses — don't hand-roll

- **[arcprize/arc-agi-benchmarking](https://github.com/arcprize/arc-agi-benchmarking)** — the official harness that produces the leaderboard. (`arcprize/model_baseline` is the old name and redirects.) Implements pass@2 (`--num_attempts`, default 2), cost/token accounting, provider adapters. Not on PyPI; `uv sync`.
- **Per-model result JSONs**: [HF `arcprize/arc_agi_v1_public_eval`](https://huggingface.co/datasets/arcprize/arc_agi_v1_public_eval) — **77 models × ~400 tasks**. You can compute baselines without spending a GPU-hour.
- **Live leaderboard as raw JSON**: `https://arcprize.org/media/data/leaderboard/v1.json` (also `v2`, `v3`).
- **[mxbi/arckit](https://github.com/mxbi/arckit)** — `pip install arckit`. Data loading, pass@2 scoring, visualization. Closest to "just import it."
- Datasets: [fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI) (400/400), [arcprize/ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) (1000 train/120 eval). ⚠️ `github.com/arcprize/ARC-AGI` is now the *ARC-AGI-3 Toolkit*, not the v1 dataset.
- ⚠️ **`arc_challenge` in lm-evaluation-harness and inspect_evals is NOT this benchmark.** It's [AI2 Reasoning Challenge (Clark et al. 2018)](https://arxiv.org/abs/1803.05457) — grade-school science multiple-choice. There is **no ARC-AGI eval in any mainstream harness**. Any "Phi-4-mini scores 83.7% on ARC" headline is that one.

## 7. ARC-AGI-2 and -3

**ARC-AGI-2** (Mar 2025): symbolic interpretation, compositional multi-rule reasoning, contextual rule application. Human-calibrated with **407 participants / 13,405 attempts**; every eval task solved by ≥2 humans within 2 attempts. Efficiency is first-class (cost-per-task on the leaderboard axis). 1000 train / 120 public eval / 120 semi-private / 120 private.

Current verified semi-private (snapshot 2026-07-31):

| Track | Best | Human |
|---|---|---|
| ARC-AGI-1 API | **98.0%** Gemini 3.1 Pro @ $0.52 | 98% @ $17 — **saturated** |
| ARC-AGI-2 API | **92.5%** GPT-5.6 Sol (Max) @ $1.44 | 100% |
| ARC-AGI-2 Kaggle-constrained | **~24%** NVARC @ $0.20 | 100% |
| ARC-AGI-3 | **30.2%** Claude Opus 5 (High) | 100% |

The **~68-point gap between the API track (92.5%) and the Kaggle track (~24%)** is the single most important fact here, and conflating them is the most common error in secondary coverage.

**ARC-AGI-3** ([arXiv 2603.24621](https://arxiv.org/abs/2603.24621)): interactive/agentic game environments, no instructions, efficiency-scored against human action baselines. Launched Mar 2026; 25 public demo environments. Frontier models were **<1% in March 2026**, ~30% by July. Humans 100%. Milestone #1 winner was Tufa Labs' "Duck Harness" — a *small open LLM writing Python in a live REPL*, which is itself a harness-over-model result.

---

## 8. Specific problems in `arc_attribution.py`

Beyond the two fatal design issues:

1. **`parse_grid` will silently under-count.** It requires digit rows immediately after the prompt and returns `None` on any preamble. Qwen3-4B/8B are *instruct* models. Given the documented **33–48% wrong-shape rate** for 7–8B models, plus parse failures, you'll conflate "no grid emitted" with "wrong grid." Extract the *last* grid-like block instead.
2. **No chat template**, and Qwen3 defaults to `enable_thinking=True`. Raw completion avoids the `<think>` blowup (good — `max_new=600` would be entirely consumed by thinking), but it's an untested regime. Decide deliberately and document it.
3. **`temperature=0.8`** is precisely the setting the literature says to avoid for grid output. BARC uses **t=0** + beam for grids.
4. **You score pass@1 majority-vote, not the official pass@2.** Top-2 most-common grids is the standard and strictly more generous.
5. **`test[0]` only** — 14 training tasks have multiple test inputs; official scoring needs all of them.
6. **Grid-size filter binds on output only.** Eligible tasks include 30×30 inputs.
7. **You use 40 of 238 eligible tasks.** The pool is free; use all of it.
8. `json.dump` of the whole growing dict every task is O(n²) — harmless at this scale, but use JSONL.

---

## What I'd do instead

Your research question is good. Rebuild both axes:

**Harness axis — replace majority-vote with something documented to work.** Cheapest credible option: **augmentation voting (AIRV)** — apply the 8 dihedral transforms × color permutations to the task, decode **greedily** on each, invert the transform, vote. Worth **+6 to +11 pts** in every study that measured it, and it's ~30 lines. Better option if you want a big delta: **program synthesis with execution filtering** — have the model emit Python, run it against the training pairs, keep only programs that reproduce them. That's verifiable selection, it's where the large harness deltas live, and it works at t=1.0.

**Model axis — you need a non-zero substrate.** At 0.63% there is nothing for a harness to amplify. Either (a) move up to models with a measurable baseline (Qwen3-32B+, or API models via `arc-agi-benchmarking`), or (b) reframe: run *one* model across a harness ladder (direct → CoT → augmentation-vote → program+execution-filter), which reproduces the Moghe & Chin design and would give you a real, publishable ladder on hardware you have.

**Metrics — add continuous ones so you get signal at low exact-match.** Report pass@2 *plus* **correct-shape rate** and **cell-level accuracy**. Shape accuracy is discriminative exactly where exact-match isn't (Mistral-7B 48% mismatch → GPT-4o 11%), it's an ARC Prize–sanctioned secondary indicator, and it will separate your 4B from your 8B when exact-match cannot.

**Pool** — all 238 small-grid training tasks minimum; ideally all 400 plus the 400 eval tasks so your numbers are comparable to published ones.

**Free win** — before running anything, pull [HF `arcprize/arc_agi_v1_public_eval`](https://huggingface.co/datasets/arcprize/arc_agi_v1_public_eval) (77 models × 400 tasks of per-task pass/fail with costs). Much of the model-axis half of your decomposition is already sitting there.

---

### Things I could not verify

- **No published measurement exists of the exact-grid agreement rate between independent LLM samples on ARC.** H-ARC defines the metric and explicitly declines to compute it for models. Indirect evidence (ARChitects' ~9.3 candidates/task; Moghe & Chin's "strongly bimodal" candidate distribution) points *against* your assumption — models agree with themselves a lot. **This looks like a genuinely unoccupied measurement, and it is cheap to run.** It may be a better paper than the one you planned.
- DiARC's post-method numbers (97% ARC-AGI-1 from a 4B) — I'd treat as contaminated; only the baseline column is usable.
- No paper runs a clean head-to-head of execution-filtered program voting vs. grid majority voting on ARC. The crux in §4 rests on convergent indirect evidence, not one controlled ablation.
- Land's "9 formats / ~10%" claim is prose only — no table, no n, no model named.
- Nobody has ablated the ARChitects'/NVARC custom tokenizer against a stock one.
- Current ARC Prize 2026 Kaggle standings (auth-gated).
- MIT TTT's 61.9% (abstract) vs 62.8% (Table 1) inconsistency; TTT Fig. 7 values are read off a figure.

**Sources:** [ARC Prize guide](https://arcprize.org/guide/1) · [leaderboard](https://arcprize.org/leaderboard) · [2024 tech report](https://arxiv.org/abs/2412.04604) · [2025 tech report](https://arxiv.org/abs/2601.10904) · [ARC-AGI-2](https://arxiv.org/abs/2505.11831) · [ARC-AGI-3](https://arxiv.org/abs/2603.24621) · [DiARC](https://arxiv.org/abs/2606.26530) · [Fluid Intelligence Deficiency](https://arxiv.org/abs/2502.07190) · [Test-Time Training](https://arxiv.org/abs/2411.07279) · [Product of Experts](https://arxiv.org/abs/2505.07859) · [The LLM ARChitect](https://da-fr.github.io/arc-prize-2024/the_architects.pdf) · [BARC](https://arxiv.org/abs/2411.02272) · [SOAR](https://arxiv.org/abs/2507.14172) · [AIRV](https://arxiv.org/abs/2506.14276) · [TRM](https://arxiv.org/abs/2510.04871) · [TRM analysis](https://arxiv.org/abs/2512.11847) · [HRM autopsy](https://arcprize.org/blog/hrm-analysis) · [Cost-Effective Harnesses](https://arxiv.org/abs/2607.06764) · [Land 2026](https://arxiv.org/abs/2606.31543) · [Greenblatt](https://blog.redwoodresearch.org/p/getting-50-sota-on-arc-agi-with-gpt) · [Berman](https://jeremyberman.substack.com/p/how-i-got-the-highest-score-on-arc-agi-again) · [NVARC](https://github.com/1ytic/NVARC) · [Xu et al.](https://arxiv.org/abs/2305.18354) · [Pattern Machines](https://arxiv.org/abs/2307.04721) · [Mitchell et al.](https://arxiv.org/abs/2510.02125) · [H-ARC](https://arxiv.org/abs/2409.01374) · [arc-agi-benchmarking](https://github.com/arcprize/arc-agi-benchmarking) · [arckit](https://github.com/mxbi/arckit)
