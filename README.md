# Jacobian lens — open replication, controls, and the Consciousness-Indicator Scorecard

> **Attribution.** This repository is a **derivative** of Anthropic's reference implementation
> [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) (Apache-2.0),
> companion code for [*Verbalizable Representations Form a Global Workspace in Language
> Models*](https://transformer-circuits.pub/2026/workspace/index.html). The `jlens/` package is
> theirs; the upstream README is preserved as [`UPSTREAM_README.md`](UPSTREAM_README.md).
> **Everything else here is independent work and is not endorsed by, affiliated with, or
> reviewed by Anthropic.** Their notice — "not maintained and not accepting contributions" —
> applies to the upstream library, not to this fork.

Anthropic's global-workspace result rests on a closed model; the paper's own invited
commentators could not check it. This repository does the replication on **open weights**, runs
the controls the source papers omit, and reports the results that came out negative.

## What is here

| directory | contents |
|---|---|
| `jlens/` | **upstream** Anthropic library (unmodified) |
| `results/` | our measurements — OLMo post-training ladder, metacognition, ignition, the reviewer battery, cross-validation, best-of-N |
| `proposal/` | the **Consciousness-Indicator Scorecard** and its design notes |
| `docs/` | technique lineage; the interpretability curriculum survey |
| `modal_*.py`, `spark_*.py` | runnable experiments (Modal cloud / DGX Spark) |

## Findings

**Post-training installs a viewpoint, decoupled from capability.** Across the fully open OLMo-3
ladder, post-training moves the J-space ~31% from base (cos 0.69) while MMLU is flat to slightly
*down*. **Method sets the magnitude, not domain**: SFT+DPO moves it ~5× more than RLVR, and
varying the RLVR domain at matched capability adds ~1%. First quantitative, controlled test of a
claim Anthropic could only state qualitatively. → `results/posttrain/`

**A covert error signal, made reportable by SFT.** The base model's workspace predicts whether
its own answer is wrong (AUROC 0.69) *beyond what its output distribution reveals*, while its
verbal self-assessment is at chance (0.51). Post-training raises verbal self-eval to 0.71–0.78
— at **SFT**, not RLVR. Two dissociable milestones. → `results/metacognition_result.md`

**The reviewers' battery, executed.** Dehaene & Naccache proposed six tests and noted Anthropic
could run them; none were, because the model is closed. All six were run here, with an honest
mixed outcome: two clean signatures, one partial, three inconclusive under first-pass
adaptations whose flaws are documented. → `results/reviewer_tests_results.md`

**External cross-validation, and a per-layer refit floor.** Against Neuronpedia's independently
fitted lens for the same model, the final layer agrees to cosine **0.9998** — but agreement is
strongly layer-dependent (0.884 at layer 0). J-space claims must be read against a *per-layer*
floor. → `results/neuronpedia_crossvalidation.md`

**Two negatives we report as results.** Introspection came out as *steering, not introspection*.
And the covert error signal does **not** improve best-of-N selection (+0.008 over a logprob
baseline) — question-level and sample-level metacognition are different tasks, and we only have
the first. → `results/posttrain/bestofn_validity.md`

## Related

- [**spinning-up-in-mech-interp**](https://github.com/m9h/spinning-up-in-mech-interp) — the
  curriculum that teaches these techniques; rungs 6 and 8 use this repo.
- [**tri-lens**](https://github.com/m9h/tri-lens) — do three instruments agree about the same
  activation?
- [**societies-of-thought**](https://github.com/m9h/societies-of-thought) — the adversarial
  replication: rebuild a no-code/no-data paper, then try to break it. Sibling project;
  see `docs/JSPACE.md` there for how the two claims connect, and
  `NOTE_FROM_SOT_AGENT.md` here for the current cross-project state.
- [**controls-and-trajectories**](https://github.com/m9h/controls-and-trajectories) — the
  published null/trajectory datasets.
- Lenses on the Hub: [mhough/olmo3-jacobian-lenses](https://huggingface.co/mhough/olmo3-jacobian-lenses).

Apache-2.0, matching upstream. Work by Morgan Hough, Orthogonal Research and Education Lab (OREL).
