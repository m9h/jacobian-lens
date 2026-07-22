# Four papers, one move

*What each technique actually is, where it came from, and what it controls for.*
*Compiled 2026-07-18 from primary sources.*

## The move

Every paper below does the same thing:

> **Write a direction into the residual stream. Read out an effect. Name it after a
> construct from cognitive science.**

That is a legitimate research design. It becomes a measurement only when you can show the
effect is specific to *that* direction rather than to perturbation in general. The papers
differ almost entirely in three places: **where the direction comes from**, **what space
the readout lands in**, and **what control is run**.

The third column is the one that varies least, and least usefully.

---

## The four papers

| | **Manifolds**<br>Gurnee et al., 21 Oct 2025 | **Introspection**<br>Lindsey, 29 Oct 2025 | **Societies of Thought**<br>Kim et al., 15 Jan 2026 | **J-space**<br>Gurnee/Lindsey et al., 6 Jul 2026 |
|---|---|---|---|---|
| **Direction from** | PCA of 150 per-condition mean activations | difference-in-means concept vector | SAE feature decoder (#30939, L15) | Jacobian lens `J_l = E[∂h_final/∂h_l]` |
| **Basis defined by** | data | data | learned dictionary | model (+ data to estimate `J_l`) |
| **Readout space** | activation (unlabeled axes) | model's verbal report | task accuracy | **vocabulary (tokens)** |
| **Intervention** | zero-ablate top-k PCs; mean-patch `a − μ_orig + μ_c` | add vector at ~⅔ depth, strength 2–4 | add decoder direction × α | scale / ablate / patch in lens coords `V=[v_s,v_t]` |
| **Scored by** | loss effect, probe RMSE | LLM judge, 4 conjunctive criteria | Countdown accuracy | token rank (`pass@k` = min rank over layers) |
| **Models** | Claude 3.5 Haiku | Opus 4.1 + 9 Claude models | DeepSeek-R1-Distill-Llama-8B | Sonnet 4.5, Opus 4.6 (+38 open lenses) |
| **Code released** | none | none | none | **yes** (Apache-2.0 + 38 lenses) |
| **Construct claimed** | place/boundary cells; feature manifolds | introspection, metacognition | society of thought | global workspace |

### Controls, side by side

| control | Manifolds | Introspection | SoT | J-space |
|---|---|---|---|---|
| matched-dimension **random subspace** | ✅ | — | ✗ | ✗ |
| **norm-matched random vector** | — | ✅ (9/100 vs concept) | ✗ | ✗ |
| **negative/inverted** direction | — | ⚠️ **fails** (equally effective; "no discernible pattern") | ✗ | ✗ |
| no-intervention baseline | — | ✅ (0/100 false positives) | ✅ | ✅ (strength 0) |
| unrelated-question specificity | — | ✅ | ✗ | partial (selectivity evals) |
| **model randomization** (untrained control) | ✗ | ✗ | ✗ | ✗ |
| **distance/drift null** for geometry claims | ✗ | n/a | n/a | ✗ |

Two things to read off this table:

**Lindsey's introspection paper is the best-controlled of the four.** It has the
norm-matched random vector that the others lack. Its weakness is one *it reports itself*:
injecting the **negation** of a concept vector was "comparably effective," with reported
words showing "no discernible pattern" (*mirror, water, Pennsylvania, awareness*). His own
reading — "we suspect the former [confabulation] is likely" — concedes that identification
may be confabulated even if detection is real.

**No paper runs a model-randomization control.** That is the standard sanity check in the
attribution literature (Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018),
and its absence across all four is the clearest single gap. We ran it for the J-lens: it
**passes** (0.0003 vs 0.3414 next-token accuracy on randomized blocks). That is a point in
Anthropic's favour that no one had established.

---

## Where these sit in the older toolchain

The lineage matters because most of these techniques are not new — only their framing is.

### Readouts into vocabulary space

| tool | map | fitted? |
|---|---|---|
| **logit lens** (nostalgebraist, 2020) | `unembed(h_l)` | no |
| **tuned lens** (Belrose et al., 2023) | `unembed(A_l h_l + b_l)` | learned affine |
| **J-lens** (2026) | `unembed(J_l h_l)` | `J_l` estimated as an expected Jacobian |

These are one family. The J-lens is an **analytically-derived tuned lens** — and it
*converges to* the logit lens with depth: we measured `‖J−I‖/‖I‖` falling 8.75 → 0.53 from
layer 0 to 26. Anthropic concede the same in prose ("the logit lens... capture[s] much of
the workspace-like structure"). Our own numbers put the J-lens advantage at 1.7× over the
logit lens at 27B, shrinking to ~1.05× at 32B.

### Directions in activation space

| tool | how the direction is found |
|---|---|
| **probing classifiers** (Alain & Bengio, 2016) | supervised, on labels |
| **difference-in-means / concept vectors** (Bolukbasi 2016; Zou et al. RepE 2023) | contrast two prompt sets |
| **PCA on activations** | unsupervised variance |
| **SAEs** (Bricken et al. 2023; Cunningham et al.) | learned sparse dictionary |

The manifolds paper's PCA and Lindsey's concept vectors are **close cousins** — both are
data-defined directions in activation space needing post-hoc interpretation. Neither lands
in token space.

**This is what the J-space paper actually contributed.** Swapping a data-defined
activation direction for a *vocabulary-indexed* readout is what makes "verbalizable" a
meaningful predicate. PCA structurally cannot make that claim; its components have no
words attached.

### Interventions

| tool | operation |
|---|---|
| **causal tracing / activation patching** (Vig 2020; Meng et al. ROME 2022) | copy activations between runs |
| **activation addition** (Turner et al. 2023); **CAA** (Rimsky et al. 2024) | add a steering vector |
| **mean-patching** (manifolds) | `a − μ_orig + μ_c`, in a PCA basis |
| **lens-coordinate patching** (J-space) | same operation, in a learned lens basis |

The last two are the same surgical move in different coordinate systems. That, not the
lens, is what J-space inherits from the manifolds paper.

---

## What this implies

1. **The techniques are mature, and so is the literature on how they fail.** Activation
   patching, steering vectors and probes predate these papers by years, and each has a
   documented failure mode: attention weights that look explanatory and are not (Jain &
   Wallace); interpretability illusions in BERT (Bolukbasi et al.); subspace patching that
   yields clean, causal-looking, *wrong* results by exploiting dormant pathways (Makelov,
   Lange & Nanda); and saliency maps that survive scrutiny until you randomize the model
   (Adebayo et al.). Full references below.

   **I cannot verify whether these papers cite that literature** — the J-space
   bibliography is JS-loaded and did not survive text extraction, so a zero keyword count
   proves nothing. An earlier draft of this document asserted they do not. That claim is
   withdrawn.

   What *is* verifiable is more interesting: **Adam Pearce co-authored "An
   Interpretability Illusion for BERT" (2021) and is an author on both the manifolds paper
   and the J-space paper.** The relevant expertise is on the team. So the missing
   model-randomization control is not an awareness gap — which makes it a more interesting
   fact about the field's incentives, not a less damning one.

2. **Two independent groups converged on the same gap.** Anthropic and the Chicago/Google/SFI
   group share no authors and do not cite each other, yet both steer a direction, observe an
   effect, and borrow a construct — and both headline claims dissolve under a matched
   control (SoT's effect is a Countdown artifact; J-space's "emergence" was one lucky
   prompt).

3. **The one checkable paper borrows its checkability from a volunteer.** Neuronpedia
   (`hijohnnylin/neuronpedia`, MIT, created **June 2023** — three years before the J-space
   paper) is an independent open-source platform owned by an individual, not by Anthropic.
   It also hosts Gemma Scope (DeepMind), Llama Scope (OpenMOSS) and circuit-tracer graphs.
   Every published lens's `config.yaml` states the division of labour verbatim: *"Jacobian
   lens ('jlens') by Anthropic PBC... Apache-2.0. **Fit via Neuronpedia
   run-all-fit-lens.py**."* Anthropic wrote the estimator; a third party ran the fits and
   hosts them. Anthropic's own HuggingFace org publishes **zero models**.

   Three things follow:

   - **The convergence criterion is Neuronpedia's code, not Anthropic's.** `jlens.fit()`
     ships with no stopping rule, yet every published lens was fit to one
     (`--stop_at_delta 0.002 --min_prompts 100 --stop_window 10`). That logic lives in
     Neuronpedia's unreleased wrapper. The gap `jlens-lab` fills is not a withheld
     component — it is a seam between two owners, only one of whom published.
   - **The broken artifact is the third party's.** `qwen3-32b`'s "lens" is a raw fit
     checkpoint at `n_done=80` against a config claiming 615 — a pipeline error from a
     small operation, not a defect in the paper's method. We audited all 38: **exactly one
     is broken.** (`qwen3.6-27b` initially looked broken from a directory listing showing
     only `.DS_Store` files; its `.pt` is present and valid at 1000 prompts. Corrected.)
   - **Independent scrutiny of this literature currently rests on one person's side
     project.** That is a fragile foundation for a field making consciousness-adjacent
     claims, and it is an argument for funding measurement infrastructure directly rather
     than assuming the labs will supply it.

4. **Only one of the four is independently checkable at all.** Manifolds, Introspection and SoT
   released no code, no vectors, no data. Every headline claim in all four rests on models
   the public cannot access — Claude 3.5 Haiku, Opus 4.1, Sonnet 4.5, Opus 4.6. The J-space
   paper's 38 published lenses are for *other* models than the ones it makes claims about.

5. **The rebuttal has the same problem.** Singh, Linzen & Ravfogel's *"Can LLMs Introspect?
   A Reality Check"* (arXiv:2605.26242) is the properly-controlled replication — three-way
   discrimination (input-level vs activation-level vs no intervention), an input-only
   classifier baseline, and a relabeled control that drops accuracy to chance. It tests
   Llama-3.1-8B/70B, Qwen-2.5-72B and Gemma-3-27B. No code release found.
   **Note:** their models go to 70–72B, so a "small-model artifact" objection does *not*
   apply to their negative result — we checked, and dropped that line of attack.


---

## What the OLMo-3 replication adds to the lineage

We run the same moves on fully open weights, which lets us supply the controls the four
papers lack and — crucially — connect them to the introspection debate they set off. So our
work is not only a response to the J-space/GWT paper; it engages the whole lineage.

- **vs Manifolds / J-space (readout + lens-coordinate patching).** The Jacobian-lens ladder
  gives the first *quantitative* version of the post-training "point of view" shift the
  J-space paper states qualitatively (Instruct cos 0.69 from base; method-driven, decoupled
  from capability), and the **model-randomization** control that *no* paper in the table ran
  passes on the J-lens.

- **vs Introspection (concept injection + verbal report).** Lindsey injects a concept vector
  and asks the model to report it; his self-reported weakness is that injecting the
  **negation** was "comparably effective" (identification may be confabulated). We run the
  injection on OLMo (`modal_introspection.py`) with the controls that debate is missing: a
  lens **manipulation-check** — inject at low layers, read the workspace at higher layers, so
  we can tell "the concept is genuinely in the workspace" from "the model confabulated a
  plausible word" — plus his own **negation** control and a **base-vs-Instruct** split.
  *First pass (`introspection/*.json`):* the manipulation-check is clean (workspace readback of
  the concept: baseline −0.4 → inject +20.8 → negation −17.2 on base), and the **negation
  control is cleaner than Lindsey's** — injecting −concept produces the concept 0% of the time
  (his was "comparably effective"). **But at the strength used the injection forces the token
  into the output** (`"France France France…"`) — that is *steering, not introspection*.
  *v2 (the real test — strength sweep × "asked to report" vs "neutral continuation"):* **at
  every strength that places the concept in the workspace, it appears equally in the asked and
  neutral prompts (introspection-gap ≈ 0, both base and Instruct).** There is no regime where
  the injected concept surfaces *only when the model is asked* — so on OLMo-3 this is steering,
  not introspection, and the manipulation-check rules out a failed injection (readback ~20).
  **A clean negative, on the Reality Check's side, on open weights.** Caveat: OLMo-7B is far
  smaller than the frontier models where introspection was claimed, so this is "no introspection
  in OLMo-7B," not "introspection is fake" — a scale/model-dependence question the open ladder
  could pursue.

- **vs the Reality Check (Singh, Linzen & Ravfogel).** They argue the field lacks "mechanistic
  evidence of a **dissociable second-order process**", the thing that would separate genuine
  introspection from input-driven pattern matching. Our metacognition ladder supplies exactly
  that, on open weights: the base model's workspace **covertly** tracks its own errors
  (AUROC 0.66, beyond output confidence) while its **verbal** self-report is at chance (0.51);
  post-training raises the report to 0.78 (Instruct) / 0.72 (Think) — and only via SFT+DPO, not
  RLVR. The internal state and its reportability **come apart**, and they come apart by training
  stage and method. That is a dissociable second-order process, demonstrated rather than
  asserted — and it reframes "post-training installs self-monitoring (C2)" as *post-training
  makes a pre-existing covert signal reportable*, not creates it.

The through-line: the four papers write a direction and read an effect; the open replication
lets us **verify the direction is doing what the label says** (manipulation-check), **run the
control the author flags** (negation), and **exhibit the dissociation the rebuttal demands**
(covert vs reportable) — the three things the lineage, and its rebuttal, are missing.

---

## References

Verified against arXiv on 2026-07-18.

**Failure modes of interpretability methods**
- Adebayo, J., Gilmer, J., Muelly, M., et al. (2018). *Sanity Checks for Saliency Maps.*
  NeurIPS 2018. [arXiv:1810.03292](https://arxiv.org/abs/1810.03292)
- Jain, S. & Wallace, B. C. (2019). *Attention is not Explanation.* NAACL 2019.
  [arXiv:1902.10186](https://arxiv.org/abs/1902.10186)
- Bolukbasi, T., **Pearce, A.**, Yuan, A., Coenen, A., et al. (2021). *An Interpretability
  Illusion for BERT.* [arXiv:2104.07143](https://arxiv.org/abs/2104.07143)
  — note the second author is also on the manifolds and J-space papers.
- Makelov, A., Lange, G. & Nanda, N. (2023). *Is This the Subspace You Are Looking For? An
  Interpretability Illusion for Subspace Activation Patching.* ICLR 2024.
  [arXiv:2311.17030](https://arxiv.org/abs/2311.17030)

**Lenses**
- nostalgebraist (2020). *interpreting GPT: the logit lens.* LessWrong. *(not on arXiv)*
- Belrose, N., Ostrovsky, I., McKinney, L., Furman, Z., et al. (2023). *Eliciting Latent
  Predictions from Transformers with the Tuned Lens.*
  [arXiv:2303.08112](https://arxiv.org/abs/2303.08112)

**The four papers analysed**
- Gurnee, W., Ameisen, E., Kauvar, I., Tarng, J., Pearce, A., Olah, C. & Batson, J. (2025).
  *When Models Manipulate Manifolds: The Geometry of a Counting Task.* 21 Oct 2025.
  https://transformer-circuits.pub/2025/linebreaks/index.html
- Lindsey, J. (2025). *Emergent Introspective Awareness in Large Language Models.*
  29 Oct 2025. https://transformer-circuits.pub/2025/introspection/index.html ·
  [arXiv:2601.01828](https://arxiv.org/abs/2601.01828)
- Kim, J., Lai, S., Scherrer, N., Agüera y Arcas, B. & Evans, J. (2026). *Reasoning Models
  Generate Societies of Thought.* [arXiv:2601.10825](https://arxiv.org/abs/2601.10825)
- Gurnee, W., Sofroniew, N., Pearce, A., et al. & Lindsey, J. (2026). *Verbalizable
  Representations Form a Global Workspace in Language Models.* 6 Jul 2026.
  https://transformer-circuits.pub/2026/workspace/index.html

**Infrastructure**
- Lin, J. *Neuronpedia — open source interpretability platform.* MIT, created 21 Jun 2023.
  https://github.com/hijohnnylin/neuronpedia · https://www.neuronpedia.org ·
  lenses at https://huggingface.co/neuronpedia/jacobian-lens

**The controlled replication**
- Singh, S., Linzen, T. & Ravfogel, S. (2026). *Can LLMs Introspect? A Reality Check.*
  25 May 2026. [arXiv:2605.26242](https://arxiv.org/abs/2605.26242)
  — tests Llama-3.1-8B/70B, Qwen-2.5-72B, Gemma-3-27B-it.

**Cited from memory, NOT verified in this pass** — check before reuse: Alain & Bengio
(2016) probing classifiers; Zou et al. (2023) representation engineering; Turner et al.
(2023) activation addition; Rimsky et al. (2024) contrastive activation addition; Bricken
et al. (2023) and Cunningham et al. (2023) on SAEs; Vig et al. (2020) and Meng et al.
(2022, ROME) on causal tracing/patching.
