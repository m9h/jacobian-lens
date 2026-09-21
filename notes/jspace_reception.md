# J-space: links, commentary, and who has replicated it

**Research note, 2026-09-21.** Answers three questions: are our links current, has the paper drawn
published responses, and has anyone replicated or extended it on open models. Short answer to the
last one: **yes, at least five groups, and one of them is doing our niche.**

Evidence tier is marked per item. Where it says *search summary + repo page*, treat as unverified.

## 1. Links — one we were missing

| | status |
|---|---|
| [transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html) | ✅ live (what we cite) |
| [anthropic.com/research/global-workspace](https://www.anthropic.com/research/global-workspace) | ✅ live |
| **[arXiv:2607.15495](https://arxiv.org/abs/2607.15495)** | ★ **we had no arXiv ID recorded.** Add it — arXiv is citable and versioned where the TC thread is not |
| **[External commentary (PDF)](https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf)** | ★ **we did not know this existed.** 8 pages, ~22k words, three solicited commentaries |

## 2. The response literature is *in* the paper's own release

Anthropic solicited and published three external commentaries. This is unusual and worth noting as
a practice: **they shipped their own adversarial review.**

**Stanislas Dehaene & Lionel Naccache** — the originators of the global neuronal workspace model
their paper is named after. Verdict: real parallels, with limits. Their abstract:

> "...although the machine approximates the functional architecture of conscious processing, there
> are still key differences — in its anatomy and its sense of self, and in its lack of a body and
> of an enduring episodic memory — which warrant caution in drawing parallels with the human mind."

★ **They cite Ericsson & Simon (1993) — protocol analysis — directly**, noting human introspection
"is largely restricted to slow serial computations," alongside confabulation (Gazzaniga), choice
blindness (Johansson et al. 2005) and the perception/action dissociation (Aglioti et al. 1995), and
say *"it seems that the J-space suffers from a similar dissociation."* **This is independent
convergence on the framing behind [`self-report-validity`](https://github.com/m9h) — from Dehaene,
not from us.** It is also a strong citation to have for that project.

**Butlin, Shiller, Plunkett & Long (Eleos AI Research)** — co-leads of *Consciousness in Artificial
Intelligence* (2023), i.e. the authors of the indicator framework our Scorecard work is built on,
commenting on whether this moves their own indicators. *(Not yet read in full — flagged.)*

**Neel Nanda (GDM)** — see below; his commentary **contains an original replication**.

## 3. ★ Replications and extensions on open models

**Neel Nanda + MATS scholars Camila Blank and Agam Bhatia — Qwen 3.6 27B.** In the commentary
itself. *(S1: read in the PDF.)*

- **Replicated:** verbal-report causal effect ("weak but positive"), CKA analysis ("somewhat
  similar squares... though less clean"), directed modulation ("moderate success"), and the §A.6
  quantitative evals for **multilingual (probing and causal) and typo**.
- **Did not replicate:** **poetry and arithmetic** — which they attribute to "experimenter error or
  worse model capabilities," not to the method. **Multihop factual recall** was confounded: "swapping
  the answer turned out to strictly dominate," because their multihop dataset had linearly related
  pairs like France/Paris.
- **Scaling test:** also ran it on **Qwen3.5-397B-A17B**, ~1 hour for n=4 prompts on 8×H200.
- His summary judgement: *"a strong validation that J-Space is an important result and a rich domain
  for future work"*, and he wants to replicate it for auditing Gemini.

★ **The cost finding we should act on.** Nanda: *"while the paper averages over n=1000 prompts to
compute their Jacobian, their provided ablations show that much smaller ones work fine, e.g. n=10 is
almost as good, and n=1 is pretty respectable. As cost is O(n · d_model) backward passes, using a
smaller n is a big saving! We used n=25."* **We have been fitting at far higher n.** If n=25
suffices, our matched-budget refit floor and the pending RoPE refit both get dramatically cheaper —
and our 40-prompt ladder fits were less under-powered than we assumed.

**[solarkyle/jspace](https://github.com/solarkyle/jspace) — Gemma-4-12B, preregistered.**
*(S2: repo page read, code not run.)* **This is our niche, occupied.** Self-description: "25k
prompts, frozen probes, public traces, a prospective transfer miss, and an answer-identity
confound." Their finding: a ~300 KB LightGBM classifier over J-lens workspace readouts predicts
Gemma-4-12B's wrong answers **better than the model's own output confidence**, transfers across
datasets *within a task family*, and **fails a preregistered universal-transfer test** — holding on
grounded and retrieval QA, breaking on veracity judgment.

⚠️ **This bears directly on our own metacognition result and may explain it.** We found the covert
error signal does *not* improve best-of-N (probe 0.364 vs logprob 0.356 vs random 0.344, oracle
0.656) and suspected between-item structure wearing within-item clothes
([PITFALLS #23](https://github.com/m9h/spinning-up-in-mech-interp/blob/master/PITFALLS.md)). Their
result is the same shape from the other side: works *across* questions, fails to transfer
universally. And they report an **answer-identity confound** they say "can affect other published
probe results" — **we should check whether ours is exposed before doing anything else with that
line.**

**Other open-model infrastructure** *(S3: search summaries, unverified):*
[`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens) — pre-fitted lenses
for **38 open models**; [`model-organisms-for-real/jacobian-lens`](https://github.com/model-organisms-for-real/jacobian-lens)
PR #1 — demo notebooks, a Figure-27 replication on open weights, adaptations to `olmo2-0425-1B-sft`
and `gemma-3-1b-it`; an [nnsight tutorial](https://nnsight.net/tutorials/mini-papers/jacobian-lens/);
and a [J-lens CKA explorer](https://eliebak.com/viz/jspace-open).

## 4. ★ Our RoPE erratum was filed against Neuronpedia too — and closed

[hijohnnylin/neuronpedia#235](https://github.com/hijohnnylin/neuronpedia/issues/235): *"olmo-3-1025-7b
Jacobian lens was fitted under transformers 5.11.0, which applies YaRN to the wrong layers"* —
**status: Closed.** So the bug @venvoo reported to us is real, independently filed upstream, and
**Neuronpedia has already dealt with theirs while our refit is still pending**
([ROPE_ERRATUM.md](../results/ROPE_ERRATUM.md)). That moves the refit up the queue.

Note also: their issue says **5.11.0**, ours resolves **5.9.0** — different versions, same bug
window. Consistent, and further evidence the report is sound.

## What this changes

1. **Add the arXiv ID** to `docs/technique-lineage.md` and anywhere we cite the TC thread alone.
2. **Refit at n=25, not n≥100.** Nanda's reading of the paper's own ablations makes the pending
   RoPE refit cheap. Verify the ablation claim against §A of the paper first.
3. **Check our metacognition work for the answer-identity confound** solarkyle report.
4. **The Scorecard's competitive position needs updating.** Eleos AI are commenting on J-space
   directly, and a preregistered reliability campaign on Gemma 4 already exists. Neither is the
   standing-adjudication function we propose, but both are closer than anything in the last survey.
5. **Dehaene citing Ericsson & Simon** is the strongest external support the self-report-validity
   project has, and it came from the founder of the theory rather than from us.
