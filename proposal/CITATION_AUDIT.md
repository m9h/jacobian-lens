# Citation audit — what the proposal's claims actually rest on

The ACL 2026 auditability call proposes **source-based auditing**: dependency chains showing which
claims hold up which other claims. This applies it to our own proposal, because a document arguing
that the field does not check its sources should be able to say how well it checked its own.

**Evidence depth, worst to best:**

| tier | meaning |
|---|---|
| ⬛ **S3 — search summary** | I never opened the source. A search engine's paraphrase, or a subagent's report. |
| 🟨 **S2 — abstract / landing page** | I fetched the paper's abstract page or a lab's public page, not the paper. |
| 🟩 **S1 — full text** | I read the paper body. |
| ✅ **S0 — verified by us** | We ran it, or fetched the primary artifact directly (API, repo metadata). |

---

## ⬛ S3 — load-bearing claims resting on a search summary

**These are the ones to fix before submission.** Each is a specific quantitative or quoted claim
about someone else's work that I have only thirdhand.

| claim in the proposal | status |
|---|---|
| *"Anthropic's circuit tracing on Claude 3.5 Haiku produced satisfying insight for about a quarter of tested prompts"* | ⬛ **A specific quantitative claim about another lab's flagship result, taken from a search-result paraphrase.** I have not found the primary source. **Do not submit as stated.** |
| *"DeepMind's months-long Chinchilla circuit analysis produced a brittle, partial explanation"* | ⬛ same paraphrase, no primary source located |
| *"DeepMind has publicly shifted from 'ambitious reverse-engineering' to 'pragmatic interpretability'"* | ⬛ quoted phrases I never sourced to a document |
| MIB's *"SAE features are not better than neurons"* | ⬛ quoted from a search summary of the paper. The **claim** is corroborated independently by Transluce (S1), but **the quotation is not verified.** |
| MIT Tech Review "2026 Breakthrough Technology" | ⬛ trivially checkable, not checked |
| NDIF **$9M NSF**; Goodfire **$50M Series A**; Open Phil ~$336M / $46–50M yr | ⬛ search summaries; plausible, unverified |
| All ARC numbers — DiARC 0.63%, ARChitects 50.5 vs 51.5, Greenblatt 50%, BARC +13.75, SOAR c=1000, the 2026 leaderboard (98% / 92.5% / 30.2%), Duck Harness, the ~68-point gap | ⬛ **from our research subagent's report.** Well-sourced *in that report* with links, but I did not open the underlying papers. The report itself lists what it could not verify. |
| *"Anthropic's own report: no static transduction solution above 10%"* | ⬛ same |

## 🟨 S2 — abstract or landing page only

| claim | note |
|---|---|
| *The Dead Salmons of AI Interpretability* (2512.18792) | abstract page. **The framing device of the whole proposal.** The characterisation of which methods fail on random nets comes from the abstract; I have not seen its experiments or numbers. |
| *Make MI Auditable* (2606.00033) — the three mechanisms | abstract page. The mechanisms and "position paper, nothing implemented" are from the abstract. |
| ~~Diplomacy — *More Victories, Less Cooperation*~~ | ✅ **RESOLVED — and the claim was wrong.** See below. |
| Tracr (2301.05062), TracrBench (2409.13714) | search summaries; the mechanism is well known but unread here |
| AI Agent Behavioral Science (Nature HSSC); Machine Psychology | search summaries; the "moving beyond architectural interpretability" quote is unverified |
| AgentTrace (2602.10133) | search summary |
| Model Organism Lottery; Activation Oracles blind-spots (2607.23379); Wang et al. rLLC (2410.02984) | search summaries (curriculum, not proposal) |
| Martian $1M prize; Decode Research funders; TSG Lab publications | fetched their own public pages — primary, but self-descriptive |
| Nisbett & Wilson (1977); Ericsson & Simon | ⬛/🟨 **from my own knowledge, not fetched this session.** Well-established, but the specific bridge to CoT is my argument, not a citation. |

## 🟩 S1 — read in full

| source | what it supports |
|---|---|
| Transluce, *Circuits Are Sparse in the Neuron Basis* | the neuron-basis result, ~100×, RelP, three reproduced case studies |
| **ABC**, *Rigorous Agentic Benchmarks* (2507.02825v2) | τ-bench 38%, SWE-Lancer 100%, SWE-bench-Verified 24% of top-50, CVE-Bench 33%, and the checklist's structure. **Reading this corrected an attribution error and softened a claim.** |

## ✅ S0 — verified by us this session

| claim | how |
|---|---|
| Rung 5's attention-sink bug: activation ~123 at position 0, content max 16.6, mean 0.029; 28× over-steer; steering ladder 0/5, 0/5, 5/5 | ran it |
| Gate result: +3.24 vs +0.71 and +1.37; corr +0.459 among active positions | ran it |
| All 6 rungs + gate pass from a clean venv on torch 2.13 / transformers 5.14 | ran it |
| Diplomacy paper has 13 citations, **zero** touching faithfulness/CoT/interpretability | Semantic Scholar API |
| SAELens → `decoderesearch/SAELens`; circuit-tracer → `decoderesearch/`; lucent last pushed 2025-03-21; tuned-lens 2025-08-07 | GitHub API |
| ARENA deep link 404; OREL 404; Microscope 503; MIB leaderboard live | HTTP status |
| Our own results — SoT reversal, J-lens/NLA 42×, OLMo ~10×, Adebayo pass, three retractions | our published artifacts |

---

---

## Resolved: the Diplomacy claim was wrong

**Checked 2026-08-04 by reading the ACL PDF directly.** The proposal had described an *ablation*:
communication switched off, score barely moves. **No such experiment exists in the paper.**

What the paper does: AMR-maps stated intentions onto actual moves across 24 games / ~27,000
messages, with human deception annotation (318 annotated as lies, 1,167 perceived). What it
concludes: Cicero's communication is *"more transactional, relying on its optimal strategy rather
than the alliance building which is the hallmark of top human players"*; humans reliably identify
it; it is **less** deceptive and persuasive than humans. It states the attribution question as
**open**: *"it is unclear if Cicero's success is due to its use of natural language or its
strategic model."*

**Provenance of the error.** The sentence *"changing inputs related to communication did not
significantly impact its high score"* is from a **USC ISI press page**, not the paper. Reading it
as an ablation was my inference on top of a paraphrase.

**Second lesson, and the sharper one.** An automated summariser reading the same PDF reported the
*opposite* — that removing communication significantly dropped the win rate. Also absent from the
paper. **Two secondary sources, two contradictory wrong answers, and only the primary text
settles it.** A summary that agrees with your prior is not evidence; a summary that contradicts it
is not evidence either.

---

## Round 1 of reads — results (2026-08-04)

Four sources promoted from ⬛/🟨 to 🟩 by reading the PDFs. **One claim was wrong; three held.**

| source | verdict |
|---|---|
| **Diplomacy** (ACL 2024) | ❌ **WRONG — corrected.** No ablation exists. See above. |
| **Dead Salmons** (2512.18792) | ✅ **Verbatim accurate.** Méloux, Dirupo, Portet & Peyrard (Grenoble Alpes; Dirupo at Icahn/Mount Sinai), 21 Dec 2025. The quoted list — "feature attribution, probing, sparse auto-encoding, and even causal analyses" on randomly initialized networks — is exact, as is the statistical-estimator prescription and the identifiability point. **Bonus material we did not have:** their own Figure 1 runs a *minimal* dead-salmon artifact — a **randomly initialized BERT**, 300 IMDb sentences, where principal components correlate spuriously with sentiment and a probe reaches nontrivial cross-validated accuracy. And the primary source for SAEs-on-random-nets is **Heap et al. (2025)**, which we had only via ARENA's paraphrase. |
| **MIB** (2504.13151) | ✅ **Verbatim accurate, with a needed qualifier.** Full sentence: *"For causal variable localization, we find that the supervised DAS method performs best, while SAE features are not better than neurons, i.e., non-featurized hidden vectors."* So it is the **causal-variable-localization track**, where "neuron" means a single dimension of a hidden vector — not a general statement that SAEs lose to neurons. Transluce's result concerns *circuits* on MLP activations. Both find learned dictionaries failing to beat the raw basis, but **in different tracks**; "two unrelated methods, same conclusion" now carries that qualifier. Authors include **David Bau** (NDIF) and **Michael Hanna** (Decode Research). |
| **Product of Experts** (2505.07859) | ✅ **Exact.** Franzen, Disselhoff & Hartmann. Table: Llama-3.2-3B 14.9%→61.4%; NeMo-Minitron-8B 18.3%→71.6%. Harness on the frozen 8B **+53.3**; 3B→8B swap at baseline **+3.4**; at the end **+10.2**. All three as claimed. 71.6% = 286.5/400 on the **public** eval set, SOTA among publicly available approaches. |

**Read-through rate so far: 1 error in 4 checks**, and the error was in the claim I liked best.

---

## Round 2 — ARC leaderboard numbers verified against the primary source (2026-08-04)

Fetched ARC Prize's own leaderboard JSON (`arcprize.org/media/data/leaderboard/v{1,2,3}.json`,
generated 2026-07-31), which is the authority behind every number in the opening section.

| our claim | official | |
|---|---|---|
| ARC-AGI-1: 98.0% Gemini 3.1 Pro | **98.0%** (human panel 98.0%) | ✅ |
| ARC-AGI-2: 92.5% GPT-5.6 Sol | **92.5%** Sol (Max); human panel 100% | ✅ |
| ARC-AGI-3: 30.2% Claude Opus 5 | **30.2%** Opus 5 (High) — ~4× the next model (Sol Max 7.8%) | ✅ |

**All three exact.** This promotes them from ⬛ (subagent report) to ✅, and materially raises
confidence in the rest of that report, though it does not verify it item by item.

**A claim that did NOT verify.** Good Start Labs' write-up states OpenAI tripled GPT-5.6 Sol on
ARC-AGI-3 from **13.3% → 38.3%** via two API settings (retained reasoning + compaction), at ~6×
fewer output tokens. No GPT-5.6 Sol tier on the official semi-private board is near either figure
(Max 7.8 / xHigh 7.0 / High 2.1). Their numbers are presumably on the **public preview**
environments, which are a different set. **Do not cite as an ARC-AGI-3 score.** The *structure* of
the claim — same weights, two settings, large gain, lower cost — would be the single best
harness-vs-model datapoint available if sourced to OpenAI directly, so it is worth chasing.

**★ A parsing error of our own, caught by a control.** The leaderboard `score` field is a
**fraction**, not a percentage. Read naively, ARC-AGI-3 appears to show Opus 5 at "0.3%" and I was
one step from filing a correction claiming our own proposal was wrong. It was caught by
sanity-checking v1 against the independently known 98% figure — a positive control on the
*parsing*, not on the claim. Same lesson as the rung-5 attention sink: **the number that looks
empirical is exactly where an artifact hides.**

## What to do before submission

1. **Cut or source the ⬛ claims still in the first table.** The Anthropic "quarter of prompts" and the DeepMind quotes remain the highest-risk items: specific, quotable, about named labs, unsourced. A reviewer from either lab would notice.
2. ~~Read the Diplomacy paper~~ — done, claim was wrong, corrected.
3. ~~Read the dead salmon paper~~ — done, accurate.
4. ~~Spot-check ARC numbers~~ — done on the load-bearing one; the rest of the subagent's report gains credibility but is not individually verified.

Nothing here is known to be false. The point is that **"I could not find a problem" and "I checked" are different claims**, and this proposal has been making the second while doing the first.
