# Draft — a short confirmation for hijohnnylin/neuronpedia #235, if Morgan wants to post it

*Drafted 2026-10-15 by the agent. Not posted. An earlier version of this file (same day) claimed
the September refit was an outlier; that was wrong and is withdrawn — see
`results/ROPE_ERRATUM.md` §"Retraction, 2026-10-15".*

---

**Title:** olmo-3-1025-7b: independent confirmation that the 21 Sep refit is on the corrected
forward (and a number for how much the RoPE bug moved the lens)

Hi Johnny — a data point for #235 from an independent pipeline (our fork of anthropics/jlens on
Modal, B200, transformers 5.19.0, same WikiText-103 protocol).

On the first prompt of the wikitext stream — which both loaders pick — the single-prompt layer-30
identity distance is:

| | your `convergence.csv`, row 1 | ours |
|---|---|---|
| per-layer RoPE (fixed) | 0.242644 (Sept lens) | **0.24290** (stock 5.19.0) |
| YaRN on all 32 layers (the 5.0–5.12 behaviour) | 0.233647 (June lens) | **0.23380** (bug forced back on via `rope_parameters`) |

So your September lens matches a fixed forward and your June lens matches the buggy one, to 0.1%
each, from a different implementation. On the same 8 prompts the bug moves the lens by cosine
0.886 at layer 0 and +3.9% identity distance at layer 30 — consistent with the +4–6% @venvoo
reported. Script: `m9h/jacobian-lens/modal_rope_emulation_test.py`.

Two small things you may already know: the September run stopped at 568 of 1,000 prompts on
`stop_at_delta 0.002` with the running identity distance still drifting about −1e-4 per prompt
(the June lens had 616); and your `fit_lens.py` provenance block is the right design — we had
no such record in our files and spent six weeks believing our own lenses were fitted under the
buggy version when the Modal image had actually resolved 5.14.1. We're adopting your format.
