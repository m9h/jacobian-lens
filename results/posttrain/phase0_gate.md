> ## ✅ ERRATUM RETRACTED FOR THESE LENSES (2026-10-15)
> The banner below was wrong about **our** lenses. They were fitted on Modal under transformers
> **5.14.1**, which already has the per-layer RoPE fix; the 5.9.0 figure came from `uv.lock`, which the
> Modal image never used (verified by probing the cached image). The numbers here were never affected
> by the bug. Neuronpedia's June lens was. See ROPE_ERRATUM.md §"Retraction, 2026-10-15". The
> original banner is kept for the record:
>
> ## ⚠️ ERRATUM (2026-09-05)
> These results were computed from lenses fitted under **transformers 5.9.0**, which applied YaRN
> `rope_scaling` to **all 32 OLMo-3 layers** instead of the 8 full-attention ones — so **24 of 32
> layers ran the wrong RoPE**. Numbers here will move on refit. See
> [ROPE_ERRATUM.md](../ROPE_ERRATUM.md). Reported externally by @venvoo.

# Phase 0 gate: does the workspace hold discourse markers? — FAILED as a general claim

The n=1 observation (OLMo-3-7B's workspace preferentially expresses discourse/contrastive
markers over literal next-tokens) was tested band-wide, with a 1000-permutation null,
across 8 ungated cross-family models. **Pre-registered gate: significant in a majority.**

    model            enrichment    p
    qwen3-1.7b          0.65x     0.81
    qwen3-4b            0.98x     0.61
    qwen3-8b            1.30x     0.37
    qwen3-14b           1.30x     0.37
    qwen2.5-7b-it       0.33x     0.96
    gpt2-small          0.36x     0.94
    olmo-3-1025-7b     11.62x     0.00   <- lone outlier
    (pythia-70m: too few layers)

**1/7 significant. The gate FAILS.** "The workspace holds discourse markers" is NOT a
general property — I over-generalised an OLMo-specific result by calling it "the
workspace." The general content line is dropped.

The OLMo outlier (11.6x, everything else ~1x) is either the real, OLMo-specific effect
that motivates the whole OLMo program, or a model-specific artifact. Being checked with
four free controls (tokenizer confound, motor-band denominator validity, band-choice
robustness, and what the top tokens actually are) before it is allowed to reframe
anything. A lone 10x outlier among 1x is treated as suspect until proven.
