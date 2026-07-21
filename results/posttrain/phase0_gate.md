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
