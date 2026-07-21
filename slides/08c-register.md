---
category: research
section: results
weight: 84
title: "Result 6 --- one model's workspace holds a suppressed register"
slide_summary: |
  A cheap, no-GPU content probe (divide the embedding norm out through the motor band)
  suggested OLMo-3-7B's workspace holds discourse markers over literal next-tokens. $n{=}1$,
  hand-picked layers. \textbf{So I pre-registered a gate} and ran it band-wide, with a
  1000-permutation null, across 7 cross-family models.

  \medskip
  \textbf{As a general claim it fails: significant in 1 of 7.} I had over-generalised an
  OLMo-specific result by calling it ``the workspace.'' The general line is dropped.

  \medskip
  But OLMo was an $11.6\times$ outlier among $\sim\!1\times$, and it is real and broader than
  discourse. Its workspace band preferentially expresses an informal / negative / profane /
  connective register its \emph{output} layers suppress:

  \medskip
  \scriptsize
  \begin{tabular}{@{}lll@{}}
  \toprule
  \textbf{lexicon} & \textbf{OLMo-3-7B} & \textbf{Qwen3-8B (same tokens+measure)} \\
  \midrule
  discourse & 11.5x \; $p{=}0$ & 1.3x \; n.s. \\
  negative & 12.1x \; $p{=}0$ & 0.0x \; n.s. \\
  profanity & 8.6x \; $p{=}0$ & 4.1x \; marg. \\
  informal & 11.0x \; $p{=}0$ & 0.0x \; n.s. \\
  \bottomrule
  \end{tabular}

  \normalsize
  \medskip
  \textbf{Survives all three free controls:} permutation null; \emph{cross-model flatness}
  (a broken measure would flag Qwen too --- it is flat); and \emph{frequency} (register beats
  frequency-matched neutral words in 5/5 Zipf bins; $\mathrm{corr}$ with Zipf $=-0.15$).

  \medskip
  \textbf{Still $n{=}1$ model.} The open test is the one OLMo uniquely permits: do these
  tokens trace to a distinct slice of Dolma (raw-web / toxic register)? Runnable \emph{free}
  on infini-gram --- a direct test of Hoel's ``just an output transformation.'' Not yet run.
---
