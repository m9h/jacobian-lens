---
category: research
section: discussion
weight: 90
title: "Who is allowed to check any of this?"
slide_summary: |
  \textbf{No external party holds activations for Claude, GPT or Gemini.} There is no
  programme to apply to. The published ceiling (METR, May 2026) is black-box plus raw
  chain-of-thought, under NDA, with the lab approving publications --- and Apollo and
  UK AISI got \textbf{under a week} with Sonnet 4.5.

  \medskip
  How did the invited commentators check the J-space claims? \textbf{They didn't.}
  Nanda received “an advance draft” and replicated on \textbf{Qwen3.5-27B}. Dehaene and
  Naccache: \emph{“We suggested to the Anthropic team that they could run exactly the
  same tests\ldots”} --- the originators of the theory could not test it themselves.

  \medskip
  \textbf{And the checkability is borrowed.} Neuronpedia --- an MIT-licensed project
  created June 2023 and run by \emph{one person} --- fits and hosts these lenses, alongside
  DeepMind's and OpenMOSS's SAEs. Anthropic's own HuggingFace org publishes zero models.
  The convergence criterion that is missing from the public chain lives in
  \emph{Neuronpedia's} unreleased wrapper.

  \medskip
  Auditing all 38 published lenses: \textbf{one is defective} --- \texttt{qwen3-32b} is a
  mid-fit checkpoint at $n{=}80$ against a config claiming 615. We used it as a dense
  control and it returned identical J-lens and logit-lens top-5s.
---
