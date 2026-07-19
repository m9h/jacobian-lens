---
category: research
section: introduction
weight: 10
title: "The claim"
slide_summary: |
  \textbf{July 2026.} Anthropic: \emph{“Verbalizable Representations Form a Global
  Workspace in Language Models.”} Claude contains a “J-space” functionally analogous
  to the global workspace of Baars and Dehaene. 1.2M views in a day.

  \medskip
  Reported properties: reportability, controllability, causal role in reasoning,
  flexible reuse --- and \textbf{limited scope}: ablating the J-space destroys multi-step
  reasoning and summarisation while sparing fluency, sentiment, and factual recall.

  \medskip
  \textbf{The method.} A Jacobian lens:
  $$\mathrm{lens}_l(h) = \mathrm{unembed}(J_l\,h), \qquad J_l = \mathbb{E}\left[\partial h_{\mathrm{final}} / \partial h_l\right]$$
  Apache-2.0, plus 38 fitted lenses for open models.
---
