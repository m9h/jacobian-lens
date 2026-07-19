---
category: research
section: background
weight: 40
title: "What each paper controls for"
slide_summary: |
  \scriptsize
  \begin{tabular}{@{}lcccc@{}}
  \toprule
  \textbf{Control} & \textbf{Manif.} & \textbf{Intro.} & \textbf{SoT} & \textbf{J-space} \\
  \midrule
  matched random subspace      & \textbf{Y} & --- & --- & --- \\
  norm-matched random vector   & --- & \textbf{Y} & --- & --- \\
  negative / inverted direction & --- & \textbf{fails} & --- & --- \\
  no-intervention baseline     & --- & \textbf{Y} & \textbf{Y} & \textbf{Y} \\
  \textbf{model randomization} & --- & --- & --- & --- \\
  \textbf{drift null (geometry)} & --- & n/a & n/a & --- \\
  \bottomrule
  \end{tabular}

  \normalsize
  \medskip
  \textbf{Lindsey's introspection paper is the best controlled of the four} --- and its
  weakness is one it reports itself: injecting the \emph{negation} of a concept vector was
  “comparably effective,” words showing “no discernible pattern.”

  \medskip
  \textbf{No paper runs a model-randomization control} (Adebayo et al.\ 2018). Notably,
  a co-author of \emph{An Interpretability Illusion for BERT} is on two of these papers ---
  the expertise is in the room.
---
