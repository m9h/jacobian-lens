---
category: research
section: results
weight: 60
title: "Result 2 --- the metric rewards noise"
slide_summary: |
  The paper's lens-quality score is \textbf{minimum rank over ${\sim}35$ layers}. That
  hands a diffuse readout \emph{one lottery ticket per layer}. A confident lens correctly
  reading the content puts an unrelated word far down at \emph{every} layer.

  \medskip
  Qwen3.5-27B, at the \texttt{\^{}} of the ASCII face, on \textbf{Anthropic's own lens}:

  \medskip
  \begin{tabular}{@{}lcl@{}}
  \toprule
  \textbf{lens} & \textbf{rank(“nose”)} & \textbf{top-5} \\
  \midrule
  J-lens & \textbf{2} & \texttt{smile, nose, '\^{}, noses, grin} \\
  logit lens & 5 & \texttt{\textbackslash n, \textasciitilde, .., -, N} \\
  \bottomrule
  \end{tabular}

  \medskip
  Comparable scores. One has understood it is looking at a face; the other emits
  punctuation. \textbf{\texttt{pass@k} cannot tell them apart.}

  \medskip
  This is why the logit lens “beat” the J-lens on mean \texttt{pass@10} at 4B and 8B in
  our first pass --- an artifact of the metric, which we nearly reported as a refutation.
---
