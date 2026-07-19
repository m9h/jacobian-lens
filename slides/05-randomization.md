---
category: research
section: results
weight: 50
title: "Result 1 --- the lens passes randomization"
slide_summary: |
  Randomize the transformer \emph{blocks}; keep the \textbf{trained} embedding, final norm
  and unembedding --- so token identity stays meaningful and the control cannot pass for
  the wrong reason. Score two ways per layer.

  \medskip
  \begin{tabular}{@{}lrr@{}}
  \toprule
   & \textbf{peak next\_acc} & \textbf{peak echo} \\
  \midrule
  trained & 0.3414 & 0.2936 \\
  \textbf{random blocks} & \textbf{0.0003} & \textbf{0.0016} \\
  \bottomrule
  \end{tabular}

  \medskip
  Trained shows a clean crossover: early layers echo the current token, then echo decays
  as next-token prediction climbs. \textbf{Random blocks read out nothing.}

  \medskip
  \textbf{The J-lens passes.} Its structure requires learned weights. The strong form of
  “it's just the residual stream” is refuted --- and Anthropic never ran this control.
  \emph{Controls should be able to exonerate as well as convict.}

  \medskip
  \tiny Trap: \texttt{model.\_init\_weights()} is a silent no-op in transformers v5. The
  naive control leaves blocks fully trained and reports a confident false PASS.
---
