---
category: research
section: results
weight: 80
title: "Result 4 --- two retractions"
slide_summary: |
  \textbf{I reported a sharp emergence threshold.} The ASCII-face readout surfaced at
  Qwen3.5-27B (rank 2) and not at dense Qwen3-14B (rank 164). Then I reported a
  \emph{correction}: it tracked architecture, not scale.

  \medskip
  \textbf{Both were wrong.} The robust version --- the 102-vignette \texttt{association}
  eval, concepts evoked but never named --- gives a smooth monotone rise in which a
  \textbf{dense} 32B beats the 27B hybrid at every rank:

  \medskip
  \begin{tabular}{@{}lrrr@{}}
  \toprule
  \textbf{model} & \textbf{J@1} & \textbf{J@10} & \textbf{J@50} \\
  \midrule
  14B dense & 0.040 & 0.091 & 0.222 \\
  27B hybrid & 0.049 & 0.167 & 0.343 \\
  \textbf{32B dense} & \textbf{0.062} & \textbf{0.208} & \textbf{0.438} \\
  \bottomrule
  \end{tabular}

  \medskip
  The original finding was \textbf{one lucky prompt}. Architecture-dependence is
  \emph{untested}, not established.

  \medskip
  \textbf{This is the point of the exercise.} The discipline the field lacks is not
  cleverness --- it is running the control that kills your own result, and saying so.
---
