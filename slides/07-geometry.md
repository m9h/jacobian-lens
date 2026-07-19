---
category: research
section: results
weight: 70
title: "Result 3 --- the tripartite figure is mostly drift"
slide_summary: |
  The sensory/workspace/motor block structure is the one part Hoel concedes is \emph{not}
  baked in by construction. Build a \textbf{distance-only null}: entries depending solely
  on $|i-j|$, same decay profile, \emph{zero blocks by construction}.

  \medskip
  \scriptsize
  \begin{tabular}{@{}lrrr@{}}
  \toprule
  \textbf{model} & \textbf{real} & \textbf{null} & \textbf{excess} \\
  \midrule
  qwen3-1.7b & 0.117 & 0.092 & +0.025 \\
  qwen3-4b & 0.078 & 0.062 & +0.016 \\
  qwen3-8b & 0.283 & 0.249 & +0.033 \\
  qwen3-14b & 0.293 & 0.267 & +0.026 \\
  gpt-oss-20b & 0.210 & 0.157 & \textbf{+0.053} \\
  qwen3.5-27b & 0.267 & 0.217 & \textbf{+0.050} \\
  \bottomrule
  \end{tabular}

  \normalsize
  \medskip
  \textbf{The null recovers 79--91\%.} Excess doubles at ${\ge}20$B but stays small.
  Raw blockiness rising $0.08\!\to\!0.29$ with scale is the \emph{decay profile
  steepening}, not blocks appearing.

  \medskip
  \textbf{Both sides overclaimed}: Anthropic on the sharpness, Hoel on the absence ---
  his substitution argument generalises from small models, the regime where there is
  nothing to find.
---
