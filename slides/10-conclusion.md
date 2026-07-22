---
category: research
section: conclusion
weight: 100
title: "Where this leaves the claim"
slide_summary: |
  \textbf{What survives.} The J-lens is a real instrument: it passes randomization, and it
  reads content at mid-layers that a logit lens cannot. The ablation dissociation --- fluency
  and recall spared, multi-step reasoning destroyed --- is a genuine finding.

  \medskip
  \textbf{What does not.} The sharp tripartite geometry is mostly smooth drift. The
  lens-quality metric rewards noise. The emergence story was one prompt. The ``structured
  point-of-view shift'' from post-training is a low rank $dJ$ inherits from $J$. And the paper
  itself concedes the architecture: \emph{“no obviously separable input processors”};
  broadcast “within a single feedforward pass rather than through recurrent loops.”

  \medskip
  \textbf{Two positive results, both on fully open weights.} (1) On the OLMo-3 ladder,
  post-training reshapes the J-space ${\sim}31$\% (Instruct) while capability stays flat ---
  a viewpoint shift decoupled from prediction, method-driven and domain-invariant. Claim 6
  supported on artifacts anyone can rerun. (2) OLMo-3-7B's workspace holds a suppressed
  informal/charged register surviving three controls --- the one claim whose data is open
  enough to trace.

  \medskip
  On Butlin \& Long's scorecard that is \textbf{GWT-1 and GWT-3 given away}. What remains
  is a capacity-limited, causally-central, top-down-gated bottleneck. That is a real object.
  \emph{Whether it is a global workspace is a different question.}

  \medskip
  \textbf{The generalisable finding.} Two independent groups --- no shared authors, neither
  citing the other --- made the same move and shipped the same missing control, and both
  headline claims dissolved under it. The techniques are mature; the failure literature
  exists; the controls are not being run.

  \medskip
  \small All results reproducible: \texttt{github.com/m9h/jacobian-lens} $\cdot$
  tooling: \texttt{github.com/m9h/jlens-lab}
---
