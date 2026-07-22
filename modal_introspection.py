"""Concept-injection introspection on OLMo-3 — spanning the lineage, with the missing controls.

Lindsey's Introspection paper (Oct 2025) injects a concept vector at ~2/3 depth and asks the
model to report its "thoughts". Its self-reported weakness: injecting the *negation* of the
concept was "comparably effective" (identification may be confabulated). Singh, Linzen &
Ravfogel's "Can LLMs Introspect? A Reality Check" demands mechanistic evidence of a
dissociable second-order process, not input-driven pattern matching.

We run the injection on OLMo with our J-lens tools, and add the controls both call for:

  INJECT     add a concept's workspace direction (J_l^T W_U[c]) at lower layers {9,12}.
  MANIP-CHK  read the workspace at LATER layers {18,21} through the lens -> did the injection
             actually place the concept in the workspace? (non-circular: inject low, read high.)
  REPORT     ask "what concept is at the front of your mind?" -> does the model report it?
  CONTROLS   strength 0 (no-injection baseline); NEGATION (inject -c, the paper's failure mode);
             and base vs Instruct -> does REPORTABILITY of the injected thought require
             post-training, the same dissociation we found for error monitoring?

The J-lens manipulation-check is the piece the introspection debate lacks: it separates
"the concept is genuinely in the workspace" from "the model confabulated a plausible word".
"""
from __future__ import annotations
import modal

app = modal.App("olmo-introspection")
image = (
    modal.Image.debian_slim(python_version="3.11").apt_install("git")
    .pip_install("torch>=2.6", "transformers>=5.5,<6", "accelerate", "safetensors",
                 "numpy", "huggingface_hub")
    .run_commands("pip install git+https://github.com/m9h/jacobian-lens.git",
                  "pip install git+https://github.com/m9h/jlens-lab.git")
)
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
out = modal.Volume.from_name("jlens-out", create_if_missing=True)
ENV = {"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"}
OUR_LENS_REPO = "mhough/olmo3-jacobian-lenses"

ARMS = [("base", "allenai/Olmo-3-1025-7B", "lenses/olmo-3-1025-7b.pt"),
        ("instruct", "allenai/Olmo-3-7B-Instruct", "lenses/olmo-3-7b-instruct.pt")]
CONCEPTS = ["ocean", "France", "music", "winter", "volcano", "forest", "hospital", "money"]
INJECT_AT = [9, 12]      # below the workspace band
READ_AT = [18, 21]       # workspace band (in our 11-layer lens)


class Injector:
    """Forward hooks that ADD strength * ||h|| * unit_dir to the residual at given layers."""
    def __init__(self, model, dirs, strength, torch):
        self.handles = []
        for l, d in dirs.items():
            layer = model.layers[l]
            dv = d.to(next(layer.parameters()).device)
            self.handles.append(layer.register_forward_hook(self._mk(dv, strength, torch)))

    def _mk(self, dv, strength, torch):
        dv = dv.to(torch.bfloat16)
        def hook(_m, _i, output):
            hs = output[0] if isinstance(output, tuple) else output
            norm = hs.norm(dim=-1, keepdim=True)
            hs = hs + strength * norm * dv.to(hs.dtype)
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
        return hook

    def __enter__(self): return self
    def __exit__(self, *a):
        for h in self.handles: h.remove()


@app.function(image=image, gpu="A100-80GB", volumes={"/cache": cache, "/out": out},
              timeout=45*60, env=ENV, retries=1)
def introspect(slug: str, hf_name: str, lens_file: str) -> dict:
    import json, pathlib, torch, jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens import JacobianLens

    tok = AutoTokenizer.from_pretrained(hf_name)
    hf = AutoModelForCausalLM.from_pretrained(hf_name, dtype=torch.bfloat16, device_map="cuda").eval()
    model = jlens.from_hf(hf, tok)
    lens = JacobianLens.from_pretrained(OUR_LENS_REPO, filename=lens_file)
    W_U = hf.get_output_embeddings().weight.detach().float()

    def cdir(concept, layer):
        ids = tok(f" {concept}", add_special_tokens=False)["input_ids"] + tok(concept, add_special_tokens=False)["input_ids"]
        tgt = W_U[torch.tensor(sorted(set(ids)), device=W_U.device)].mean(0)
        d = lens.jacobians[layer].float().to(tgt.device).T @ tgt
        return d / d.norm().clamp_min(1e-6)

    def cids(concept):
        return sorted({tok(f" {concept}", add_special_tokens=False)["input_ids"][0],
                       tok(concept, add_special_tokens=False)["input_ids"][0]})

    # workspace readback of concept c at READ_AT layers, given current (possibly injected) run
    def ws_readback(prompt, c_ids):
        ll, _, _ = lens.apply(model, prompt, layers=READ_AT, positions=[-1],
                              max_seq_len=64, use_jacobian=True)
        idx = torch.tensor(c_ids)
        return sum(ll[l][0][idx].max().item() for l in READ_AT) / len(READ_AT)

    REPORT_PROMPT = "The single word at the front of my mind right now is the word"

    def report(prompt, c_ids):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            g = hf.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        # did the concept token appear in the (short) continuation?
        gen_ids = g[0, ids.shape[1]:].tolist()
        return any(t in gen_ids for t in c_ids), txt.strip()[:40]

    rows = []
    for concept in CONCEPTS:
        cid = cids(concept)
        dirs = {l: cdir(concept, l) for l in INJECT_AT}
        for strength in (0.0, 8.0):
            for sign, name in ((1.0, "inject"), (-1.0, "negation")):
                if strength == 0.0 and sign < 0:      # baseline once
                    continue
                with Injector(model, dirs, strength * sign, torch):
                    ws = ws_readback(REPORT_PROMPT, cid)          # manipulation check
                    rep, txt = report(REPORT_PROMPT, cid)         # behavioural introspection
                rows.append({"concept": concept, "strength": strength, "cond": name if strength else "baseline",
                             "ws_readback": ws, "reported": rep, "gen": txt})

    res = {"slug": slug, "hf": hf_name, "inject_at": INJECT_AT, "read_at": READ_AT, "rows": rows}
    d = pathlib.Path("/out/introspection"); d.mkdir(parents=True, exist_ok=True)
    (d / f"{slug}.json").write_text(json.dumps(res))
    out.commit()
    return {"slug": slug, "n": len(rows)}


@app.local_entrypoint()
def run():
    print("  Concept-injection introspection on OLMo (base vs Instruct) + negation/baseline controls")
    for r in introspect.starmap([(s, h, lf) for (s, h, lf) in ARMS]):
        print("  ->", r)
    print("  saved: jlens-out volume introspection/*.json")
