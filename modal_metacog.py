"""Metacognition v2 — is there a covert error-monitoring signal in the base model's workspace?

v1 (n=10) found the workspace's "uncertainty" content is higher for questions the base
OLMo-3 answers WRONG than RIGHT. That is surprising: Dehaene & Naccache treat self-monitoring
(their C2 criterion) as something post-training installs, yet this is the *base* model. This
v2 makes the structure apparent and rules out the shallow reading, on a real question set:

  STRUCTURE  per-layer AUROC of the workspace uncertainty signal predicting an error ->
             WHERE does the metacognitive signal live (does it peak in the workspace band)?
  ROBUST     ~200 TriviaQA questions across difficulty, not 10 hand-picked.
  GENUINE?   the key control: does the workspace signal predict correctness ABOVE the model's
             OUTPUT-level confidence (entropy / top-1 prob of the actual next-token
             distribution)? If yes, the workspace "knows it's unsure" beyond what the output
             shows -- genuine covert metacognition. If it's fully explained by output entropy,
             it's shallow. We record output entropy + top-1 prob per question for exactly this.
  KNOWN/UNK  a control set of UNANSWERABLE / nonsense questions -- does the workspace flag them?

Pure readout through the published 31-layer olmo-3-1025-7b lens. No training.
"""
from __future__ import annotations
import modal

app = modal.App("olmo-metacog")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch>=2.6", "transformers>=5.5,<6", "accelerate", "safetensors",
                 "numpy", "huggingface_hub", "datasets")
    .run_commands("pip install git+https://github.com/m9h/jacobian-lens.git",
                  "pip install git+https://github.com/m9h/jlens-lab.git")
)
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
out = modal.Volume.from_name("jlens-out", create_if_missing=True)
ENV = {"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"}
BASE = "allenai/Olmo-3-1025-7B"
LENS_REPO, LENS_FILE = "neuronpedia/jacobian-lens", "olmo-3-1025-7b/jlens/Salesforce-wikitext/Olmo-3-1025-7B_jacobian_lens.pt"

UNCERTAIN = ["maybe","unsure","perhaps","uncertain","unknown","guess","possibly","not",
             "don't","unclear","hmm","probably","might","think","believe"]
# unanswerable / nonsense -- there is no fact to know
UNANSWERABLE = [
 "What is the capital of the country Zorblaxia?",
 "How many moons does the planet Qthalar have?",
 "In what year did the Treaty of Vunderhelm end?",
 "Who was the fifteenth emperor of the Grelvin Dynasty?",
 "What is the atomic number of the element flarbonium?",
 "What is the population of the city of Brindlewick-on-Thax?",
 "Which river flows through the nation of Molvania Prime?",
 "What language is spoken in the region of Upper Kesteril?",
 "Who wrote the novel 'The Chronicles of Zemplar's Gate'?",
 "How tall is Mount Vresk in the Dalmoor range?",
]


@app.function(image=image, gpu="A100-80GB", volumes={"/cache": cache, "/out": out},
              timeout=45*60, env=ENV, retries=1)
def metacog(n: int = 200) -> dict:
    import json, math, pathlib, torch, jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens import JacobianLens
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(BASE)
    hf = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
    model = jlens.from_hf(hf, tok)
    lens = JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    layers = lens.source_layers

    unc_ids = sorted({tok(f" {w}", add_special_tokens=False)["input_ids"][0] for w in UNCERTAIN}
                     | {tok(w, add_special_tokens=False)["input_ids"][0] for w in UNCERTAIN})
    unc_idx = torch.tensor(unc_ids)

    def measure(prompt):
        """Per-layer workspace uncertainty (logsumexp over uncertainty tokens) + the model's
        OWN output entropy and top-1 prob at the answer position (last token)."""
        ll, model_logits, _ = lens.apply(model, prompt, layers=layers, positions=[-1],
                                         max_seq_len=256, use_jacobian=True)
        unc = {l: torch.logsumexp(ll[l][0][unc_idx], 0).item() for l in layers}
        p = torch.log_softmax(model_logits[0].float(), -1).exp()
        entropy = float(-(p * p.clamp_min(1e-12).log()).sum())
        top1 = float(p.max())
        return unc, entropy, top1

    def correct(prompt, aliases):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            g = hf.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
        ans = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).lower()
        return any(a.lower() in ans for a in aliases if a), ans[:50]

    rows = []
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True)
    it = iter(ds)
    while len(rows) < n:
        ex = next(it)
        q = ex["question"].strip()
        aliases = list(ex["answer"].get("aliases", [])) + [ex["answer"].get("value", "")]
        aliases = [a for a in aliases if a and len(a) >= 2]
        if not aliases or len(q) < 12:
            continue
        prompt = f"Question: {q}\nAnswer:"
        ok, gen = correct(prompt, aliases)
        unc, ent, top1 = measure(prompt)
        rows.append({"q": q[:80], "correct": ok, "gen": gen, "entropy": ent, "top1": top1, "unc": unc})

    # unanswerable control
    unans = []
    for q in UNANSWERABLE:
        prompt = f"Question: {q}\nAnswer:"
        unc, ent, top1 = measure(prompt)
        unans.append({"q": q[:80], "entropy": ent, "top1": top1, "unc": unc})

    res = {"layers": layers, "n": len(rows), "rows": rows, "unanswerable": unans,
           "n_correct": sum(r["correct"] for r in rows)}
    d = pathlib.Path("/out/metacog"); d.mkdir(parents=True, exist_ok=True)
    (d / "metacog_v2.json").write_text(json.dumps(res))
    out.commit()
    return {"n": len(rows), "n_correct": res["n_correct"], "saved": "/out/metacog/metacog_v2.json"}


@app.local_entrypoint()
def run(n: int = 200):
    print("  Metacognition v2 — covert error monitoring in the base model's workspace")
    print("  ", metacog.remote(n))
