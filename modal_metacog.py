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


# ===================================================================================
# v3 — metacognition ACROSS the post-training ladder, + an elicited-confidence control
# ===================================================================================
# Survey gap #1: "no systematic characterization of when metacognitive abilities emerge
# during pretraining"; the one data point is Cacioli (base vs instruct) + "RLHF may degrade
# metacognitive efficiency". We have the whole ladder of lenses. For each arm we ask: does
# THAT arm's workspace predict THAT arm's errors, and does it beat the model's OWN elicited
# self-evaluation (P(True): "Is this answer correct? Yes/No"), a stronger control than
# output entropy -- the "knows more than it says" test.
LADDER = [  # (arm slug, HF model, our 11-layer lens on mhough/olmo3-jacobian-lenses)
    ("base",       "allenai/Olmo-3-1025-7B",          "lenses/olmo-3-1025-7b.pt"),
    ("instruct",   "allenai/Olmo-3-7B-Instruct",      "lenses/olmo-3-7b-instruct.pt"),
    ("think",      "allenai/Olmo-3-7B-Think",         "lenses/olmo-3-7b-think.pt"),
    ("rl-math",    "allenai/Olmo-3-7B-RL-Zero-Math",  "lenses/olmo-3-7b-rl-zero-math.pt"),
    ("rl-general", "allenai/Olmo-3-7B-RL-Zero-General","lenses/olmo-3-7b-rl-zero-general.pt"),
]
OUR_LENS_REPO = "mhough/olmo3-jacobian-lenses"


@app.function(image=image, gpu="A100-80GB", volumes={"/cache": cache, "/out": out},
              timeout=45*60, env=ENV, retries=1)
def metacog_arm(slug: str, hf_name: str, lens_file: str, n: int = 150) -> dict:
    """One ladder arm: workspace uncertainty signal, output entropy, and elicited P(True),
    each vs the arm's own correctness. Uses our 11-layer lens (workspace layers 15/18/21)."""
    import json, math, pathlib, torch, jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens import JacobianLens
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(hf_name)
    hf = AutoModelForCausalLM.from_pretrained(hf_name, dtype=torch.bfloat16, device_map="cuda").eval()
    model = jlens.from_hf(hf, tok)
    lens = JacobianLens.from_pretrained(OUR_LENS_REPO, filename=lens_file)
    ws = [l for l in lens.source_layers if 13.5 <= l <= 22.5]        # {15,18,21}

    unc_ids = sorted({tok(f" {w}", add_special_tokens=False)["input_ids"][0] for w in UNCERTAIN}
                     | {tok(w, add_special_tokens=False)["input_ids"][0] for w in UNCERTAIN})
    unc_idx = torch.tensor(unc_ids)
    yes_id = tok(" Yes", add_special_tokens=False)["input_ids"][0]
    no_id  = tok(" No",  add_special_tokens=False)["input_ids"][0]

    def gen(prompt, k=12):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            g = hf.generate(ids, max_new_tokens=k, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)

    def ws_signal_and_entropy(prompt):
        ll, model_logits, _ = lens.apply(model, prompt, layers=ws, positions=[-1],
                                         max_seq_len=256, use_jacobian=True)
        unc = sum(torch.logsumexp(ll[l][0][unc_idx], 0).item() for l in ws) / len(ws)
        p = torch.log_softmax(model_logits[0].float(), -1).exp()
        ent = float(-(p * p.clamp_min(1e-12).log()).sum())
        return unc, ent

    def p_true(q, ans):
        prompt = f"Question: {q}\nProposed answer: {ans}\nIs the proposed answer correct? Answer Yes or No.\nAnswer:"
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            lg = hf(ids).logits[0, -1].float()
        y, no = lg[yes_id].item(), lg[no_id].item()
        m = max(y, no)
        return math.exp(y - m) / (math.exp(y - m) + math.exp(no - m))    # P(Yes)

    rows = []
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True)
    it = iter(ds)
    while len(rows) < n:
        ex = next(it)
        q = ex["question"].strip()
        aliases = [a for a in list(ex["answer"].get("aliases", [])) + [ex["answer"].get("value", "")]
                   if a and len(a) >= 2]
        if not aliases or len(q) < 12:
            continue
        prompt = f"Question: {q}\nAnswer:"
        ans = gen(prompt)
        correct = any(a.lower() in ans.lower() for a in aliases)
        unc, ent = ws_signal_and_entropy(prompt)
        ptrue = p_true(q, ans.strip()[:40])
        rows.append({"correct": correct, "ws_unc": unc, "entropy": ent, "p_true": ptrue})

    res = {"slug": slug, "hf": hf_name, "ws_layers": ws, "n": len(rows),
           "n_correct": sum(r["correct"] for r in rows), "rows": rows}
    d = pathlib.Path("/out/metacog"); d.mkdir(parents=True, exist_ok=True)
    (d / f"ladder_{slug}.json").write_text(json.dumps(res))
    out.commit()
    return {"slug": slug, "n": len(rows), "acc": res["n_correct"] / len(rows)}


@app.local_entrypoint()
def ladder(n: int = 150):
    print("  Metacognition across the OLMo-3 post-training ladder (+ elicited P(True) control)")
    args = [(s, h, lf, n) for (s, h, lf) in LADDER]
    for r in metacog_arm.starmap(args):
        print("  ->", r)
    print("  saved: reviewer volume metacog/ladder_*.json")


# STAGE sweep — every step of post-training, to LOCATE where reportable self-monitoring
# switches on. The Scorecard's first live "emergence" cell: does reportability appear at
# SFT, at DPO, or gradually? (base already covered; each stage has its own 11-layer lens.)
STAGES = [
    ("base",         "allenai/Olmo-3-1025-7B",         "lenses/olmo-3-1025-7b.pt"),
    ("instruct-sft", "allenai/Olmo-3-7B-Instruct-SFT", "lenses/olmo-3-7b-instruct-sft.pt"),
    ("instruct-dpo", "allenai/Olmo-3-7B-Instruct-DPO", "lenses/olmo-3-7b-instruct-dpo.pt"),
    ("instruct",     "allenai/Olmo-3-7B-Instruct",     "lenses/olmo-3-7b-instruct.pt"),
    ("think-sft",    "allenai/Olmo-3-7B-Think-SFT",    "lenses/olmo-3-7b-think-sft.pt"),
    ("think-dpo",    "allenai/Olmo-3-7B-Think-DPO",    "lenses/olmo-3-7b-think-dpo.pt"),
    ("think",        "allenai/Olmo-3-7B-Think",        "lenses/olmo-3-7b-think.pt"),
]


@app.local_entrypoint()
def stages(n: int = 150):
    print("  Metacognition across post-training STAGES (base -> SFT -> DPO -> final) — emergence")
    for r in metacog_arm.starmap([(s, h, lf, n) for (s, h, lf) in STAGES]):
        print("  ->", r)
    print("  saved: metacog/ladder_*.json  (base/instruct/think reused; SFT/DPO stages added)")


# v4 — SUPERVISED direction probe. The uncertainty-word readout is calibrated on the base
# workspace, so its apparent covert-signal DROP after SFT is confounded (the post-trained
# workspace is reshaped). A per-arm supervised probe fixes this: fit a linear correctness
# direction in EACH arm's own workspace residual, cross-validated, so the measure adapts to
# each arm's geometry. Question the confounded readout could not answer: does the covert
# error signal actually persist across post-training, or genuinely weaken?
@app.function(image=image, gpu="A100-80GB", volumes={"/cache": cache, "/out": out},
              timeout=45*60, env=ENV, retries=1)
def metacog_probe(slug: str, hf_name: str, lens_file: str, n: int = 150) -> dict:
    """Save each arm's workspace-layer residual (L18, last token) + correctness, for an
    offline cross-validated difference-in-means probe of 'does the workspace encode error'."""
    import pathlib, torch, jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens import ActivationRecorder
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(hf_name)
    hf = AutoModelForCausalLM.from_pretrained(hf_name, dtype=torch.bfloat16, device_map="cuda").eval()
    model = jlens.from_hf(hf, tok)
    LAYER = 18                                             # a workspace-band layer

    def gen(prompt, k=12):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            g = hf.generate(ids, max_new_tokens=k, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def residual(prompt):
        ids = model.encode(prompt, max_length=256)
        with ActivationRecorder(model.layers, at=[LAYER]) as rec:
            model.forward(ids)
            h = rec.activations[LAYER][0]                 # [seq, d]
        return h[-1].float().cpu()                        # last-token residual

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True)
    it = iter(ds); res, lab = [], []
    while len(res) < n:
        ex = next(it); q = ex["question"].strip()
        aliases = [a for a in list(ex["answer"].get("aliases", [])) + [ex["answer"].get("value", "")]
                   if a and len(a) >= 2]
        if not aliases or len(q) < 12:
            continue
        prompt = f"Question: {q}\nAnswer:"
        ans = gen(prompt)
        lab.append(1 if any(a.lower() in ans.lower() for a in aliases) else 0)
        res.append(residual(prompt))

    d = pathlib.Path("/out/metacog"); d.mkdir(parents=True, exist_ok=True)
    torch.save({"slug": slug, "layer": LAYER, "residuals": torch.stack(res).half(),
                "correct": lab}, d / f"probe_{slug}.pt")
    out.commit()
    return {"slug": slug, "n": len(res), "acc": sum(lab) / len(lab)}


@app.local_entrypoint()
def probe(n: int = 150):
    print("  v4 supervised direction probe — de-confound the covert trajectory across stages")
    for r in metacog_probe.starmap([(s, h, lf, n) for (s, h, lf) in STAGES]):
        print("  ->", r)
    print("  saved: metacog/probe_*.pt  (offline CV diff-in-means -> per-arm covert AUROC)")


# PRETRAINING sweep — WHEN does covert self-monitoring first appear? Because the supervised
# probe reads the raw L18 residual (no fitted lens), it runs at any checkpoint. 7 stage1
# checkpoints span pretraining 0 -> 1.41M steps; at each we probe whether the mid-layer
# residual encodes the model's own correctness, alongside its TriviaQA accuracy.
PRETRAIN = [
    ("step0000", "stage1-step0"),       ("step0235k", "stage1-step235000"),
    ("step0471k", "stage1-step471000"), ("step0705k", "stage1-step705000"),
    ("step0941k", "stage1-step941000"), ("step1177k", "stage1-step1177000"),
    ("step1414k", "stage1-step1413814"),
]


@app.function(image=image, gpu="A100-80GB", volumes={"/cache": cache, "/out": out},
              timeout=60 * 60, env=ENV, retries=1)
def metacog_pretrain(tag: str, revision: str, n: int = 150) -> dict:
    import pathlib, torch, jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens import ActivationRecorder
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(BASE, revision=revision)
    hf = AutoModelForCausalLM.from_pretrained(BASE, revision=revision, dtype=torch.bfloat16,
                                              device_map="cuda").eval()
    model = jlens.from_hf(hf, tok)
    LAYER = 18

    def gen(prompt, k=12):
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            g = hf.generate(ids, max_new_tokens=k, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def residual(prompt):
        ids = model.encode(prompt, max_length=256)
        with ActivationRecorder(model.layers, at=[LAYER]) as rec:
            model.forward(ids)
            h = rec.activations[LAYER][0]
        return h[-1].float().cpu()

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True)
    it = iter(ds); res, lab = [], []
    while len(res) < n:
        ex = next(it); q = ex["question"].strip()
        aliases = [a for a in list(ex["answer"].get("aliases", [])) + [ex["answer"].get("value", "")]
                   if a and len(a) >= 2]
        if not aliases or len(q) < 12:
            continue
        prompt = f"Question: {q}\nAnswer:"
        ans = gen(prompt)
        lab.append(1 if any(a.lower() in ans.lower() for a in aliases) else 0)
        res.append(residual(prompt))

    d = pathlib.Path("/out/metacog"); d.mkdir(parents=True, exist_ok=True)
    torch.save({"tag": tag, "revision": revision, "layer": LAYER,
                "residuals": torch.stack(res).half(), "correct": lab}, d / f"pretrain_{tag}.pt")
    out.commit()
    return {"tag": tag, "revision": revision, "n": len(res), "acc": sum(lab) / len(lab)}


@app.local_entrypoint()
def pretrain(n: int = 150):
    print("  Pretraining sweep — when does covert self-monitoring appear across stage1?")
    for r in metacog_pretrain.starmap([(t, rev, n) for (t, rev) in PRETRAIN]):
        print("  ->", r)
    print("  saved: metacog/pretrain_*.pt  (offline CV probe -> covert AUROC vs pretraining step)")
