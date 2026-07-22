"""Every consciousness test Dehaene & Naccache asked for, run in the open on OLMo-3.

Their commentary on Gurnee & Lindsey proposes a battery of the tests they use to probe
consciousness in humans, and notes Anthropic could run them. Claude is closed and most were
not run; OLMo-3 makes all of them runnable. This harness implements six, each as a READOUT
through the published 31-layer olmo-3-1025-7b Jacobian lens and/or a J-space ABLATION
(project the concept's readout direction out of the residual at chosen layers).

    1. ignition        graded evidence -> threshold nonlinearity + bifurcation (readout)
    2. trace_cond      variable-gap dependency, J-space ablation impairs long gaps (ablation)
    3. avoidance       inclusion/exclusion: early vs late layer ablation (ablation)
    4. local_global    global-rule vs local-transition prediction (ablation)
    5. metacognition   does the J-space encode correctness / known-vs-unknown (readout)
    6. dual_task       holding two concepts -> capacity interference (readout)

Everything is a pure forward pass on a frozen model + the fitted lens. No training.
"""

from __future__ import annotations
import modal

app = modal.App("olmo-reviewer-tests")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch>=2.6", "transformers>=5.5,<6", "accelerate", "safetensors",
                 "numpy", "huggingface_hub")
    .run_commands("pip install git+https://github.com/m9h/jacobian-lens.git",
                  "pip install git+https://github.com/m9h/jlens-lab.git")
)
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
out = modal.Volume.from_name("jlens-out", create_if_missing=True)
ENV = {"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"}

BASE = "allenai/Olmo-3-1025-7B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "olmo-3-1025-7b/jlens/Salesforce-wikitext/Olmo-3-1025-7B_jacobian_lens.pt"
GPU = "A100-80GB"


# ------------------------------------------------------------------ shared machinery
def _setup(torch):
    """Load model (LensModel), tokenizer, lens, and the unembed matrix W_U."""
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens import JacobianLens
    tok = AutoTokenizer.from_pretrained(BASE)
    hf = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                              device_map="cuda").eval()
    model = jlens.from_hf(hf, tok)
    lens = JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    W_U = hf.get_output_embeddings().weight.detach().float()      # [vocab, d_model]
    return hf, model, tok, lens, W_U


def _ids(tok, *words):
    s = set()
    for w in words:
        t = tok(w, add_special_tokens=False)["input_ids"]
        if t:
            s.add(t[0])
    return sorted(s)


def _concept_variants(c):
    return [f" {c}", c, f" {c.lower()}", c.lower(), f" {c.capitalize()}"]


def _read(lens, model, prompt, layers, ids, torch, use_jac=True, pos=-1):
    """Max lens logit over `ids` at position `pos`, per layer."""
    ll, _, _ = lens.apply(model, prompt, layers=layers, positions=[pos],
                          max_seq_len=256, use_jacobian=use_jac)
    idx = torch.tensor(ids)
    return {l: ll[l][0][idx].max().item() for l in layers}


def _ablation_dirs(lens, W_U, token_ids, layers, torch):
    """Residual direction that increases the lens logit of `token_ids` at each layer:
    d_l = normalize(J_l^T @ mean_t W_U[t]). Projecting it out ablates the concept's
    J-space presence at that layer."""
    tgt = W_U[torch.tensor(token_ids)].mean(0)                    # [d_model]
    dirs = {}
    for l in layers:
        d = lens.jacobians[l].float().T @ tgt                    # J_l^T W_U[c]
        dirs[l] = (d / d.norm().clamp_min(1e-6))
    return dirs


class _Ablator:
    """Forward hooks that project a set of directions out of the residual at given layers."""
    def __init__(self, model, dirs, torch):
        self.handles = []
        self.torch = torch
        for l, d in dirs.items():
            layer = model.layers[l]
            dv = d.to(next(layer.parameters()).device).to(torch.bfloat16)
            self.handles.append(layer.register_forward_hook(self._mk(dv)))

    def _mk(self, dv):
        def hook(_m, _i, output):
            hs = output[0] if isinstance(output, tuple) else output
            proj = (hs.to(dv.dtype) @ dv).unsqueeze(-1) * dv     # component along dv
            hs = hs - proj.to(hs.dtype)
            if isinstance(output, tuple):
                return (hs,) + tuple(output[1:])
            return hs
        return hook

    def __enter__(self): return self
    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


def _gen(hf, tok, prompt, torch, max_new=8):
    """Greedy continuation (for behavioral tests)."""
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        out = hf.generate(ids, max_new_tokens=max_new, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def _band(lens, lo=0.45, hi=0.75):
    L = lens.source_layers
    return [l for l in L if lo * L[-1] <= l <= hi * L[-1]]


# =================================================================== 1. IGNITION v2
# Individually-WEAK clues (each ambiguous), only jointly diagnostic, so the threshold sits
# mid-range. Bifurcation: at threshold, sample many equal-size clue SUBSETS -> bimodal?
IGN = [
 {"c":"France","off":"Brazil","base":"The travelers filled a notebook with memories of their trip.",
  "clues":[" A bakery sold warm bread early each morning."," People lingered over long lunches with wine.",
           " An accordion played across the square."," The streets were narrow and paved with cobblestones.",
           " A cafe served small cups of strong coffee."," The menu offered several kinds of soft cheese.",
           " A slow river passed under old stone bridges."," Neighbors kissed on each cheek in greeting."]},
 {"c":"Japan","off":"Egypt","base":"The travelers filled a notebook with memories of their trip.",
  "clues":[" Everyone removed their shoes before going inside."," Meals came in many small careful portions.",
           " The train arrived exactly on time."," People bowed slightly when greeting."," A garden was arranged around a single stone.",
           " Tea was prepared with slow deliberate movements."," Vending machines stood on every corner."," Pink blossoms lined the riverbank."]},
 {"c":"hospital","off":"hotel","base":"The building was busy from early in the morning.",
  "clues":[" The floors smelled faintly of disinfectant."," Staff wore matching uniforms and soft shoes.",
           " A steady beeping came from down the hall."," Visitors spoke in low worried voices."," Someone was wheeled past on a narrow bed.",
           " A board listed names and room numbers."," Bright lights stayed on all night."," A cart of small paper cups went room to room."]},
 {"c":"winter","off":"summer","base":"She looked out the window before leaving.",
  "clues":[" She wrapped a thick scarf around her neck."," Her breath showed in the air."," The branches outside were bare and grey.",
           " Frost covered the car windows."," The days ended dark by late afternoon."," People hurried indoors stamping their boots.",
           " The heating clicked on with a low hum."," A grey sky promised more of the same."]},
 {"c":"library","off":"gym","base":"They spent the whole afternoon in the building.",
  "clues":[" People spoke only in hushed whispers."," Long rows of shelves stretched to the back."," Someone stamped a card at the front desk.",
           " A sign asked visitors to return items on time."," Tables were spread with open notebooks."," The air smelled of old paper.",
           " A cart of returned items waited to be sorted."," Lights buzzed above the quiet aisles."]},
]


@app.function(image=image, gpu=GPU, volumes={"/cache": cache, "/out": out},
              timeout=40*60, env=ENV, retries=1)
def ignition() -> dict:
    import json, pathlib, random, torch
    hf, model, tok, lens, W_U = _setup(torch)
    layers = lens.source_layers
    ws = _band(lens)

    def on_off(prompt, it):
        on = _read(lens, model, prompt, layers, _ids(tok, *_concept_variants(it["c"])), torch)
        off = _read(lens, model, prompt, layers, _ids(tok, *_concept_variants(it["off"])), torch)
        return {l: on[l]-off[l] for l in layers}

    graded = []
    for it in IGN:
        curves = [on_off(it["base"]+"".join(it["clues"][:k]), it) for k in range(len(it["clues"])+1)]
        graded.append({"c":it["c"], "contrast":curves})

    # bifurcation at threshold: k* where the ws-band contrast first exceeds its midpoint
    def wsmean(c): return sum(c[l] for l in ws)/len(ws)
    bif=[]
    for it in IGN:
        full=[wsmean(on_off(it["base"]+"".join(it["clues"][:k]), it)) for k in range(len(it["clues"])+1)]
        lo,hi=min(full),max(full)
        kstar=next((k for k in range(len(full)) if full[k]>=lo+0.5*(hi-lo)), len(full)//2)
        rng=random.Random(len(it["c"])); pool=it["clues"]; vals=[]
        for _ in range(30):                       # 30 equal-size subsets AT threshold count
            sub=rng.sample(pool, max(1,min(kstar,len(pool))))
            vals.append(wsmean(on_off(it["base"]+"".join(sub), it)))
        bif.append({"c":it["c"], "kstar":kstar, "full":full, "subset_ws":vals})

    res={"layers":layers, "ws_band":[ws[0],ws[-1]], "graded":graded, "bifurcation":bif}
    _save("ignition_v2", res); return {"test":"ignition_v2", "items":len(IGN)}


# =================================================================== 2. TRACE CONDITIONING
# Lindsey's exact proposed paradigm: the last word is determined by the first (violin->river),
# separated by a variable number of distractors. GNW: bridging a GAP needs the workspace, so
# J-space ablation should impair long-gap far more than the adjacent no-gap case.
TRACE_PAIRS = [("violin","river"),("anchor","ladder"),("copper","meadow"),
               ("lantern","harvest"),("velvet","glacier"),("marble","thunder")]
DISTRACTORS = [" and","the","then","also","some","many","often","quite","very","really",
               "here","there","today","later","maybe","indeed","simply","clearly"]


@app.function(image=image, gpu=GPU, volumes={"/cache": cache, "/out": out},
              timeout=40*60, env=ENV, retries=1)
def trace_cond() -> dict:
    import json, pathlib, torch
    hf, model, tok, lens, W_U = _setup(torch)
    ws = _band(lens)

    def last_logit(prompt, target_ids, dirs=None):
        if dirs is None:
            r = _read(lens, model, prompt, ws, target_ids, torch)  # workspace readout of target
            return sum(r.values())/len(r)
        # behavioral: probability the model actually predicts the target next, with/without ablation
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            logits = hf(ids).logits[0,-1].float()
        lp = torch.log_softmax(logits, -1)
        return max(lp[t].item() for t in target_ids)

    gaps=[0,2,4,8,16]
    rows=[]
    for first,last in TRACE_PAIRS:
        tids=_ids(tok, f" {last}", last)
        dirs=_ablation_dirs(lens, W_U, tids, ws, torch)
        for g in gaps:
            fill="".join(" "+DISTRACTORS[i%len(DISTRACTORS)] for i in range(g))
            # the rule is stated once, then re-invoked; last token to predict is `last`
            ctx=(f"Rule: after {first} comes {last}. {first}{fill} therefore next comes")
            base=last_logit(ctx, tids)
            with _Ablator(model, dirs, torch):
                pass
            # behavioral effect of ablation on predicting `last`
            clean=last_logit(ctx, tids, dirs="behav")
            with _Ablator(model, dirs, torch):
                abl=last_logit(ctx, tids, dirs="behav")
            rows.append({"pair":[first,last],"gap":g,"ws_readout":base,
                         "logp_clean":clean,"logp_ablated":abl,"drop":clean-abl})
    _save("trace_cond", {"gaps":gaps,"rows":rows}); return {"test":"trace_cond","pairs":len(TRACE_PAIRS)}


# =================================================================== 3. AVOIDANCE (incl/excl)
# Gurnee's paradigm on OLMo: a passage implies a concept without naming it; instruct the model
# to NAME it vs AVOID it; ablate the concept's J-space at EARLY (workspace) vs LATE (motor)
# layers. GNW/prefrontal prediction: early-layer ablation breaks AVOIDANCE (suppression) far
# more than NAMING; late-layer ablation lowers naming under both.
AVOID_ITEMS = [
 {"c":"France","passage":"Their trip included croissants, the Louvre, and a climb up the famous iron tower."},
 {"c":"chess","passage":"They opened with a pawn, developed a knight, castled, and finally forced checkmate."},
 {"c":"winter","passage":"Snow piled on the fence, breath fogged in the freezing air, and icicles hung from the gutter."},
 {"c":"Japan","passage":"They slept on tatami mats, rode the bullet train, and watched cherry blossoms fall."},
 {"c":"hospital","passage":"Nurses hurried between rooms, monitors beeped by each bed, and a surgeon scrubbed in."},
]


@app.function(image=image, gpu=GPU, volumes={"/cache": cache, "/out": out},
              timeout=40*60, env=ENV, retries=1)
def avoidance() -> dict:
    import torch
    hf, model, tok, lens, W_U = _setup(torch)
    L=lens.source_layers
    early=[l for l in L if 6<=l<=15]     # workspace band
    late=[l for l in L if 18<=l<=27]     # motor band

    def names(text, c):
        g=_gen(hf, tok, text, torch, max_new=6).lower()
        return c.lower() in g

    rows=[]
    for it in AVOID_ITEMS:
        c=it["c"]; tids=_ids(tok, f" {c}", c, f" {c.lower()}", c.lower())
        d_early=_ablation_dirs(lens, W_U, tids, early, torch)
        d_late=_ablation_dirs(lens, W_U, tids, late, torch)
        name_p=f"{it['passage']} In one word, the place or thing this describes is:"
        avoid_p=(f"{it['passage']} Name any word in the same category but do NOT say the obvious "
                 f"one it describes. One word:")
        r={"c":c}
        # clean
        r["name_clean"]=names(name_p, c); r["avoid_fail_clean"]=names(avoid_p, c)
        for tag,dirs in (("early",{**d_early}),("late",{**d_late})):
            with _Ablator(model, dirs, torch):
                r[f"name_{tag}"]=names(name_p, c)
                r[f"avoid_fail_{tag}"]=names(avoid_p, c)
        rows.append(r)
    _save("avoidance", {"rows":rows}); return {"test":"avoidance","items":len(AVOID_ITEMS)}


# =================================================================== 4. LOCAL-GLOBAL
# Bekinschtein: local transition (predictable, unconscious) vs global rule (needs the
# workspace). Sequences where the final token is fixed by the GLOBAL pattern but not the
# local one (1 2 3 1 2 3 1 2 -> 3) vs a LOCAL repeat (3 3 3 3 -> 3). Test whether J-space
# ablation of the correct continuation impairs the GLOBAL case more than the LOCAL case.
@app.function(image=image, gpu=GPU, volumes={"/cache": cache, "/out": out},
              timeout=40*60, env=ENV, retries=1)
def local_global() -> dict:
    import torch
    hf, model, tok, lens, W_U = _setup(torch)
    ws=_band(lens)
    syms=["A","B","C","D"]
    rows=[]
    def logp(prompt, tids, dirs=None):
        ids=tok(prompt, return_tensors="pt").input_ids.to("cuda")
        if dirs is None:
            with torch.no_grad(): logits=hf(ids).logits[0,-1].float()
        else:
            with _Ablator(model, dirs, torch):
                with torch.no_grad(): logits=hf(ids).logits[0,-1].float()
        lp=torch.log_softmax(logits,-1)
        return max(lp[t].item() for t in tids)
    for period in (2,3,4):
        pat=syms[:period]
        seq=(pat*6)[:12]                       # global-rule sequence
        nxt=pat[len(seq)%period]
        tids=_ids(tok, f" {nxt}", nxt)
        dirs=_ablation_dirs(lens, W_U, tids, ws, torch)
        gp=" ".join(seq)+" "
        loc=" ".join([nxt]*12)+" "             # local-repeat sequence (same target)
        rows.append({"kind":"global","period":period,
                     "clean":logp(gp,tids),"ablated":logp(gp,tids,dirs)})
        rows.append({"kind":"local","period":period,
                     "clean":logp(loc,tids),"ablated":logp(loc,tids,dirs)})
    _save("local_global", {"rows":rows}); return {"test":"local_global","n":len(rows)}


# =================================================================== 5. METACOGNITION (C2)
# Does the J-space encode the model's own correctness? Ask factual questions the base model
# gets RIGHT vs WRONG (verified by generation), and read whether a "confidence/uncertainty"
# signal in the workspace separates them -- the machine analog of error monitoring.
META_Q = [
 ("The capital of France is","Paris"),("The capital of Australia is","Canberra"),
 ("The chemical symbol for gold is","Au"),("The largest planet is","Jupiter"),
 ("The author of Hamlet is","Shakespeare"),("The capital of Kazakhstan is","Astana"),
 ("The square root of 144 is","12"),("The capital of Bhutan is","Thimphu"),
 ("The smallest prime number is","2"),("The capital of Burkina Faso is","Ouagadougou"),
]
UNCERTAIN = ["maybe","unsure","not","unknown","guess","perhaps","uncertain","don't","hmm"]


@app.function(image=image, gpu=GPU, volumes={"/cache": cache, "/out": out},
              timeout=40*60, env=ENV, retries=1)
def metacognition() -> dict:
    import torch
    hf, model, tok, lens, W_U = _setup(torch)
    ws=_band(lens)
    unc_ids=_ids(tok, *[f" {w}" for w in UNCERTAIN], *UNCERTAIN)
    rows=[]
    for q,ans in META_Q:
        gen=_gen(hf, tok, q, torch, max_new=6)
        correct=ans.lower() in gen.lower()
        r=_read(lens, model, q, ws, unc_ids, torch, pos=-1)     # workspace 'uncertainty' signal
        rows.append({"q":q,"answer":ans,"gen":gen.strip()[:40],"correct":correct,
                     "ws_uncertainty":sum(r.values())/len(r)})
    _save("metacognition", {"rows":rows}); return {"test":"metacognition","n":len(META_Q)}


# =================================================================== 6. DUAL-TASK
# The capacity bottleneck: hold ONE vs TWO concepts in mind. If the workspace is limited,
# a concept's J-space presence should be WEAKER when a second concept must be held too.
DUAL = [("France","chess"),("winter","hospital"),("Japan","library"),
        ("ocean","music"),("desert","wedding")]


@app.function(image=image, gpu=GPU, volumes={"/cache": cache, "/out": out},
              timeout=40*60, env=ENV, retries=1)
def dual_task() -> dict:
    import torch
    hf, model, tok, lens, W_U = _setup(torch)
    ws=_band(lens)
    rows=[]
    for a,b in DUAL:
        ida=_ids(tok, *_concept_variants(a)); idb=_ids(tok, *_concept_variants(b))
        single=f"Hold the concept of {a} in mind while you count to five. one two three four five. Now:"
        dual=f"Hold BOTH {a} and {b} in mind while you count to five. one two three four five. Now:"
        sa=_read(lens, model, single, ws, ida, torch)
        da=_read(lens, model, dual, ws, ida, torch)
        rows.append({"a":a,"b":b,"single_a":sum(sa.values())/len(sa),
                     "dual_a":sum(da.values())/len(da)})
    _save("dual_task", {"rows":rows}); return {"test":"dual_task","n":len(DUAL)}


# ------------------------------------------------------------------ save + run-all
def _save(name, obj):
    import json, pathlib
    d=pathlib.Path("/out/reviewer_tests"); d.mkdir(parents=True, exist_ok=True)
    (d/f"{name}.json").write_text(json.dumps(obj))
    out.commit()


@app.local_entrypoint()
def run_all():
    for fn in (ignition, trace_cond, avoidance, local_global, metacognition, dual_task):
        print("  ->", fn.remote())
    print("  all saved to jlens-out volume: reviewer_tests/*.json")
