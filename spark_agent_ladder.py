"""J-space movement across an AGENTIC method ladder: Qwen3-8B -> ColdStart SFT -> RL.

Complements our OLMo post-training result (SFT+DPO moves the J-space ~31% from base; RLVR ~6%)
in a different domain and a different family. OpenThoughts released the pre-RL and post-RL
checkpoints of the same agent, so the RL step is isolated.

Sized for the GB10: 8B models (~15GB) rather than the 32B data ladder, which OOM'd this host at
--mem=90G while loading. Each model is purged after fitting -- only ~27GB of disk is free.
"""
import gc, json, os, shutil, sys, time
import torch, jlens
from jlens.fitting import jacobian_for_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

ARMS = [("base",     "Qwen/Qwen3-8B"),
        ("sft",      "open-thoughts/OpenThinkerAgent-8B-ColdStartSFTForRL"),
        ("rl",       "open-thoughts/OpenThinkerAgent-8B-RL")]
N_PROMPTS, MAX_LEN, DIM_BATCH, LAYER_STEP = 40, 128, 32, 4
OUT = "/home/mhough/Workspace/agent_ladder"
os.makedirs(OUT, exist_ok=True)

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
prompts = []
for r in ds:
    t = (r["text"] or "").strip()
    if len(t) > 300:
        prompts.append(t[:2000])
    if len(prompts) >= N_PROMPTS:
        break
print(f"{len(prompts)} wikitext prompts", flush=True)


def purge(repo):
    d = os.path.join(os.environ.get("HF_HOME", ""), "hub",
                     "models--" + repo.replace("/", "--"))
    shutil.rmtree(d, ignore_errors=True)


for name, repo in ARMS:
    dest = f"{OUT}/{name}.pt"
    if os.path.exists(dest):
        print(f"[{name}] already fitted, skipping", flush=True); continue
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo)
    hf = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16,
                                              device_map="cuda").eval()
    model = jlens.from_hf(hf, tok)
    nl = hf.config.num_hidden_layers
    layers = list(range(0, nl - 1, LAYER_STEP))
    print(f"[{name}] {repo}  {nl} layers, fitting {len(layers)}, "
          f"resident {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
    jsum, n = {l: None for l in layers}, 0
    for i, p in enumerate(prompts):
        try:
            J, _, _ = jacobian_for_prompt(model, p, layers, max_seq_len=MAX_LEN,
                                          dim_batch=DIM_BATCH)
        except Exception as e:
            print(f"  [{name}] skip {i}: {type(e).__name__}", flush=True)
            gc.collect(); torch.cuda.empty_cache(); continue
        for l in layers:
            v = J[l].float().cpu()
            jsum[l] = v.clone() if jsum[l] is None else jsum[l] + v
        n += 1
        if n % 10 == 0:
            print(f"  [{name}] {n}/{len(prompts)}  {(time.time()-t0)/n:.1f}s/prompt", flush=True)
    if n == 0:
        print(f"[{name}] NO PROMPTS SUCCEEDED — aborting", flush=True); sys.exit(1)
    torch.save({"name": name, "repo": repo, "n": n, "layers": layers,
                "J": {l: (jsum[l] / n).to(torch.float16) for l in layers}}, dest)
    print(f"[{name}] saved n={n} in {(time.time()-t0)/60:.1f} min", flush=True)
    del hf, model, jsum; gc.collect(); torch.cuda.empty_cache()
    purge(repo)

# ---- compare ----
B = torch.load(f"{OUT}/base.pt", map_location="cpu", weights_only=False)
print(f"\n{'arm':6s} {'cos(J_base, J_arm)':>20} {'move':>8}   per-layer min/max")
res = {}
for name, _ in ARMS[1:]:
    a = torch.load(f"{OUT}/{name}.pt", map_location="cpu", weights_only=False)
    cs = []
    for l in B["J"]:
        if l not in a["J"]: continue
        x = B["J"][l].float().flatten().double(); y = a["J"][l].float().flatten().double()
        cs.append(float(x @ y / (x.norm() * y.norm())))
    m = sum(cs) / len(cs); res[name] = {"cos": m, "min": min(cs), "max": max(cs)}
    print(f"{name:6s} {m:20.4f} {100*(1-m):7.1f}%   {min(cs):.3f}/{max(cs):.3f}")
# the RL step in isolation
S = torch.load(f"{OUT}/sft.pt", map_location="cpu", weights_only=False)
R = torch.load(f"{OUT}/rl.pt", map_location="cpu", weights_only=False)
cs = []
for l in S["J"]:
    if l not in R["J"]: continue
    x = S["J"][l].float().flatten().double(); y = R["J"][l].float().flatten().double()
    cs.append(float(x @ y / (x.norm() * y.norm())))
res["rl_step"] = {"cos": sum(cs)/len(cs), "min": min(cs), "max": max(cs)}
print(f"\nRL step alone  cos(J_sft, J_rl) = {sum(cs)/len(cs):.4f}  "
      f"({100*(1-sum(cs)/len(cs)):.1f}% move)")
print("\nOLMo reference: SFT+DPO 0.69 (~31%) | RLVR 0.94 (~6%) | pooled refit floor ~0.97")
json.dump(res, open(f"{OUT}/result.json", "w"), indent=1)
print("LADDER DONE", flush=True)
