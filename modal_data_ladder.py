"""Does J-space movement scale with training-DATA QUANTITY?

Our OLMo result: post-training METHOD sets how far the J-space moves (SFT+DPO ~5x RLVR) while
task DOMAIN adds ~1%. The missing axis is quantity. OpenThoughts-Agent released a log-spaced
data ladder on a fixed base and fixed method:

    Qwen3-32B  ->  SFT-316 -> 1K -> 3.16K -> 10K -> 31.6K -> 100K      (300x sweep)

Same base, same recipe, only the number of SFT examples changes. So cos(J_base, J_arm) as a
function of data size isolates quantity from method and domain.

Protocol matches our OLMo ladder and Neuronpedia's fits: wikitext-103, seq_len 128, bfloat16.
The base is refitted here rather than taken from Neuronpedia, whose published qwen3-32b trace
stops at n=80 -- mixing budgets would confound the comparison we are trying to make.

  modal run modal_data_ladder.py
"""
import os, modal

app = modal.App("otagent-data-ladder")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("git")
         .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
         .pip_install("transformers>=4.57,<6", "accelerate", "safetensors", "datasets",
                      "huggingface_hub", "sentencepiece", "protobuf")
         .pip_install("git+https://github.com/anthropics/jacobian-lens.git"))
cache = modal.Volume.from_name("hf-cache-nla", create_if_missing=True)

BASE = "Qwen/Qwen3-32B"
ARMS = [("base", BASE, 0)] + [
    (f"SFT-{n}", f"open-thoughts/OpenThinkerAgent-32B-SFT-{n}", v)
    for n, v in [("316", 316), ("1K", 1000), ("3.16K", 3160), ("10K", 10000),
                 ("31.6K", 31600), ("100K", 100000)]]
N_PROMPTS, MAX_LEN, DIM_BATCH = 40, 128, 64   # starting point; halves on OOM


@app.function(image=image, gpu="B200", timeout=180*60, volumes={"/cache": cache},
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN","")})])
def fit_one(name: str, repo: str, n_examples: int):
    import gc, torch, jlens
    from jlens.fitting import jacobian_for_prompt
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    os.environ["HF_HOME"]="/cache"; os.environ["HF_HUB_DISABLE_XET"]="1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

    ds=load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="train",streaming=True)
    prompts=[]
    for r in ds:
        t=(r["text"] or "").strip()
        if len(t)>300: prompts.append(t[:2000])
        if len(prompts)>=N_PROMPTS: break

    tok=AutoTokenizer.from_pretrained(repo)
    hf=AutoModelForCausalLM.from_pretrained(repo,dtype=torch.bfloat16,device_map="cuda").eval()
    model=jlens.from_hf(hf,tok)
    nl=hf.config.num_hidden_layers
    layers=list(range(0, nl-1, 4))          # every 4th; backward passes are shared across layers
    print(f"[{name}] {nl} layers, fitting {len(layers)}", flush=True)

    jsum={l:None for l in layers}; n=0; db=DIM_BATCH
    for i,p in enumerate(prompts):
        J=None
        while J is None and db >= 4:
            try:
                J,_,_ = jacobian_for_prompt(model,p,layers,max_seq_len=MAX_LEN,dim_batch=db)
            except torch.cuda.OutOfMemoryError:
                # step down and REMEMBER it -- a fixed dim_batch guessed too high cost us an
                # entire 7-model run once already.
                gc.collect(); torch.cuda.empty_cache(); db //= 2
                print(f"  [{name}] OOM -> dim_batch={db}", flush=True)
            except Exception as e:
                print(f"  [{name}] skip {i}: {type(e).__name__}: {str(e)[:60]}", flush=True)
                break
        if J is None: continue
        for l in layers:
            jsum[l]= J[l].float().cpu().clone() if jsum[l] is None else jsum[l]+J[l].float().cpu()
        n+=1
        if n%10==0: print(f"  [{name}] {n}/{len(prompts)}", flush=True)
    del hf, model; gc.collect(); torch.cuda.empty_cache()
    out={"name":name,"repo":repo,"n_examples":n_examples,"n_prompts":n,
         "J":{l:(jsum[l]/n).to(torch.float16) for l in layers}}
    import shutil
    shutil.rmtree(f"/cache/hub/models--{repo.replace('/','--')}", ignore_errors=True)
    torch.save(out, f"/cache/ladder_{name}.pt")
    cache.commit()
    print(f"[{name}] done, n={n}", flush=True)
    return {"name":name,"n_examples":n_examples,"n_prompts":n,"layers":layers}


@app.function(image=image, timeout=30*60, volumes={"/cache": cache})
def compare():
    import torch, math
    base=torch.load("/cache/ladder_base.pt",map_location="cpu",weights_only=False)
    Jb=base["J"]; rows=[]
    for name,_,nex in ARMS[1:]:
        try: a=torch.load(f"/cache/ladder_{name}.pt",map_location="cpu",weights_only=False)
        except Exception as e: print(f"  {name}: MISSING ({e})"); continue
        cs=[]
        for l in Jb:
            if l not in a["J"]: continue
            x=Jb[l].float().flatten().double(); y=a["J"][l].float().flatten().double()
            cs.append(float(x@y/(x.norm()*y.norm())))
        rows.append((name,nex,sum(cs)/len(cs),min(cs),max(cs)))
    print(f"\n{'arm':10s} {'SFT examples':>13} {'cos(J_base,J_arm)':>18} {'move %':>8}  {'per-layer min/max':>20}")
    for name,nex,c,lo,hi in rows:
        print(f"{name:10s} {nex:13,d} {c:18.4f} {100*(1-c):7.1f}%  {lo:.3f}/{hi:.3f}")
    print("\nOLMo reference: SFT+DPO cos 0.69 (~31% move); RLVR 0.94 (~6%); refit floor ~0.97 pooled.")
    print("If cos falls smoothly with log(data), quantity drives the shift; if it saturates early,")
    print("the viewpoint installs with the first few hundred examples and more data adds little.")
    return rows


@app.local_entrypoint()
def main():
    for name, repo, nex in ARMS:
        print(fit_one.remote(name, repo, nex))
    print(compare.remote())
