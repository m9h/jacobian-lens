"""Find the largest dim_batch that fits, BEFORE launching 7 sequential 32B fits.

Qwen3-30B-A3B (d=2048) already OOM'd at dim_batch=128 on an H200. Qwen3-32B has d=5120, so
128 was never going to fit. Measure instead of guess: 2 prompts per setting, report fit/OOM
and seconds per prompt so the full run can be costed.
"""
import os, modal
app = modal.App("ladder-calib")
image = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
         .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
         .pip_install("transformers>=4.57,<6","accelerate","safetensors","datasets",
                      "huggingface_hub","sentencepiece","protobuf")
         .pip_install("git+https://github.com/anthropics/jacobian-lens.git"))
cache = modal.Volume.from_name("hf-cache-nla", create_if_missing=True)

@app.function(image=image, gpu="B200", timeout=90*60, volumes={"/cache": cache},
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN","")})])
def calib():
    import time, torch, jlens
    from jlens.fitting import jacobian_for_prompt
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    os.environ["HF_HOME"]="/cache"; os.environ["HF_HUB_DISABLE_XET"]="1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    M="Qwen/Qwen3-32B"
    ds=load_dataset("Salesforce/wikitext","wikitext-103-raw-v1",split="train",streaming=True)
    prompts=[]
    for r in ds:
        t=(r["text"] or "").strip()
        if len(t)>300: prompts.append(t[:2000])
        if len(prompts)>=2: break
    tok=AutoTokenizer.from_pretrained(M)
    hf=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16,device_map="cuda").eval()
    model=jlens.from_hf(hf,tok); nl=hf.config.num_hidden_layers
    tot=torch.cuda.get_device_properties(0).total_memory/2**30
    print(f"GPU {torch.cuda.get_device_name(0)} {tot:.0f}GiB | model resident "
          f"{torch.cuda.memory_allocated()/2**30:.1f}GiB | layers {nl} d {hf.config.hidden_size}",
          flush=True)
    for nlayers,label in [(len(range(0,nl-1,4)),"every 4th"), (len(range(0,nl-1,8)),"every 8th")]:
        layers=list(range(0,nl-1,4 if label=="every 4th" else 8))
        for db in (64,32,16):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                t0=time.time()
                for p in prompts[:1]:
                    jacobian_for_prompt(model,p,layers,max_seq_len=128,dim_batch=db)
                dt=time.time()-t0
                pk=torch.cuda.max_memory_allocated()/2**30
                print(f"  layers={len(layers):2d} ({label:9s}) dim_batch={db:3d}: OK  "
                      f"{dt:5.1f}s/prompt  peak {pk:5.1f}GiB", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  layers={len(layers):2d} ({label:9s}) dim_batch={db:3d}: OOM", flush=True)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  layers={len(layers):2d} dim_batch={db:3d}: {type(e).__name__} {str(e)[:60]}", flush=True)
                torch.cuda.empty_cache()

@app.local_entrypoint()
def main(): calib.remote()
