"""Calibrate dim_batch for Qwen3-32B J-lens fitting on the GB10 (128GB unified).

Measured, not guessed: dim_batch=128 OOM'd on an H200 for a d=2048 model, and Qwen3-32B is
d=5120. On unified memory a GPU OOM can starve the host, so this runs under Slurm with a hard
--mem cap and steps DOWN from a conservative starting point.
"""
import os, sys, time, gc
import torch, jlens
from jlens.fitting import jacobian_for_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

M = "Qwen/Qwen3-32B"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
prompts = []
for r in ds:
    t = (r["text"] or "").strip()
    if len(t) > 300:
        prompts.append(t[:2000])
    if len(prompts) >= 1:
        break

print("loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(M)
hf = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()
model = jlens.from_hf(hf, tok)
nl, d = hf.config.num_hidden_layers, hf.config.hidden_size
print(f"{M}: {nl} layers, d={d} | resident {torch.cuda.memory_allocated()/2**30:.1f} GiB",
      flush=True)

for step in (4, 8):
    layers = list(range(0, nl - 1, step))
    for db in (32, 16, 8):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.time()
            jacobian_for_prompt(model, prompts[0], layers, max_seq_len=128, dim_batch=db)
            dt = time.time() - t0
            pk = torch.cuda.max_memory_allocated() / 2**30
            print(f"  layers={len(layers):2d} (every {step}) dim_batch={db:3d}: OK  "
                  f"{dt:6.1f}s/prompt  peak {pk:6.1f} GiB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  layers={len(layers):2d} (every {step}) dim_batch={db:3d}: OOM", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  layers={len(layers):2d} dim_batch={db:3d}: {type(e).__name__}: "
                  f"{str(e)[:70]}", flush=True)
            gc.collect(); torch.cuda.empty_cache()
print("CALIB DONE", flush=True)
