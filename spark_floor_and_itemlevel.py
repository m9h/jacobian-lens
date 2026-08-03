"""Two open exposures, closed.

(A) MATCHED-BUDGET REFIT FLOOR. The two new ladders were fitted at 40 prompts; our only floor
    (~0.97 pooled) comes from 616-prompt fits. Fit the SAME model twice on DISJOINT prompt
    samples at 40 and measure how far two honest fits sit apart at that budget.

(B) ITEM-LEVEL CONTROL. Our ladder computes cos(mean_p J_base, mean_p J_arm) -- it averages
    Jacobians THEN compares, which is magnitude-weighted. The within-item version compares per
    prompt THEN averages. The societies-of-thought agent found these differ hugely on their data
    (+0.0110 -> +0.0023, between-item estimate excluded at 3.1 SE). We have never measured our gap.
"""
import gc, json, os, shutil, sys, time
import torch, jlens
from jlens.fitting import jacobian_for_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

OUT = "/mnt/t9/jlens/floor_itemlevel"; os.makedirs(OUT, exist_ok=True)
MAX_LEN, DIM_BATCH, N = 128, 32, 40

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
P = []
for r in ds:
    t = (r["text"] or "").strip()
    if len(t) > 300: P.append(t[:2000])
    if len(P) >= 2 * N: break
SET_A, SET_B = P[:N], P[N:2*N]        # disjoint samples, same corpus, same budget


def fit(repo, prompts, layer_step, keep_per_prompt=False, tag=""):
    tok = AutoTokenizer.from_pretrained(repo)
    hf = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda").eval()
    model = jlens.from_hf(hf, tok); nl = hf.config.num_hidden_layers
    layers = list(range(0, nl - 1, layer_step))
    jsum, n, per = {l: None for l in layers}, 0, []
    for i, p in enumerate(prompts):
        try:
            J, _, _ = jacobian_for_prompt(model, p, layers, max_seq_len=MAX_LEN, dim_batch=DIM_BATCH)
        except Exception as e:
            print(f"    skip {i}: {type(e).__name__}", flush=True)
            gc.collect(); torch.cuda.empty_cache(); continue
        if keep_per_prompt:
            per.append({l: J[l].to(torch.float16).cpu() for l in layers})
        for l in layers:
            v = J[l].float().cpu()
            jsum[l] = v.clone() if jsum[l] is None else jsum[l] + v
        n += 1
        if n % 10 == 0: print(f"    {tag} {n}/{len(prompts)}", flush=True)
    del hf, model; gc.collect(); torch.cuda.empty_cache()
    return {l: (jsum[l] / n) for l in layers}, per, n


def cos(a, b):
    x = a.float().flatten().double(); y = b.float().flatten().double()
    return float(x @ y / (x.norm() * y.norm()))


res = {}
# ---------------- (A) refit floor at 40 prompts, on the 8B agentic base ----------------
print("=== (A) matched-budget refit floor: Qwen3-8B, two disjoint 40-prompt samples ===", flush=True)
JA, _, nA = fit("Qwen/Qwen3-8B", SET_A, 4, tag="A1")
JB, _, nB = fit("Qwen/Qwen3-8B", SET_B, 4, tag="A2")
fl = {l: cos(JA[l], JB[l]) for l in JA if l in JB}
res["floor_40prompt_qwen3_8b"] = {"per_layer": fl, "mean": sum(fl.values())/len(fl),
                                  "n_a": nA, "n_b": nB}
print(f"  floor at 40 prompts: mean cos {sum(fl.values())/len(fl):.4f} "
      f"(min {min(fl.values()):.3f} / max {max(fl.values()):.3f})", flush=True)
print(f"  -> a ladder 'move' smaller than {100*(1-sum(fl.values())/len(fl)):.1f}% is not "
      f"distinguishable from refit noise at this budget", flush=True)
json.dump(res, open(f"{OUT}/result.json", "w"), indent=1)

# ---------------- (B) item-level control on the OLMo ladder ----------------
print("\n=== (B) item-level control: OLMo-3-7B base vs Instruct ===", flush=True)
Jb, per_b, nb = fit("allenai/Olmo-3-1025-7B", SET_A, 3, keep_per_prompt=True, tag="B-base")
Ji, per_i, ni = fit("allenai/Olmo-3-7B-Instruct", SET_A, 3, keep_per_prompt=True, tag="B-inst")
layers = sorted(set(Jb) & set(Ji))
avg_then_cmp = sum(cos(Jb[l], Ji[l]) for l in layers) / len(layers)      # current statistic
m = min(len(per_b), len(per_i))
per_prompt = []
for k in range(m):
    per_prompt.append(sum(cos(per_b[k][l], per_i[k][l]) for l in layers) / len(layers))
cmp_then_avg = sum(per_prompt) / len(per_prompt)                          # item-level statistic
import statistics as st
res["item_level"] = {"avg_then_compare": avg_then_cmp, "compare_then_avg": cmp_then_avg,
                     "gap": avg_then_cmp - cmp_then_avg, "n_prompts": m,
                     "per_prompt_sd": st.pstdev(per_prompt)}
print(f"  averaged-then-compared (published statistic): {avg_then_cmp:.4f}  "
      f"-> {100*(1-avg_then_cmp):.1f}% move")
print(f"  compared-then-averaged (item-level)         : {cmp_then_avg:.4f}  "
      f"-> {100*(1-cmp_then_avg):.1f}% move   (sd across prompts {st.pstdev(per_prompt):.3f})")
print(f"  GAP: {avg_then_cmp - cmp_then_avg:+.4f}")
json.dump(res, open(f"{OUT}/result.json", "w"), indent=1)
print("FLOOR+ITEMLEVEL DONE", flush=True)
