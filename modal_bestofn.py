"""Does the covert workspace signal actually pick better answers than the model's own confidence?

Our metacognition result is a DESCRIPTION: the base model's workspace predicts its own errors at
AUROC 0.69, beyond output confidence. This asks the validity question, in the form RewardBench 2
used to validate itself -- best-of-N selection:

  generate N samples per question, rank them, take the top one, measure accuracy.

  RANKERS
    random          -- the floor
    mean logprob    -- the model's own confidence (the baseline to beat)
    workspace probe -- a cross-validated difference-in-means probe on the raw layer-L residual
    oracle          -- the ceiling (was any sample correct?)

If the probe beats mean-logprob, the covert signal is USEFUL, not merely present. If it does not,
the signal is real but not actionable -- which is also a result, and the one this framework
exists to be able to report.

The probe needs no lens, so it runs at any checkpoint.

  modal run modal_bestofn.py
"""
import os, modal

app = modal.App("workspace-bestofn")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
         .pip_install("transformers>=4.57,<6", "accelerate", "safetensors", "datasets",
                      "huggingface_hub", "numpy", "scikit-learn", "sentencepiece", "protobuf"))
cache = modal.Volume.from_name("hf-cache-nla", create_if_missing=True)

MODEL = "allenai/Olmo-3-1025-7B"
LAYER = 18                 # the workspace band, matching the metacognition work
N_SAMPLES, N_QUESTIONS, TEMP = 8, 250, 0.8


@app.function(image=image, gpu="H100", timeout=120*60, volumes={"/cache": cache},
              secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN","")})])
def run():
    import re, torch, numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    from sklearn.model_selection import StratifiedKFold
    os.environ["HF_HOME"]="/cache"; os.environ["HF_HUB_DISABLE_XET"]="1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

    ds = load_dataset("mandarjoshi/trivia_qa","rc.nocontext",split="validation",streaming=True)
    qs=[]
    for r in ds:
        al=[a.lower().strip() for a in r["answer"]["aliases"]] + [r["answer"]["value"].lower().strip()]
        qs.append({"q": r["question"], "aliases": set(a for a in al if a)})
        if len(qs) >= N_QUESTIONS: break
    print(f"{len(qs)} questions", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    rows=[]
    for qi,item in enumerate(qs):
        prompt = f"Question: {item['q']}\nAnswer:"
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=16, do_sample=True, temperature=TEMP,
                             top_p=0.95, num_return_sequences=N_SAMPLES,
                             return_dict_in_generate=True, output_scores=True,
                             pad_token_id=tok.pad_token_id)
        seqs = out.sequences                                  # [N, prompt+gen]
        gen = seqs[:, ids.shape[1]:]
        # mean logprob of each sampled continuation = the model's own confidence
        lps = torch.stack(out.scores, dim=1).log_softmax(-1)   # [N, T, V]
        tokl = lps.gather(2, gen.unsqueeze(2)).squeeze(2)      # [N, T]
        mask = (gen != tok.pad_token_id).float()
        mean_lp = ((tokl*mask).sum(1)/mask.sum(1).clamp_min(1)).float().cpu()
        # workspace residual at the last prompt+answer token, per sample
        with torch.no_grad():
            hs = m(seqs, output_hidden_states=True).hidden_states[LAYER]   # [N, T, d]
        h_last = hs[:, -1, :].float().cpu()
        for k in range(N_SAMPLES):
            txt = tok.decode(gen[k], skip_special_tokens=True).strip().lower()
            txt = re.split(r"[\n\.]", txt)[0].strip()
            correct = any(a in txt for a in item["aliases"]) if txt else False
            rows.append({"q": qi, "k": k, "text": txt, "correct": bool(correct),
                         "mean_logprob": float(mean_lp[k]), "h": h_last[k]})
        if qi % 25 == 0: print(f"  {qi+1}/{len(qs)}", flush=True)
    del m; torch.cuda.empty_cache()

    H = torch.stack([r["h"] for r in rows]).numpy()
    y = np.array([r["correct"] for r in rows], dtype=int)
    qid = np.array([r["q"] for r in rows])
    lp = np.array([r["mean_logprob"] for r in rows])
    print(f"\nsamples {len(rows)}, correct {y.mean():.3f}", flush=True)

    # cross-validated difference-in-means probe, SPLIT BY QUESTION so no leakage
    uq = np.unique(qid); strat = np.array([y[qid==q].max() for q in uq])
    probe_score = np.zeros(len(rows))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(uq, strat):
        trq, teq = set(uq[tr]), set(uq[te])
        itr = np.array([i for i,q in enumerate(qid) if q in trq])
        ite = np.array([i for i,q in enumerate(qid) if q in teq])
        if y[itr].sum()==0 or (1-y[itr]).sum()==0: continue
        mu_c, mu_w = H[itr][y[itr]==1].mean(0), H[itr][y[itr]==0].mean(0)
        w = mu_c - mu_w; w = w/ (np.linalg.norm(w)+1e-9)
        probe_score[ite] = H[ite] @ w

    def best_of_n(score, seed=0):
        rng = np.random.default_rng(seed); acc=[]
        for q in uq:
            idx = np.where(qid==q)[0]
            if len(idx)==0: continue
            s = score[idx] if score is not None else rng.random(len(idx))
            acc.append(y[idx][int(np.argmax(s))])
        return float(np.mean(acc))
    oracle = float(np.mean([y[qid==q].max() for q in uq]))
    r_acc = float(np.mean([best_of_n(None, seed=s) for s in range(20)]))
    lp_acc = best_of_n(lp); pr_acc = best_of_n(probe_score)

    from sklearn.metrics import roc_auc_score
    auroc_probe = roc_auc_score(y, probe_score); auroc_lp = roc_auc_score(y, lp)
    print(f"\n{'ranker':28s} {'best-of-8 acc':>14}")
    print(f"{'random (floor)':28s} {r_acc:14.3f}")
    print(f"{'mean logprob (baseline)':28s} {lp_acc:14.3f}")
    print(f"{'workspace probe':28s} {pr_acc:14.3f}")
    print(f"{'oracle (ceiling)':28s} {oracle:14.3f}")
    gap = oracle - r_acc
    print(f"\ngain over baseline: {pr_acc-lp_acc:+.3f}")
    print(f"fraction of the available headroom captured -- "
          f"logprob {(lp_acc-r_acc)/gap:.1%}, probe {(pr_acc-r_acc)/gap:.1%}" if gap>0 else "")
    print(f"AUROC over samples: probe {auroc_probe:.3f} | logprob {auroc_lp:.3f}")
    return {"random":r_acc,"logprob":lp_acc,"probe":pr_acc,"oracle":oracle,
            "auroc_probe":float(auroc_probe),"auroc_logprob":float(auroc_lp),
            "n_questions":len(uq),"n_samples":len(rows)}


@app.local_entrypoint()
def main():
    import json; print(json.dumps(run.remote(), indent=1))
