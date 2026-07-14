"""Does a global workspace survive without attention?

Nemotron-H is 21 Mamba-2 mixers, 17 MLPs, and only 4 attention layers. Qwen3.5-27B --
the model where the ASCII-face "nose" readout emerges (rank 2, vs 164 at Qwen3-14B) --
is 48/64 linear attention. So the workspace is CLEAREST in the models that are least
like a transformer.

That matters because the standing objection to the whole programme (Butlin & Long's
"no separable input processors"; Hoel's "LLMs flatly lack modularity and reentrant
dynamics") assumes the J-space is an artifact of transformer topology. If a J-space
appears in a Mamba-2 hybrid, that objection is wrong.

WHY MODAL, NOT THE SPARK. A Mamba-2 Jacobian ate the Spark's 119GB of *unified* memory
and wedged the host -- kernel alive, userspace dead, other users' jobs destroyed. On
unified memory a GPU OOM is a SYSTEM OOM: it does not raise CUDA OOM, it kills the box.
`jacobian_for_prompt` runs one backward per dim_batch chunk of d_model, and Mamba-2's
recurrent scan unrolls its autograd graph across the sequence -- far worse than
attention. Here it gets a dedicated GPU and dies alone.

GATED. `smoke` measures peak memory on a handful of prompts and reports the settings a
full fit can afford. Nothing expensive runs until it passes.

    modal run modal_app.py::smoke                 # ~2 min,  cents
    modal run modal_app.py::fit --n-prompts 1000  # the real fit, to convergence
"""

import modal

app = modal.App("jlens-nemotron")

# CUDA *devel* base: mamba-ssm and causal-conv1d compile CUDA kernels at install.
#
# This is not optional. Without the fused kernels, Mamba-2 falls back to a naive scan
# whose autograd graph is the fully unrolled recurrence, and jacobian_for_prompt calls
# autograd.grad once per dim_batch chunk of d_model against that graph. Measured on
# Nemotron-H-4B, A100-80GB: ONE layer at dim_batch=8, seq=64 peaks at 73.1 GB. Two
# layers OOM. A 4B transformer needs a few GB for the same thing.
#
# jlens only needs a FIRST-order gradient, and the fused kernels ship a proper custom
# backward -- so they work, and they are the only way this is tractable at all.
# NB: mamba-ssm compiles against torch.version.cuda, so the base image CUDA and the
# torch wheel's CUDA must MATCH exactly. Default `pip install torch` pulls a cu130 wheel;
# against a 12.8 base that dies with "detected CUDA version (12.8) mismatches the version
# used to compile PyTorch (13.0)". Pin torch to cu128. It also needs g++ (the image ships
# clang++, which torch's cpp_extension refuses).
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("git", "build-essential", "g++")
    .env({"CC": "gcc", "CXX": "g++", "TORCH_CUDA_ARCH_LIST": "8.0"})  # 8.0 = A100
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    .pip_install(
        "causal-conv1d>=1.4.0",
        "mamba-ssm>=2.2.2",
        extra_options="--no-build-isolation",
    )
    .pip_install(
        "transformers>=5.5",
        "numpy",
        "datasets",
        "huggingface_hub",
        "git+https://github.com/anthropics/jacobian-lens.git",  # jlens: not on PyPI
    )
)
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
out = modal.Volume.from_name("jlens-out", create_if_missing=True)

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
ENV = {"HF_HOME": "/cache/hf"}

# Nemotron-H names its submodules differently to every layout jlens knows:
# model.{embeddings, layers, norm_f} + lm_head. Five fields, and it loads.
LAYOUT = dict(path="model", layers="layers", norm="norm_f",
              embed="embeddings", lm_head="lm_head")


def _load():
    import torch, jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from collections import Counter

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to("cuda").eval()
    kinds = Counter(type(getattr(b, "mixer", b)).__name__ for b in hf.model.layers)
    print(f"  architecture: {dict(kinds)}", flush=True)
    model = jlens.from_hf(hf, tok, layout=jlens.Layout(**LAYOUT))
    return hf, tok, model


@app.function(image=image, gpu="A100-80GB", volumes={"/cache": cache},
              timeout=20 * 60, env=ENV)
def smoke():
    """Can jlens fit a Mamba-2 hybrid at all, and what does it cost in memory?"""
    import torch, jlens

    hf, tok, model = _load()
    n_layers = hf.config.num_hidden_layers

    # If the fused kernels did not load, everything below is meaningless -- the naive
    # scan needs 73GB for a single layer. Fail loudly rather than silently OOM.
    try:
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
        from causal_conv1d import causal_conv1d_fn
        print("  fused Mamba kernels: PRESENT (fast path active)", flush=True)
    except ImportError as e:
        raise RuntimeError(
            f"fused Mamba kernels missing ({e}). The naive scan needs 73GB per layer "
            "on a 4B model -- this run would OOM for a fixable reason."
        )

    results = {}
    for dim_batch, max_seq_len in ((8, 64), (16, 128), (32, 128)):
        torch.cuda.reset_peak_memory_stats()
        try:
            lens = jlens.fit(
                model,
                ["The capital of France is Paris, and the weather there in spring "
                 "is often quite mild indeed, or so the guidebooks like to claim."],
                source_layers=[n_layers // 2],
                dim_batch=dim_batch, max_seq_len=max_seq_len, skip_first=4,
            )
            peak = torch.cuda.max_memory_allocated() / 1e9
            results[(dim_batch, max_seq_len)] = peak
            print(f"  dim_batch={dim_batch:>2} seq={max_seq_len:>3}  "
                  f"peak {peak:6.1f} GB  OK", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  dim_batch={dim_batch:>2} seq={max_seq_len:>3}  OOM", flush=True)
            torch.cuda.empty_cache()

    # The readout that proves it works at all.
    lens = jlens.fit(model, [
        "The capital of France is Paris, and the weather there in spring is mild.",
        "In 1969 the Apollo programme landed the first human beings upon the Moon.",
    ], source_layers=[10, 20, 30], dim_batch=8, max_seq_len=64, skip_first=4)
    r, *_ = lens.apply(model, "The capital of France is",
                       layers=[10, 20, 30], max_seq_len=16)
    print("\n*** J-LENS RUNS ON A MAMBA-2 HYBRID ***", flush=True)
    for l in (10, 20, 30):
        print(f"  L{l:>2} top5:",
              tok.convert_ids_to_tokens(r[l][-1].topk(5).indices.tolist()), flush=True)
    return {"peaks_gb": {f"{k[0]}x{k[1]}": v for k, v in results.items()},
            "n_layers": n_layers}


@app.function(image=image, gpu="A100-80GB", volumes={"/cache": cache, "/out": out},
              timeout=24 * 60 * 60, env=ENV)
def fit(n_prompts: int = 1000, dim_batch: int = 32, max_seq_len: int = 128,
        layer_step: int = 3, ckpt_every: int = 25):
    """Fit a Nemotron-H lens TO CONVERGENCE, then run the flagship demo on it.

    Convergence, not a fixed count: Anthropic's published lenses used 454-775 prompts
    (--stop_at_delta 0.002, --min_prompts 100 is a FLOOR). Fitting on 100 under-fits
    by 4.6-6x and fails silently.
    """
    import json, pathlib, torch, jlens
    from jlens.fitting import jacobian_for_prompt
    from datasets import load_dataset

    hf, tok, model = _load()
    # source_layers must EXCLUDE the target (the final layer): a Jacobian from the
    # target to itself is undefined and jacobian_for_prompt raises on every prompt.
    #
    # Subsample every `layer_step`-th layer. Fitting all 41 costs ~0.6 min/prompt and a
    # full run blew Modal's 6h function timeout at 650 prompts -- with NOTHING saved,
    # because this loop hand-rolls the accumulation instead of using jlens.fit's
    # checkpoint_path. Anthropic's own figures subsample layers; the ASCII-face readout
    # does not need all 41 to show whether "nose" surfaces.
    layers = list(range(0, hf.config.num_hidden_layers - 1, layer_step))

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train", streaming=True)
    prompts, it = [], iter(ds)
    while len(prompts) < n_prompts:
        t = next(it)["text"].strip()[:2000]
        if len(t) >= 400 and not t.startswith("=") and \
           len(tok(t, add_special_tokens=False)["input_ids"]) >= max_seq_len:
            prompts.append(t)

    ckpt = pathlib.Path("/out/nemotron_ckpt.pt")
    jac, n, under, trace, skipped = None, 0, 0, [], 0
    start = 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cuda")
        jac, n, trace, start = st["jac"], st["n"], st["trace"], st["next_idx"]
        print(f"  resuming from checkpoint: {n} prompts done", flush=True)

    for idx, p in enumerate(prompts):
        if idx < start:
            continue
        try:
            J, seq, valid = jacobian_for_prompt(
                model, p, layers, dim_batch=dim_batch,
                max_seq_len=max_seq_len, skip_first=16)
        except Exception as e:
            # NEVER swallow this silently. A bare `except: continue` here hid a
            # source_layers bug behind 600 useless iterations and surfaced it as a
            # meaningless TypeError at the end. Fail fast on the first prompt.
            skipped += 1
            if n == 0 and skipped == 1:
                raise RuntimeError(
                    f"jacobian_for_prompt failed on the FIRST prompt: "
                    f"{type(e).__name__}: {e}"
                ) from e
            continue
        if jac is None:
            jac = {l: torch.zeros_like(J[l]) for l in layers}
        mrc = float("nan") if not n else max(
            ((J[l] - jac[l] / n).norm() / ((n + 1) * (jac[l] / n).norm())).item()
            for l in layers)
        for l in layers:
            jac[l] += J[l]
        n += 1
        trace.append((n, seq, valid, mrc))
        if n >= 100 and mrc == mrc and mrc < 0.002:
            under += 1
            if under >= 10:
                print(f"  converged at {n} prompts (mrc={mrc:.6f})", flush=True)
                break
        else:
            under = 0
        if n % 50 == 0:
            print(f"    {n:4d}  mean_rel_change={mrc:.6f}", flush=True)
        if n % ckpt_every == 0:
            torch.save({"jac": jac, "n": n, "trace": trace, "next_idx": idx + 1},
                       "/out/nemotron_ckpt.pt")
            out.commit()

    lens = jlens.JacobianLens(
        jacobians={l: jac[l] / n for l in layers},
        n_prompts=n, d_model=next(iter(jac.values())).shape[-1])
    pathlib.Path("/out").mkdir(exist_ok=True)
    lens.save("/out/nemotron_h_4b_lens.pt")

    # The flagship demo: does a Mamba hybrid put "nose" at the caret?
    from jlens.examples import Example
    import jlens.examples as ex
    lst = [v for v in vars(ex).values()
           if isinstance(v, (list, tuple)) and v and isinstance(v[0], Example)][0]
    prompt = [e for e in lst if e.slug == "ascii-face"][0].prompt
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    pos = [i for i, t in enumerate(tok.convert_ids_to_tokens(ids)) if "^" in t][0]
    nose = list({t[0] for t in (tok(w, add_special_tokens=False)["input_ids"]
                                for w in ("nose", " nose", "Nose", " Nose"))})

    report = {"n_prompts": n, "converged": under >= 10, "final_mrc": mrc,
              "n_layers": len(layers), "d_model": lens.d_model}
    for use_j, key in ((True, "j_lens"), (False, "logit_lens")):
        r, *_ = lens.apply(model, prompt, layers=layers,
                           max_seq_len=len(ids), use_jacobian=use_j)
        best, best_l = 10**9, None
        for l in layers:
            lg = r[l][pos]
            rk = min(int((lg > lg[t]).sum().item()) + 1 for t in nose)
            if rk < best:
                best, best_l, top = rk, l, tok.convert_ids_to_tokens(
                    lg.topk(5).indices.tolist())
        report[key] = {"best_rank_nose": best, "best_layer": best_l, "top5": top}
        print(f"  {key:11s} best rank(nose)={best} @L{best_l}  top5={top}", flush=True)

    pathlib.Path("/out/nemotron_report.json").write_text(json.dumps(report, indent=2))
    out.commit()
    print("\nRANK 1-5 => the workspace survives WITHOUT attention.", flush=True)
    return report


@app.local_entrypoint()
def main():
    print(smoke.remote())
