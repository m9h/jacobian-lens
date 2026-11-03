import json, struct, subprocess, sys, os
url = "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/model.safetensors-00001-of-00001.safetensors"
out = sys.argv[1]
want = ["model.language_model.layers.15.mlp.down_proj.weight","model.language_model.layers.19.mlp.down_proj.weight",
        "model.language_model.embed_tokens.weight","model.language_model.norm.weight"]
hdr8 = subprocess.run(["curl","-sfL","-r","0-7",url],capture_output=True).stdout
n = struct.unpack("<Q", hdr8)[0]
hdr = subprocess.run(["curl","-sfL","-r",f"8-{8+n-1}",url],capture_output=True).stdout
meta = json.loads(hdr)
base = 8+n
for k in want:
    t = meta[k]; s,e = t["data_offsets"]
    fn = os.path.join(out, k+".bin")
    print(k, t["dtype"], t["shape"], e-s, "bytes", flush=True)
    subprocess.run(["curl","-sfL","-r",f"{base+s}-{base+e-1}",url,"-o",fn],check=True)
    assert os.path.getsize(fn)==e-s, fn
json.dump({k:meta[k] for k in want}, open(os.path.join(out,"meta.json"),"w"))
print("done")
