import json, sys
r = json.load(open(sys.argv[1]))
# GPT-2 byte-level decoder
bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256+n); n += 1
dec = {chr(c): b for b, c in zip(bs, cs)}
def d(t):
    t = t.replace(" ", "Ġ").replace("\\n", "Ċ")
    try: return bytes(dec[c] for c in t).decode("utf-8", "replace")
    except KeyError: return t
print("cos(u15,u19) =", round(r["cos_u15_u19"], 4))
for k, v in r["unembed"].items():
    print(k, "+:", " | ".join(repr(d(t)) for t in v["plus"][:18]))
    print(k, "-:", " | ".join(repr(d(t)) for t in v["minus"][:18]))
