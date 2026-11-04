import numpy as np, json, sys, glob, os
N = sys.argv[1]; key = json.load(open(os.path.join(N, "..", "eval", "answer_key.json")))
CH = ["agent", "moving", "reward", "obstacle", "prevframe"]; ACT = ["noop", "up", "down", "left", "right"]
out = {}
for f in sorted(glob.glob(f"{N}/*.npz")):
    nid = os.path.basename(f)[:-4]; z = np.load(f); game = key[nid]["game"]
    if nid == sorted(glob.glob(f"{N}/*.npz"))[0][-12:-4]: print("npz keys:", list(z.keys()), {k: z[k].shape for k in z})
    W1 = [z[k] for k in z if z[k].shape == (500, 32)][0]; W3 = [z[k] for k in z if z[k].shape == (32, 5)][0]; b3 = [z[k] for k in z if z[k].shape == (5,)][0]
    norms = np.linalg.norm(W1, axis=1).reshape(10, 10, 5)          # per input: row, col, channel
    chan_share = (norms**2).sum((0, 1)) / (norms**2).sum()
    inner = norms[1:9, 1:9, :]
    print(f"\n=== {game} ({nid})  channel energy share: " + ", ".join(f"{c} {s:.2f}" for c, s in zip(CH, chan_share)))
    print("output-layer column norms per action: " + ", ".join(f"{a} {np.linalg.norm(W3[:, i]):.1f}" for i, a in enumerate(ACT)) + f"   b3: {np.round(b3, 2).tolist()}")
    # row/column concentration of agent channel energy within the inner 8x8
    e0 = inner[:, :, 0]**2
    rm = e0.sum(1) / e0.sum(); cm = e0.sum(0) / e0.sum()
    print(f"agent-channel energy by row (1-8): {np.round(rm, 2).tolist()}")
    print(f"agent-channel energy by col (1-8): {np.round(cm, 2).tolist()}")
    maps = {}
    for c in range(5):
        m = inner[:, :, c]; q = np.clip((m / inner.max()) * 9.99, 0, 9).astype(int)
        maps[CH[c]] = ["".join(str(v) for v in row) for row in q]
    print("inner 8x8 first-layer input norm, 0-9 scaled to this network's max; columns = channels " + " | ".join(CH))
    for r in range(8): print("  " + "   ".join(maps[c][r] for c in CH))
    out[nid] = dict(game=game, channel_energy_share=dict(zip(CH, chan_share.round(3).tolist())),
                    action_out_norms=dict(zip(ACT, np.linalg.norm(W3, axis=0).round(2).tolist())), b3=b3.round(3).tolist(),
                    agent_row_share=rm.round(3).tolist(), agent_col_share=cm.round(3).tolist(), maps=maps)
json.dump(out, open(sys.argv[2], "w"), indent=1)
