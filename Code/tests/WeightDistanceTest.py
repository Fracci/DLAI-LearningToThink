import csv
import os
import torch
import torch.nn.functional as F
from collections import defaultdict

# CONFIG
PRETRAINED   = "carryonly_pretrained.pt"     
SEEDS        = [0, 1, 2, 3, 4]
A_PATTERN    = "Weights/Carryonly_seed{seed}_modelA.pt"
B_PATTERN    = "Weights/seed{seed}_modelB.pt"    
OUT_CSV      = "weight_distance_carryonly.csv"


def load(path):
    sd = torch.load(path, map_location="cpu")
    return {k.replace("module.", ""): v.float() for k, v in sd.items()}


def is_body(k):
    return k.startswith("transformer.") or k.startswith("final_norm.")


def cos(a, b):
    return F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()


def rel_l2(a, b):
    return (torch.norm(a - b) / (torch.norm(b) + 1e-12)).item()


def layer_of(k):
    if k.startswith("transformer.layers."):
        return f"layer {k.split('.')[2]}"
    return "final_norm"


def order(lg):
    return 999 if lg == "final_norm" else int(lg.split()[1])


def run_one_seed(seed, pre):
    a_path = A_PATTERN.format(seed=seed)
    b_path = B_PATTERN.format(seed=seed)
    if not (os.path.exists(a_path) and os.path.exists(b_path)):
        print(f"  [seed {seed}] missing checkpoint(s): "
              f"{'' if os.path.exists(a_path) else a_path} "
              f"{'' if os.path.exists(b_path) else b_path}".rstrip())
        return None
    A, B = load(a_path), load(b_path)
    keys = [k for k in pre if is_body(k) and k in A and k in B]
    if not keys:
        print(f"  [seed {seed}] no shared body keys — check checkpoints.")
        return None

    per_layer = defaultdict(lambda: defaultdict(list))
    per_param = []
    for k in keys:
        cA = cos(A[k], pre[k]); cB = cos(B[k], pre[k]); cAB = cos(A[k], B[k])
        lA = rel_l2(A[k], pre[k]); lg = layer_of(k)
        per_layer[lg]["A_pre"].append(cA)
        per_layer[lg]["B_pre"].append(cB)
        per_layer[lg]["A_B"].append(cAB)
        per_layer[lg]["relL2"].append(lA)
        per_param.append([seed, k, f"{cA:.4f}", f"{cB:.4f}", f"{cAB:.4f}", f"{lA:.4f}"])

    def flat(sd): return torch.cat([sd[k].flatten() for k in keys])
    glob = {"A_pre": cos(flat(A), flat(pre)),
            "B_pre": cos(flat(B), flat(pre)),
            "A_B":   cos(flat(A), flat(B))}
    # collapse per-layer lists to means for this seed
    layer_means = {lg: {m: sum(v) / len(v) for m, v in d.items()} for lg, d in per_layer.items()}
    return layer_means, glob, per_param


def mean_std(xs):
    if not xs: return (float("nan"), float("nan"))
    m = sum(xs) / len(xs)
    if len(xs) == 1: return (m, 0.0)
    return (m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5)


def main():
    if not os.path.exists(PRETRAINED):
        print(f"Pretrained checkpoint not found: {PRETRAINED}")
        return
    pre = load(PRETRAINED)

    # accumulate per-seed results
    layer_acc = defaultdict(lambda: defaultdict(list))
    glob_acc = defaultdict(list)
    all_param_rows = []
    seeds_done = []

    print(f"Weight-distance over seeds {SEEDS} | init = {PRETRAINED}\n")
    for s in SEEDS:
        res = run_one_seed(s, pre)
        if res is None:
            continue
        layer_means, glob, per_param = res
        seeds_done.append(s)
        for lg, d in layer_means.items():
            for m, v in d.items():
                layer_acc[lg][m].append(v)
        for m, v in glob.items():
            glob_acc[m].append(v)
        all_param_rows.extend(per_param)
        print(f"  [seed {s}] global cos(A,pre)={glob['A_pre']:.4f}  "
              f"cos(B,pre)={glob['B_pre']:.4f}  cos(A,B)={glob['A_B']:.4f}")

    if not seeds_done:
        print("\nNo seeds processed — fix the A_PATTERN/B_PATTERN filenames.")
        return

    print("\n" + "=" * 78)
    print(f"Per-layer cosine, mean +/- std over {len(seeds_done)} seeds {seeds_done}")
    print(f"{'layer':<12}{'cos(A,pre)':>16}{'cos(B,pre)':>16}{'cos(A,B)':>16}{'relL2(A,pre)':>16}")
    print("-" * 78)
    summary_rows = [["layer", "cosA_pre_mean", "cosA_pre_std",
                     "cosB_pre_mean", "cosB_pre_std",
                     "cosA_B_mean", "cosA_B_std",
                     "relL2_mean", "relL2_std"]]
    for lg in sorted(layer_acc, key=order):
        am, asd = mean_std(layer_acc[lg]["A_pre"])
        bm, bsd = mean_std(layer_acc[lg]["B_pre"])
        abm, absd = mean_std(layer_acc[lg]["A_B"])
        lm, lsd = mean_std(layer_acc[lg]["relL2"])
        print(f"{lg:<12}{am:>9.3f}±{asd:<5.3f}{bm:>9.3f}±{bsd:<5.3f}"
              f"{abm:>9.3f}±{absd:<5.3f}{lm:>9.3f}±{lsd:<5.3f}")
        summary_rows.append([lg, f"{am:.4f}", f"{asd:.4f}", f"{bm:.4f}", f"{bsd:.4f}",
                             f"{abm:.4f}", f"{absd:.4f}", f"{lm:.4f}", f"{lsd:.4f}"])
    print("-" * 78)
    gA = mean_std(glob_acc["A_pre"]); gB = mean_std(glob_acc["B_pre"]); gAB = mean_std(glob_acc["A_B"])
    print(f"{'GLOBAL':<12}{gA[0]:>9.3f}±{gA[1]:<5.3f}{gB[0]:>9.3f}±{gB[1]:<5.3f}"
          f"{gAB[0]:>9.3f}±{gAB[1]:<5.3f}")
    summary_rows.append(["GLOBAL", f"{gA[0]:.4f}", f"{gA[1]:.4f}", f"{gB[0]:.4f}", f"{gB[1]:.4f}",
                         f"{gAB[0]:.4f}", f"{gAB[1]:.4f}", "", ""])
    print("=" * 78)

    print("\nReading it:")
    print(f"  cos(A_after, pretrained) = {gA[0]:.4f} ± {gA[1]:.4f}")
    print(f"  cos(B_after, pretrained) = {gB[0]:.4f} ± {gB[1]:.4f}   <- unrelated-model baseline")
    if gA[0] > gB[0] + 0.05:
        print("  => A stays markedly closer to its pretrained init than B does:")
        print("     pretraining was NOT erased — directional structure retained.")
    else:
        print("  => A is about as close to the init as B is: heavy forgetting")
        print("     (any transfer benefit is basin/bias-level, not a preserved circuit).")

    # write both the per-seed/per-param detail and the seed-averaged summary
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "param", "cos_A_pre", "cos_B_pre", "cos_A_B", "relL2_A_pre"])
        w.writerows(all_param_rows)
    summ_path = OUT_CSV.replace(".csv", "_summary.csv")
    with open(summ_path, "w", newline="") as f:
        csv.writer(f).writerows(summary_rows)
    print(f"\nsaved -> {OUT_CSV} (per-seed per-param), {summ_path} (seed-averaged per-layer)")


if __name__ == "__main__":
    main()