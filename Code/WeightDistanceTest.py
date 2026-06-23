"""
TEST 1 — Did Model A forget its pretraining?

Compares the transformer-body weights of three checkpoints:
  - pretrained  (rule30_pretrained_new.pt)   = A's initialization
  - A_after     (fine-tuned on arithmetic, started from pretrained)
  - B_after     (fine-tuned on arithmetic, started from random)

Logic: gradient descent does not permute neurons, so a model stays
directionally correlated with its OWN init. B never saw the pretrained
weights, so cos(B_after, pretrained) is the "unrelated model" baseline.

  If A forgot everything   -> cos(A_after, pretrained) ~= cos(B_after, pretrained)
  If A retained structure   -> cos(A_after, pretrained)  >  cos(B_after, pretrained)

Only transformer.* and final_norm.* are compared. embedding.* / fc_out.*
are skipped (re-initialized for the new vocab, so meaningless here).
No model class or GPU needed.
"""
import csv
import torch
import torch.nn.functional as F
from collections import defaultdict

# --- edit checkpoint names if needed ---
PRETRAINED = "rule30_rollout_pretrained.pt"
A_AFTER    = "seed0_modelA.pt"
B_AFTER    = "seed0_modelB.pt"
# ---------------------------------------


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
    # transformer.layers.N.<...>  -> "layer N" ; final_norm -> "final_norm"
    if k.startswith("transformer.layers."):
        return f"layer {k.split('.')[2]}"
    return "final_norm"


def main():
    pre, A, B = load(PRETRAINED), load(A_AFTER), load(B_AFTER)
    keys = [k for k in pre if is_body(k) and k in A and k in B]
    if not keys:
        print("No shared transformer.*/final_norm.* keys found — check checkpoint names.")
        return

    # per-layer aggregation of the three pairwise cosines
    agg = defaultdict(lambda: defaultdict(list))
    rows = []
    for k in keys:
        cA = cos(A[k], pre[k])     # A_after vs its own init
        cB = cos(B[k], pre[k])     # B_after vs that init (baseline)
        cAB = cos(A[k], B[k])      # the two fine-tuned models vs each other
        lA = rel_l2(A[k], pre[k])  # how far A moved from init
        lg = layer_of(k)
        agg[lg]["A_pre"].append(cA)
        agg[lg]["B_pre"].append(cB)
        agg[lg]["A_B"].append(cAB)
        rows.append([k, f"{cA:.4f}", f"{cB:.4f}", f"{cAB:.4f}", f"{lA:.4f}"])

    # global cosine over ALL body params concatenated (the headline number)
    def flat(sd):
        return torch.cat([sd[k].flatten() for k in keys])
    g_A_pre = cos(flat(A), flat(pre))
    g_B_pre = cos(flat(B), flat(pre))
    g_A_B = cos(flat(A), flat(B))

    print("=" * 64)
    print("Per-layer mean cosine similarity")
    print(f"{'layer':<12}{'cos(A,pre)':>12}{'cos(B,pre)':>12}{'cos(A,B)':>12}")
    print("-" * 64)
    def order(lg):
        return 999 if lg == "final_norm" else int(lg.split()[1])
    for lg in sorted(agg, key=order):
        a = sum(agg[lg]["A_pre"]) / len(agg[lg]["A_pre"])
        b = sum(agg[lg]["B_pre"]) / len(agg[lg]["B_pre"])
        ab = sum(agg[lg]["A_B"]) / len(agg[lg]["A_B"])
        print(f"{lg:<12}{a:>12.4f}{b:>12.4f}{ab:>12.4f}")
    print("-" * 64)
    print(f"{'GLOBAL':<12}{g_A_pre:>12.4f}{g_B_pre:>12.4f}{g_A_B:>12.4f}")
    print("=" * 64)
    print("\nReading it:")
    print(f"  cos(A_after, pretrained) = {g_A_pre:.4f}")
    print(f"  cos(B_after, pretrained) = {g_B_pre:.4f}   <- unrelated-model baseline")
    if g_A_pre > g_B_pre + 0.05:
        print("  => A stays markedly closer to the pretrained init than B does:")
        print("     pretraining was NOT erased — structure was retained.")
    else:
        print("  => A is about as close to the pretrained init as B is:")
        print("     consistent with heavy forgetting (benefit, if any, is basin-level).")

    with open("weight_distance0.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "cos_A_pre", "cos_B_pre", "cos_A_B", "relL2_A_pre"])
        w.writerows(rows)
    print("\nsaved -> weight_distance0.csv")


if __name__ == "__main__":
    main()