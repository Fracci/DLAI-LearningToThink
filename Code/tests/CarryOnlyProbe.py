"""
CarryOnlyProbe.py — linear-probe world-model test for the carry-only pretrained model.

Freezes the model and trains a linear probe to decode a carry latent at each query
position: either carry_in (2-way) or gen_dist, the distance back to the carry's
generating cell (GEN_DIST_MAX+1 classes). Compares the trained model against a
random-init control (the empirical floor); the gap is the result, NOT raw accuracy,
since both latents have high floors.
"""
import csv
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.Transformer import GeneralTransformer
from config import ProbeConfig, CARRYONLY_WEIGHTS
from data_generation.CarryOnlyGenerator import sample_ab, assemble, VOCAB, IGNORE, TARGET_ACTIVE, GEN_DIST_MAX

# CONFIG
CHECKPOINT  = "carryonly_pretrained.pt"
MIN_N, MAX_N = 8, 24
CHAIN_MAX   = 12
MAX_LEN     = 3 * MAX_N + 2
TARGET      = "gen_dist"      # "carry_in" (2-way) or "gen_dist" (GEN_DIST_MAX+1 way)
OUT_CSV     = f"carry_probe_layers_{TARGET}.csv"


def make_batch(bs, device):
    """Sample a batch of carry-only sequences with the chosen latent as the target."""
    seqs, targs = [], []

    for _ in range(bs):
        n = random.randint(MIN_N, MAX_N)
        a, b = sample_ab(n, CHAIN_MAX, TARGET_ACTIVE)
        seq, tgt = assemble(a, b, MAX_LEN, latent=TARGET)
        seqs.append(seq); targs.append(tgt)

    return torch.stack(seqs).to(device), torch.stack(targs).to(device)


def capture_layer(model, layer_idx):
    """Register a forward hook that stores layer `layer_idx`'s output (as fp32)."""
    store = {}

    def hook(_m, _i, out):
        store["h"] = out.detach().float()

    handle = model.transformer.layers[layer_idx].register_forward_hook(hook)
    return store, handle


def n_classes_for(target):
    """Class count for the probe: carry_in is binary; gen_dist is the clamped range."""
    return 2 if target == "carry_in" else (GEN_DIST_MAX + 1)


@torch.no_grad()
def feature_stats(model, store, device, n_batches=6):
    """Estimate per-feature mean/std at the hooked layer for standardizing inputs."""
    s = ssq = count = 0.0

    for _ in range(n_batches):
        seq, tgt = make_batch(ProbeConfig.batch_size, device)
        with autocast("cuda"):
            _ = model(seq)

        valid = (tgt != IGNORE).reshape(-1)
        f = store["h"].reshape(-1, ProbeConfig.d_model)[valid]
        s = s + f.sum(0); ssq = ssq + (f * f).sum(0); count += f.shape[0]

    mean = s / count
    std = torch.sqrt((ssq / count - mean ** 2).clamp_min(1e-6))
    return mean, std


def run_probe(model, device, layer_idx, tag):
    """Train a standardized linear probe on one layer's activations; return accuracy."""
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    store, handle = capture_layer(model, layer_idx)
    nc = n_classes_for(TARGET)
    mean, std = feature_stats(model, store, device)

    probe = nn.Linear(ProbeConfig.d_model, nc).to(device)
    opt = AdamW(probe.parameters(), lr=ProbeConfig.lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    chance = 100.0 / nc

    print(f"\n=== probing {tag} | layer {layer_idx} | target = {TARGET} "
          f"({nc}-way, chance {chance:.1f}%) ===")
    final_acc = 0.0

    for epoch in range(ProbeConfig.epochs):
        probe.train()
        total_loss = 0.0
        correct = total = 0

        for _ in range(ProbeConfig.iters_per_epoch):
            seq, tgt = make_batch(ProbeConfig.batch_size, device)
            with torch.no_grad(), autocast("cuda"):
                _ = model(seq)
            valid = (tgt != IGNORE).reshape(-1)
            f = (store["h"].reshape(-1, ProbeConfig.d_model)[valid] - mean) / std
            y = tgt.reshape(-1)[valid]

            opt.zero_grad()
            logits = probe(f)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()

        final_acc = 100.0 * correct / total
        n_pred = len(torch.unique(preds))
        print(f"  Probe Epoch [{epoch+1:2d}/{ProbeConfig.epochs}] | Loss: {total_loss/ProbeConfig.iters_per_epoch:.4f} "
              f"| Probe Accuracy: {final_acc:6.2f}% | classes predicted: {n_pred}/{nc}")

    handle.remove()
    return final_acc, chance


def load_carry(device):
    """Load the carry-only pretrained model, stripping any DataParallel prefix."""
    m = GeneralTransformer(vocab_size=VOCAB, d_model=ProbeConfig.d_model, nhead=ProbeConfig.n_heads,
                           num_layers=ProbeConfig.n_layers, dim_feedforward=ProbeConfig.dim_feedforward).to(device)
    sd = torch.load(CHECKPOINT, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m


def main():
    """Sweep all layers, probe trained vs. random-init at each, and write the
    per-layer trained/random/gap table to OUT_CSV."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Carry-only layer-sweep probe on {device} | target={TARGET}")

    try:
        trained = load_carry(device)
        print("Loaded carry-only pretrained weights.")
    except Exception as e:
        print(f"Error loading {CHECKPOINT}: {e}")
        return
    
    # random-init control built once; same architecture, untrained body — this is
    # the empirical floor that turns raw probe accuracy into a meaningful gap.
    rand = GeneralTransformer(vocab_size=VOCAB, d_model=ProbeConfig.d_model, nhead=ProbeConfig.n_heads,
                              num_layers=ProbeConfig.n_layers, dim_feedforward=ProbeConfig.dim_feedforward).to(device)

    n_layers = ProbeConfig.n_layers
    rows = [["layer", "trained_acc", "random_acc", "gap", "chance", "target", "gen_dist_max"]]
    print("\n" + "=" * 56)
    print(f"{'layer':<6}{'trained':>10}{'random':>10}{'gap':>10}")
    print("-" * 56)
    chance = None
    
    for L in range(n_layers):
        acc_t, chance = run_probe(trained, device, L, f"CARRY  L{L}")
        acc_r, _      = run_probe(rand,    device, L, f"RANDOM L{L}")
        gap = acc_t - acc_r
        gdm = GEN_DIST_MAX if TARGET == "gen_dist" else ""
        rows.append([L, f"{acc_t:.2f}", f"{acc_r:.2f}", f"{gap:.2f}", f"{chance:.2f}", TARGET, gdm])
        print(f"{L:<6}{acc_t:>10.2f}{acc_r:>10.2f}{gap:>+10.2f}")
    print("=" * 56)

    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()