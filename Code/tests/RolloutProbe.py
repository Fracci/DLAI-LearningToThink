"""
RolloutProbe.py — linear-probe world-model test for the rollout pretrained model.

Freezes the model and trains a linear probe to decode, for each cell, the
row-ABOVE information at offset N (a long-range latent: the predicted cell's
parents one full row back). Compares the trained model against a random-init
control (the empirical floor); the gap is the result. Long-range latents are 
expected to become decodable only in deeper layers, which the sweep makes visible.
"""
import csv
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

from config import ProbeConfig, ROLLOUT_WEIGHTS
from src.Transformer import GeneralTransformer

# CONFIG
CHECKPOINT  = ROLLOUT_WEIGHTS
VOCAB_SIZE  = 3
N           = 24                 # row width = the long-range offset being probed
ROWS        = 8
TARGET      = "cell_above"       # "neighborhood" (8-way) or "cell_above" (2-way)
OUT_CSV     = f"rollout_probe_layers_{TARGET}.csv"
RULE_LUT    = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0], dtype=torch.long)


def make_batch(bs, device):
    """Build a flattened Rule-30 rollout and per-position targets from the row above.

    Returns (seq, full_tgt, n_classes). full_tgt[-1]=-1 marks positions with no
    defined row-above (the first row); those are excluded from the probe.
    """
    row = torch.randint(0, 2, (bs, N), dtype=torch.long)
    grid = [row]
    for _ in range(ROWS - 1):
        left = torch.roll(row, 1, 1); right = torch.roll(row, -1, 1)
        row = RULE_LUT[left * 4 + row * 2 + right]
        grid.append(row)

    grid = torch.stack(grid, dim=1)                       # (bs, ROWS, N)
    seq = grid.reshape(bs, ROWS * N)                      # row-major flatten

    # target = property of the PREVIOUS row (offset N back) for every non-first row
    g_prev = grid[:, :-1, :]
    left = torch.roll(g_prev, 1, 2); right = torch.roll(g_prev, -1, 2)

    if TARGET == "cell_above":
        tgt_rows = g_prev                                 # the single cell directly above
        n_classes = 2
    else:
        tgt_rows = left * 4 + g_prev * 2 + right * 1      # 3-bit parent neighborhood above
        n_classes = 8

    full_tgt = torch.full((bs, ROWS * N), -1, dtype=torch.long)
    full_tgt[:, N:] = tgt_rows.reshape(bs, (ROWS - 1) * N)   # first row has no row above
    return seq.to(device), full_tgt.to(device), n_classes


def capture_layer(model, layer_idx):
    """Register a forward hook that stores layer `layer_idx`'s output (as fp32)."""
    store = {}
    def hook(_m, _i, out):
        store["h"] = out.detach().float()
    handle = model.transformer.layers[layer_idx].register_forward_hook(hook)
    return store, handle


@torch.no_grad()
def feature_stats(model, store, layer_idx, device, n_batches=6):
    """Estimate per-feature mean/std at a layer for standardizing probe inputs."""
    s = ssq = count = 0.0

    for _ in range(n_batches):
        seq, full_tgt, _ = make_batch(ProbeConfig.batch_size, device)
        x = seq[:, :-1]

        with autocast("cuda"):
            _ = model(x)
        h = store["h"]
        valid = (full_tgt[:, 1:] != -1).reshape(-1)
        f = h.reshape(-1, ProbeConfig.d_model)[valid]
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

    _, _, n_classes = make_batch(2, device)
    mean, std = feature_stats(model, store, layer_idx, device)

    probe = nn.Linear(ProbeConfig.d_model, n_classes).to(device)
    opt = AdamW(probe.parameters(), lr=ProbeConfig.lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()

    chance = 100.0 / n_classes
    print(f"\n=== probing {tag} | layer {layer_idx} | target = {TARGET} "
          f"({n_classes}-way, chance {chance:.1f}%) | offset N={N} ===")
    final_acc = 0.0

    for epoch in range(ProbeConfig.epochs):
        probe.train()
        total_loss = 0.0
        correct = total = 0

        for _ in range(ProbeConfig.iters_per_epoch):
            seq, full_tgt, _ = make_batch(ProbeConfig.batch_size, device)
            x = seq[:, :-1]

            with torch.no_grad(), autocast("cuda"):
                _ = model(x)
            h = store["h"]
            targs = full_tgt[:, 1:]                       # align targets to next-token positions
            valid = (targs != -1).reshape(-1)
            f = (h.reshape(-1, ProbeConfig.d_model)[valid] - mean) / std
            y = targs.reshape(-1)[valid]

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
              f"| Probe Accuracy: {final_acc:6.2f}% | classes predicted: {n_pred}/{n_classes}")

    handle.remove()
    return final_acc, chance


def load_rollout(device):
    """Load the rollout-pretrained model, stripping any DataParallel prefix."""
    m = GeneralTransformer(vocab_size=VOCAB_SIZE, d_model=ProbeConfig.d_model, nhead=ProbeConfig.n_heads,
                           num_layers=ProbeConfig.n_layers, dim_feedforward=ProbeConfig.dim_feedforward).to(device)
    sd = torch.load(CHECKPOINT, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m


def main():
    """Sweep all layers, probe trained vs. random-init at each, and write the
    per-layer trained/random/gap table to OUT_CSV."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rollout layer-sweep probe on {device} | target={TARGET} | offset N={N}")

    try:
        trained = load_rollout(device)
        print("Loaded rollout-pretrained weights.")
    except Exception as e:
        print(f"Error loading {CHECKPOINT}: {e}")
        return
    # random-init control built once; same architecture, untrained body.
    rand = GeneralTransformer(vocab_size=VOCAB_SIZE, d_model=ProbeConfig.d_model, nhead=ProbeConfig.n_heads,
                              num_layers=ProbeConfig.n_layers, dim_feedforward=ProbeConfig.dim_feedforward).to(device)

    n_layers = ProbeConfig.n_layers
    rows = [["layer", "trained_acc", "random_acc", "gap", "chance", "target", "offset_N"]]
    print("\n" + "=" * 56)
    print(f"{'layer':<6}{'trained':>10}{'random':>10}{'gap':>10}")
    print("-" * 56)
    chance = None
    for L in range(n_layers):
        acc_t, chance = run_probe(trained, device, L, f"ROLLOUT L{L}")
        acc_r, _      = run_probe(rand,    device, L, f"RANDOM  L{L}")
        gap = acc_t - acc_r
        rows.append([L, f"{acc_t:.2f}", f"{acc_r:.2f}", f"{gap:.2f}", f"{chance:.2f}", TARGET, N])
        print(f"{L:<6}{acc_t:>10.2f}{acc_r:>10.2f}{gap:>+10.2f}")
    print("=" * 56)

    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()