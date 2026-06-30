"""
Rule30Probe.py — linear-probe world-model test for the Rule 30 pretrained model.

Freezes the model and trains a linear probe on a hidden layer to decode each
cell's local neighborhood [i-2,i-1,i] (the model-aligned window). Compares the
trained model against a random-init control (the empirical floor); the gap is the
result. Sweeps ALL transformer layers and writes per-layer accuracies/gaps to CSV,
so the depth at which the world model emerges is shown rather than assumed.
"""
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.Transformer import GeneralTransformer
from config import ProbeConfig, RULE30_WEIGHTS
from data_generation.Rule30Generator import Rule30Dataset

# CONFIG
CHECKPOINT  = RULE30_WEIGHTS
SEQ_LENGTH  = 256
OUT_CSV     = "rule30_probe_layers.csv"
N_CLASSES   = 8                 # neighborhood is a 3-bit pattern -> 8 classes


def neighborhood_targets(state_t):
    """Per-position 3-bit label of the left-anchored neighborhood [i-2,i-1,i] (0..7)."""
    c0 = torch.roll(state_t, shifts=2, dims=1)
    c1 = torch.roll(state_t, shifts=1, dims=1)
    c2 = state_t
    return (c0 * 4 + c1 * 2 + c2 * 1).long()


def capture_layer(transformer, layer_idx):
    """Register a forward hook that stores layer `layer_idx`'s output (as fp32)."""
    store = {}
    def hook(_m, _i, out):
        store["h"] = out.detach().float()
    handle = transformer.transformer.layers[layer_idx].register_forward_hook(hook)
    return store, handle


def run_probe(transformer, device, layer_idx, tag):
    """Train a linear probe on one layer's activations; return final accuracy."""
    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad = False

    store, handle = capture_layer(transformer, layer_idx)

    probe = nn.Linear(ProbeConfig.d_model, N_CLASSES).to(device)
    opt = AdamW(probe.parameters(), lr=ProbeConfig.lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()

    loader = DataLoader(Rule30Dataset(num_samples=10000, seq_length=SEQ_LENGTH),
                        batch_size=ProbeConfig.batch_size, shuffle=True, pin_memory=True)

    print(f"\nprobing {tag} | layer {layer_idx}")
    final_acc = 0.0
    for epoch in range(ProbeConfig.epochs):
        probe.train()
        total_loss = 0.0
        correct = total = 0
        for state_t, _ in loader:
            state_t = state_t.to(device, non_blocking=True)
            nbr = neighborhood_targets(state_t)

            # forward only to populate the hook; probe trains on the cached features
            with torch.no_grad(), autocast("cuda"):
                _ = transformer(state_t)
            # drop first 2 positions: their left-anchored neighborhood is undefined
            f = store["h"][:, 2:, :]
            y = nbr[:, 2:]

            opt.zero_grad()
            logits = probe(f)
            loss = crit(logits.reshape(-1, N_CLASSES), y.reshape(-1))
            loss.backward()
            opt.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()

        final_acc = 100.0 * correct / total
        n_pred = len(torch.unique(preds))
        print(f"  Probe Epoch [{epoch+1:2d}/{ProbeConfig.epochs}] | Loss: {total_loss/len(loader):.4f} "
              f"| Probe Accuracy: {final_acc:6.2f}% | classes predicted: {n_pred}/{N_CLASSES}")

    handle.remove()
    return final_acc


def load_trained(device):
    """Load the Rule30-pretrained model, stripping any DataParallel prefix."""
    m = GeneralTransformer(vocab_size=2, d_model=ProbeConfig.d_model, nhead=ProbeConfig.n_heads,
                           num_layers=ProbeConfig.n_layers, dim_feedforward=ProbeConfig.dim_feedforward).to(device)
    sd = torch.load(CHECKPOINT, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chance = 100.0 / N_CLASSES
    print(f"Rule30 layer-sweep probe on {device} (chance = {chance:.1f}%)")

    try:
        trained = load_trained(device)
        print("Loaded Rule30-pretrained weights.")
    except Exception as e:
        print(f"Error loading weights: {e}. Run the pretraining script first.")
        return
    # random-init control built once; same architecture, untrained body.
    rand = GeneralTransformer(vocab_size=2, d_model=ProbeConfig.d_model, nhead=ProbeConfig.n_heads,
                              num_layers=ProbeConfig.n_layers, dim_feedforward=ProbeConfig.dim_feedforward).to(device)

    n_layers = ProbeConfig.n_layers
    rows = [["layer", "trained_acc", "random_acc", "gap", "chance"]]
    print("\n" + "=" * 56)
    print(f"{'layer':<6}{'trained':>10}{'random':>10}{'gap':>10}")
    print("-" * 56)
    for L in range(n_layers):
        acc_t = run_probe(trained, device, L, f"TRAINED  L{L}")
        acc_r = run_probe(rand,    device, L, f"RANDOM   L{L}")
        gap = acc_t - acc_r
        rows.append([L, f"{acc_t:.2f}", f"{acc_r:.2f}", f"{gap:.2f}", f"{chance:.2f}"])
        print(f"{L:<6}{acc_t:>10.2f}{acc_r:>10.2f}{gap:>+10.2f}")
    print("=" * 56)

    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"saved -> {OUT_CSV}")


if __name__ == "__main__":
    main()