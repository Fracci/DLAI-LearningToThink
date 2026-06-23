"""
Othello-GPT-style world-model probe for Rule 30 (alignment-corrected).

Key fix vs the textbook version: the pretraining uses shifted targets
(roll(shifts=1)), so the model's output at position i is conditioned on cells
[i-2, i-1, i] -- it NEVER sees the right neighbor i+1. We therefore probe for
the neighborhood the model ACTUALLY conditions on, not the canonical one.

    NEIGHBORHOOD = "model_aligned"  -> [i-2, i-1, i]   (default, matches training)
    NEIGHBORHOOD = "textbook"       -> [i-1, i, i+1]   (canonical, for contrast)

Probes an intermediate layer for this latent (never supervised) variable with a
linear probe; random-init control = the trivially-decodable floor.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast

from Transformer import Rule30Transformer
from Rule30Generator import Rule30Dataset

# --- config ---
CHECKPOINT  = "rule30_pretrained_new.pt"
D_MODEL, NHEAD, NUM_LAYERS, DIM_FF = 256, 8, 6, 1024
SEQ_LENGTH  = 256
BATCH_SIZE  = 128
PROBE_EPOCHS = 25
LR = 3e-4
PROBE_LAYER = 3                 # intermediate layer (0..5). Try 2,3,4.
NEIGHBORHOOD = "model_aligned"  # "model_aligned" = [i-2,i-1,i] ; "textbook" = [i-1,i,i+1]
# --------------


def neighborhood_targets(state_t):
    """Per-cell 3-bit neighborhood value 0..7. Never a training target."""
    if NEIGHBORHOOD == "model_aligned":
        c0 = torch.roll(state_t, shifts=2, dims=1)   # i-2
        c1 = torch.roll(state_t, shifts=1, dims=1)   # i-1
        c2 = state_t                                 # i
    elif NEIGHBORHOOD == "textbook":
        c0 = torch.roll(state_t, shifts=1,  dims=1)  # i-1
        c1 = state_t                                 # i
        c2 = torch.roll(state_t, shifts=-1, dims=1)  # i+1
    else:
        raise ValueError(NEIGHBORHOOD)
    return (c0 * 4 + c1 * 2 + c2 * 1).long()


def capture_layer(transformer, layer_idx):
    store = {}
    def hook(_m, _i, out):
        store["h"] = out.detach().float()    # fp32 features
    handle = transformer.transformer.layers[layer_idx].register_forward_hook(hook)
    return store, handle


def run_probe(transformer, device, tag):
    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad = False

    store, handle = capture_layer(transformer, PROBE_LAYER)

    probe = nn.Linear(D_MODEL, 8).to(device)      # linear, 8-way
    opt = AdamW(probe.parameters(), lr=LR, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()

    loader = DataLoader(Rule30Dataset(num_samples=10000, seq_length=SEQ_LENGTH),
                        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    print(f"\n=== probing {tag} | layer {PROBE_LAYER} | target = {NEIGHBORHOOD} neighborhood (0-7) ===")
    final_acc = 0.0
    for epoch in range(PROBE_EPOCHS):
        probe.train()
        total_loss = 0.0
        correct = total = 0
        for state_t, _ in loader:
            state_t = state_t.to(device, non_blocking=True)
            nbr = neighborhood_targets(state_t)

            with torch.no_grad(), autocast("cuda"):
                _ = transformer(state_t)
            f = store["h"][:, 2:, :]                  # drop 2 causally-blind edge tokens
            y = nbr[:, 2:]

            opt.zero_grad()
            logits = probe(f)                         # fp32 probe (no autocast)
            loss = crit(logits.reshape(-1, 8), y.reshape(-1))
            loss.backward()
            opt.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()

        final_acc = 100.0 * correct / total
        n_classes = len(torch.unique(preds))
        print(f"  Probe Epoch [{epoch+1:2d}/{PROBE_EPOCHS}] | Loss: {total_loss/len(loader):.4f} "
              f"| Probe Accuracy: {final_acc:6.2f}% | classes predicted: {n_classes}/8")

    handle.remove()
    return final_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Probing Experiment on device: {device}  (chance = 12.5%)")

    transformer = Rule30Transformer(vocab_size=2, d_model=D_MODEL, nhead=NHEAD,
                                    num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)
    try:
        sd = torch.load(CHECKPOINT, map_location=device)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        transformer.load_state_dict(sd)
        print("Successfully loaded pre-trained Transformer weights.")
    except Exception as e:
        print(f"Error loading weights: {e}. Make sure you ran the training script first!")
        return

    acc_trained = run_probe(transformer, device, "TRAINED model")

    # random-init control: the trivially-decodable floor
    rand = Rule30Transformer(vocab_size=2, d_model=D_MODEL, nhead=NHEAD,
                             num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)
    acc_random = run_probe(rand, device, "RANDOM-init control")

    print("\n" + "=" * 60)
    print(f"Neighborhood linear-probe accuracy @ layer {PROBE_LAYER} (target={NEIGHBORHOOD})")
    print(f"  trained model : {acc_trained:6.2f}%")
    print(f"  random control: {acc_random:6.2f}%   (chance = 12.5%)")
    print(f"  gap           : {acc_trained - acc_random:+6.2f} pts")
    print("=" * 60)
    print("A clear gap above the control = the model explicitly, linearly")
    print("represents the causal neighborhood it was never taught.")


if __name__ == "__main__":
    main()