"""
Othello-GPT-style world-model probe for Rule 30.

Why this differs from a naive probe:
  - The naive probe reads `final_norm` (one linear layer before the output)
    and predicts state_{t+1} -- the SAME thing fc_out is trained to produce.
    It is guaranteed to hit ~100% and proves nothing.
  - Here we probe an INTERMEDIATE layer for a LATENT variable the model was
    NEVER supervised on: the 3-cell neighborhood [left,center,right] (value 0-7)
    that *causes* each Rule-30 transition. If a LINEAR probe recovers it, the
    model explicitly represents the causal variables -> a genuine world model.

Two controls make the number meaningful:
  - linear probe only (a hidden layer could recompute the neighborhood itself)
  - a RANDOM-init transformer probed the same way = the "trivially decodable"
    floor. The trained model must beat this floor to claim anything.
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
PROBE_EPOCHS = 12
PROBE_LAYER = 3          # intermediate layer (0..NUM_LAYERS-1). Try 2,3,4.
LR = 1e-3
# --------------


def neighborhood_targets(state_t):
    """Per-cell 3-bit neighborhood value 0..7 = 4*left + 2*center + 1*right,
    periodic boundaries -- matches Rule30Generator. NEVER a training target."""
    left  = torch.roll(state_t, shifts=1,  dims=1)
    right = torch.roll(state_t, shifts=-1, dims=1)
    return (left * 4 + state_t * 2 + right * 1).long()   # (B, L) in [0,7]


def capture_layer(transformer, layer_idx):
    """Forward hook on the output of transformer.layers[layer_idx]."""
    store = {}
    def hook(_m, _i, out):
        store["h"] = out.detach()
    handle = transformer.transformer.layers[layer_idx].register_forward_hook(hook)
    return store, handle


def run_probe(transformer, device, tag):
    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad = False

    store, handle = capture_layer(transformer, PROBE_LAYER)

    # LINEAR probe, 8-way (no hidden layer, no nonlinearity)
    probe = nn.Linear(D_MODEL, 8).to(device)
    opt = AdamW(probe.parameters(), lr=LR, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()

    loader = DataLoader(Rule30Dataset(num_samples=10000, seq_length=SEQ_LENGTH),
                        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    print(f"\n=== probing {tag} | layer {PROBE_LAYER} | target = neighborhood (0-7) ===")
    for epoch in range(PROBE_EPOCHS):
        probe.train()
        correct = total = 0
        loss_sum = 0.0
        for state_t, _ in loader:
            state_t = state_t.to(device, non_blocking=True)
            nbr = neighborhood_targets(state_t)            # latent label

            with torch.no_grad(), autocast("cuda"):
                _ = transformer(state_t)
            feats = store["h"]                             # (B, L, d_model)

            # drop the 2 causally-blind edge tokens (as in training)
            f = feats[:, 2:, :]
            y = nbr[:, 2:]

            opt.zero_grad()
            with autocast("cuda"):
                logits = probe(f)
                loss = crit(logits.reshape(-1, 8), y.reshape(-1))
            loss.backward()
            opt.step()

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()
            loss_sum += loss.item()

        acc = 100.0 * correct / total
        print(f"  epoch {epoch+1:2d}/{PROBE_EPOCHS} | loss {loss_sum/len(loader):.4f} | neighborhood acc {acc:6.2f}%")

    handle.remove()
    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rule-30 world-model probe on: {device}")

    # trained model
    trained = Rule30Transformer(vocab_size=2, d_model=D_MODEL, nhead=NHEAD,
                                num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)
    sd = torch.load(CHECKPOINT, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    trained.load_state_dict(sd)
    acc_trained = run_probe(trained, device, "TRAINED model")

    # random-init control (the trivially-decodable floor)
    rand = Rule30Transformer(vocab_size=2, d_model=D_MODEL, nhead=NHEAD,
                             num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)
    acc_random = run_probe(rand, device, "RANDOM-init control")

    print("\n" + "=" * 60)
    print(f"Neighborhood linear-probe accuracy @ layer {PROBE_LAYER}")
    print(f"  trained model : {acc_trained:6.2f}%")
    print(f"  random control: {acc_random:6.2f}%   (chance = 12.5%)")
    print(f"  gap           : {acc_trained - acc_random:+6.2f} pts")
    print("=" * 60)
    print("A large gap above the random control = the model explicitly,")
    print("linearly represents the causal neighborhood it was never taught.")
    print("If trained ~= random, the model only computes the output, not a world model.")


if __name__ == "__main__":
    main()