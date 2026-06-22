"""
Othello-GPT-style world-model probe for Rule 30 (alignment-corrected).

Key fix: the pretraining uses shifted targets (roll(shifts=1)), so the model's
output at position i is conditioned on cells [i-2, i-1, i] -- it NEVER sees the
right neighbor i+1. The textbook neighborhood [i-1, i, i+1] is therefore the
WRONG probe target; the model was structurally prevented from representing it.

We probe for the neighborhood the model ACTUALLY conditions on:
    MODEL_ALIGNED:  [i-2, i-1, i]   (value 0-7)   <- default, matches training
    TEXTBOOK:       [i-1, i, i+1]   (value 0-7)   <- the causal rule, for contrast

Both are latent variables never supervised. Linear probe only; random-init
model = trivially-decodable floor. Per-class diagnostics expose any collapse.
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
PROBE_LAYERS = list(range(NUM_LAYERS))
NEIGHBORHOOD = "model_aligned"   # "model_aligned" = [i-2,i-1,i] ; "textbook" = [i-1,i,i+1]
# --------------


def neighborhood_targets(state_t, mode):
    """3-cell neighborhood value 0..7. Never a training target.
    model_aligned: cells the shifted-target model actually conditions on.
    textbook: the canonical Rule-30 [left,center,right]."""
    if mode == "model_aligned":
        c0 = torch.roll(state_t, shifts=2, dims=1)   # i-2
        c1 = torch.roll(state_t, shifts=1, dims=1)   # i-1
        c2 = state_t                                 # i
    elif mode == "textbook":
        c0 = torch.roll(state_t, shifts=1,  dims=1)  # i-1 (left)
        c1 = state_t                                 # i   (center)
        c2 = torch.roll(state_t, shifts=-1, dims=1)  # i+1 (right)
    else:
        raise ValueError(mode)
    return (c0 * 4 + c1 * 2 + c2 * 1).long()


def capture_layer(transformer, layer_idx):
    store = {}
    def hook(_m, _i, out):
        store["h"] = out.detach().float()
    handle = transformer.transformer.layers[layer_idx].register_forward_hook(hook)
    return store, handle


@torch.no_grad()
def feature_stats(transformer, loader, layer_idx, device, n_batches=8):
    store, handle = capture_layer(transformer, layer_idx)
    s = ssq = count = 0.0
    for i, (state_t, _) in enumerate(loader):
        if i >= n_batches:
            break
        with autocast("cuda"):
            _ = transformer(state_t.to(device))
        f = store["h"][:, 2:, :].reshape(-1, D_MODEL)
        s = s + f.sum(0); ssq = ssq + (f * f).sum(0); count += f.shape[0]
    handle.remove()
    mean = s / count
    std = torch.sqrt((ssq / count - mean ** 2).clamp_min(1e-6))
    return mean, std


def run_probe(transformer, device, layer_idx, tag):
    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad = False

    loader = DataLoader(Rule30Dataset(num_samples=10000, seq_length=SEQ_LENGTH),
                        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    mean, std = feature_stats(transformer, loader, layer_idx, device)

    store, handle = capture_layer(transformer, layer_idx)
    probe = nn.Linear(D_MODEL, 8).to(device)
    opt = AdamW(probe.parameters(), lr=LR, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()

    confusion = None
    for epoch in range(PROBE_EPOCHS):
        probe.train()
        correct = total = 0
        confusion = torch.zeros(8, 8, dtype=torch.long)   # rows=true, cols=pred
        for state_t, _ in loader:
            state_t = state_t.to(device, non_blocking=True)
            y = neighborhood_targets(state_t, NEIGHBORHOOD)

            with torch.no_grad(), autocast("cuda"):
                _ = transformer(state_t)
            f = (store["h"][:, 2:, :] - mean) / std
            yv = y[:, 2:]

            opt.zero_grad()
            logits = probe(f)
            loss = crit(logits.reshape(-1, 8), yv.reshape(-1))
            loss.backward()
            opt.step()

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == yv).sum().item()
            total += yv.numel()
            if epoch == PROBE_EPOCHS - 1:
                p_flat = preds.reshape(-1).cpu()
                y_flat = yv.reshape(-1).cpu()
                confusion += torch.bincount(y_flat * 8 + p_flat, minlength=64).reshape(8, 8)
    acc = 100.0 * correct / total
    handle.remove()

    n_pred_classes = int((confusion.sum(0) > 0).sum())
    per_class = []
    for c in range(8):
        denom = confusion[c].sum().item()
        per_class.append(100.0 * confusion[c, c].item() / denom if denom else 0.0)
    flag = "  <-- COLLAPSE (few classes predicted)" if n_pred_classes <= 3 else ""
    print(f"  {tag:<8} L{layer_idx} | acc {acc:6.2f}% | classes predicted: {n_pred_classes}/8{flag}")
    print(f"           per-class acc: " + " ".join(f"{v:4.0f}" for v in per_class))
    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rule-30 probe | target = {NEIGHBORHOOD} neighborhood (0-7) | chance 12.5%\n")

    trained = Rule30Transformer(vocab_size=2, d_model=D_MODEL, nhead=NHEAD,
                                num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)
    sd = torch.load(CHECKPOINT, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    trained.load_state_dict(sd)

    rand = Rule30Transformer(vocab_size=2, d_model=D_MODEL, nhead=NHEAD,
                             num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)

    results = []
    for L in PROBE_LAYERS:
        at = run_probe(trained, device, L, "TRAINED")
        ar = run_probe(rand,    device, L, "RANDOM")
        results.append((L, at, ar))
        print(f"  -> layer {L} gap: {at - ar:+6.2f} pts\n")

    print("=" * 56)
    print(f"{'layer':<8}{'trained':>10}{'random':>10}{'gap':>10}")
    for L, at, ar in results:
        print(f"{L:<8}{at:>10.2f}{ar:>10.2f}{at-ar:>+10.2f}")
    best = max(results, key=lambda r: r[1] - r[2])
    print("=" * 56)
    print(f"Largest gap at layer {best[0]}: {best[1]:.2f}% vs {best[2]:.2f}% "
          f"({best[1]-best[2]:+.2f} pts)  [target={NEIGHBORHOOD}]")


if __name__ == "__main__":
    main()