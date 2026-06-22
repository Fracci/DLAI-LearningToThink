"""
Othello-GPT-style world-model probe for Rule 30 (fixed).

Probes an INTERMEDIATE layer for a LATENT variable the model was never
supervised on: the 3-cell neighborhood [left,center,right] (value 0-7) that
*causes* each Rule-30 transition. Linear probe only. Random-init control =
the trivially-decodable floor; the trained model must beat it.

Fixes vs the previous version:
  - probe trains in fp32 (no autocast) -> avoids the fp16 collapse to ~50%
  - features standardized per-dim before the linear probe
  - lower LR + more epochs so the probe actually converges
  - sweeps every layer in one run, reports trained-vs-random gap per layer
  - flags the degenerate "locked on the output bit" 50% collapse if it recurs
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
PROBE_LAYERS = list(range(NUM_LAYERS))   # sweep 0..5
# --------------


def neighborhood_targets(state_t):
    """Per-cell neighborhood 0..7 = 4*left + 2*center + 1*right (periodic).
    Never a training target."""
    left  = torch.roll(state_t, shifts=1,  dims=1)
    right = torch.roll(state_t, shifts=-1, dims=1)
    return (left * 4 + state_t * 2 + right * 1).long()


def capture_layer(transformer, layer_idx):
    store = {}
    def hook(_m, _i, out):
        store["h"] = out.detach().float()    # force fp32 features
    handle = transformer.transformer.layers[layer_idx].register_forward_hook(hook)
    return store, handle


@torch.no_grad()
def feature_stats(transformer, loader, layer_idx, device, n_batches=8):
    """Mean/std per feature dim for standardization (computed once)."""
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
    var = ssq / count - mean ** 2
    std = torch.sqrt(var.clamp_min(1e-6))
    return mean, std


def run_probe(transformer, device, layer_idx, tag):
    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad = False

    loader = DataLoader(Rule30Dataset(num_samples=10000, seq_length=SEQ_LENGTH),
                        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    mean, std = feature_stats(transformer, loader, layer_idx, device)

    store, handle = capture_layer(transformer, layer_idx)
    probe = nn.Linear(D_MODEL, 8).to(device)             # linear, 8-way
    opt = AdamW(probe.parameters(), lr=LR, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()

    acc = 0.0
    for epoch in range(PROBE_EPOCHS):
        probe.train()
        correct = total = 0
        for state_t, _ in loader:
            state_t = state_t.to(device, non_blocking=True)
            nbr = neighborhood_targets(state_t)

            with torch.no_grad(), autocast("cuda"):
                _ = transformer(state_t)
            f = (store["h"][:, 2:, :] - mean) / std       # standardized, fp32
            y = nbr[:, 2:]

            opt.zero_grad()
            logits = probe(f)                             # fp32 probe (no autocast)
            loss = crit(logits.reshape(-1, 8), y.reshape(-1))
            loss.backward()
            opt.step()

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()
        acc = 100.0 * correct / total

    handle.remove()
    # collapse check: ~50% with only 2 predicted classes = locked on output bit
    note = ""
    if 48.0 <= acc <= 52.0:
        note = "  (warning: near 50% -- check for output-bit collapse)"
    print(f"  {tag:<16} layer {layer_idx} | neighborhood acc {acc:6.2f}%{note}")
    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rule-30 world-model probe on: {device}  (chance = 12.5%)\n")

    trained = Rule30Transformer(vocab_size=2, d_model=D_MODEL, nhead=NHEAD,
                                num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)
    sd = torch.load(CHECKPOINT, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    trained.load_state_dict(sd)

    rand = Rule30Transformer(vocab_size=2, d_model=D_MODEL, nhead=NHEAD,
                             num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)

    print("Per-layer linear-probe accuracy (neighborhood 0-7):")
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
    print(f"Largest gap at layer {best[0]}: trained {best[1]:.2f}% vs random {best[2]:.2f}% "
          f"({best[1]-best[2]:+.2f} pts)")
    print("A clear positive gap = the model explicitly, linearly represents the")
    print("causal neighborhood it was never taught.")


if __name__ == "__main__":
    main()