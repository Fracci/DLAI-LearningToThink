"""
World-model probe for the MULTI-STEP ROLLOUT model.

Single-step Rule 30 conditions on a LOCAL neighborhood, so its probe targeted
nearby cells. The rollout model is different: flattened as [row0 | row1 | ...],
to emit cell i of row r it must attend back ~N tokens to row r-1's cells
[i-1, i, i+1]. The latent world variable is therefore NON-LOCAL -- the
previous-row neighborhood located one period (N) back.

This probe reads an intermediate layer and asks whether that long-range
neighborhood is LINEARLY decodable. A clear gap over a random-init control means
the rollout model built an explicit, long-range world model -- the exact
capability the locality thesis predicts single-step Rule 30 lacked.

TARGET:
  "prev_row_neighborhood" -> 8-way (value 0-7 of the row-above [i-1,i,i+1])  [default]
  "cell_above"            -> 2-way (just the cell directly above, offset N back)

Alignment is predictive: the hidden state at position p (which predicts token
p+1) is probed for the cause of token p+1. Targets are computed from the grid,
so periodic-boundary wrap at row edges is exact.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast

from Transformer import Rule30Transformer

# --- config ---
CHECKPOINT  = "rule30_rollout_pretrained.pt"
VOCAB_SIZE  = 3                  # rollout model: {0, 1, PAD}
D_MODEL, NHEAD, NUM_LAYERS, DIM_FF = 256, 8, 6, 1024
N           = 24                 # fixed period for the probe (training used 16-32)
ROWS        = 8                  # -> sequence length N*ROWS = 192
BATCH_SIZE  = 128
ITERS_PER_EPOCH = 80
PROBE_EPOCHS = 25
LR = 3e-4
PROBE_LAYER = 3                  # intermediate layer (0..5). Try 2,3,4.
TARGET = "prev_row_neighborhood" # or "cell_above"
RULE_LUT = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0], dtype=torch.long)
# --------------


def make_batch(bs, device):
    """Build flattened rollout sequences + the long-range (row-above) target
    for every position whose token lives in row >= 1."""
    row = torch.randint(0, 2, (bs, N), dtype=torch.long)
    grid = [row]
    for _ in range(ROWS - 1):
        left = torch.roll(row, 1, 1); right = torch.roll(row, -1, 1)
        row = RULE_LUT[left * 4 + row * 2 + right]
        grid.append(row)
    grid = torch.stack(grid, dim=1)                       # [bs, ROWS, N]
    seq = grid.reshape(bs, ROWS * N)                      # row-major flatten

    g_prev = grid[:, :-1, :]                              # rows 0..T-2 = "row above" for rows 1..T-1
    left = torch.roll(g_prev, 1, 2); right = torch.roll(g_prev, -1, 2)
    if TARGET == "cell_above":
        tgt_rows = g_prev                                # binary: the cell directly above
        n_classes = 2
    else:
        tgt_rows = left * 4 + g_prev * 2 + right * 1      # 0..7 neighborhood
        n_classes = 8

    full_tgt = torch.full((bs, ROWS * N), -1, dtype=torch.long)   # -1 = ignore (row 0)
    full_tgt[:, N:] = tgt_rows.reshape(bs, (ROWS - 1) * N)        # token at pos q (row>=1) <- its cause
    return seq.to(device), full_tgt.to(device), n_classes


def capture_layer(model, layer_idx):
    store = {}
    def hook(_m, _i, out):
        store["h"] = out.detach().float()
    handle = model.transformer.layers[layer_idx].register_forward_hook(hook)
    return store, handle


@torch.no_grad()
def feature_stats(model, store, device, n_batches=6):
    """Per-dim mean/std over valid (row>=1) features, for standardization."""
    s = ssq = count = 0.0
    for _ in range(n_batches):
        seq, full_tgt, _ = make_batch(BATCH_SIZE, device)
        x = seq[:, :-1]
        with autocast("cuda"):
            _ = model(x)
        h = store["h"]                                   # [bs, L-1, d]
        valid = (full_tgt[:, 1:] != -1).reshape(-1)
        f = h.reshape(-1, D_MODEL)[valid]
        s = s + f.sum(0); ssq = ssq + (f * f).sum(0); count += f.shape[0]
    mean = s / count
    std = torch.sqrt((ssq / count - mean ** 2).clamp_min(1e-6))
    return mean, std


def run_probe(model, device, tag):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    store, handle = capture_layer(model, PROBE_LAYER)

    # peek n_classes
    _, _, n_classes = make_batch(2, device)
    mean, std = feature_stats(model, store, device)

    probe = nn.Linear(D_MODEL, n_classes).to(device)     # linear probe
    opt = AdamW(probe.parameters(), lr=LR, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()

    chance = 100.0 / n_classes
    print(f"\n=== probing {tag} | layer {PROBE_LAYER} | target = {TARGET} "
          f"({n_classes}-way, chance {chance:.1f}%) | offset N={N} ===")
    final_acc = 0.0
    for epoch in range(PROBE_EPOCHS):
        probe.train()
        total_loss = 0.0
        correct = total = 0
        for _ in range(ITERS_PER_EPOCH):
            seq, full_tgt, _ = make_batch(BATCH_SIZE, device)
            x = seq[:, :-1]
            with torch.no_grad(), autocast("cuda"):
                _ = model(x)
            h = store["h"]                               # [bs, L-1, d]
            targs = full_tgt[:, 1:]                      # cause of token p+1
            valid = (targs != -1).reshape(-1)
            f = (h.reshape(-1, D_MODEL)[valid] - mean) / std
            y = targs.reshape(-1)[valid]

            opt.zero_grad()
            logits = probe(f)                            # fp32 probe
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += y.numel()

        final_acc = 100.0 * correct / total
        n_pred = len(torch.unique(preds))
        print(f"  Probe Epoch [{epoch+1:2d}/{PROBE_EPOCHS}] | Loss: {total_loss/ITERS_PER_EPOCH:.4f} "
              f"| Probe Accuracy: {final_acc:6.2f}% | classes predicted: {n_pred}/{n_classes}")

    handle.remove()
    return final_acc, chance


def load_rollout(device):
    m = Rule30Transformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD,
                          num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)
    sd = torch.load(CHECKPOINT, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rollout world-model probe on {device}")

    try:
        model = load_rollout(device)
        print("Loaded rollout-pretrained weights.")
    except Exception as e:
        print(f"Error loading {CHECKPOINT}: {e}")
        return
    acc_trained, chance = run_probe(model, device, "ROLLOUT model")

    rand = Rule30Transformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD,
                             num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)
    acc_random, _ = run_probe(rand, device, "RANDOM-init control")

    print("\n" + "=" * 64)
    print(f"Long-range (row-above, offset N={N}) linear-probe @ layer {PROBE_LAYER} "
          f"| target={TARGET}")
    print(f"  rollout model : {acc_trained:6.2f}%")
    print(f"  random control: {acc_random:6.2f}%   (chance = {chance:.1f}%)")
    print(f"  gap           : {acc_trained - acc_random:+6.2f} pts")
    print("=" * 64)
    print("A clear gap = the rollout model linearly represents the row ONE PERIOD")
    print("back -- an explicit LONG-RANGE world model (vs the local one of single-step).")


if __name__ == "__main__":
    main()