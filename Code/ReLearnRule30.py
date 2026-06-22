"""
TEST 2 — Relearning-speed probe (memory under catastrophic forgetting).

A network that retained useful structure relearns its OLD task faster than a
fresh one. We take the transformer BODY of each fine-tuned model, attach a
fresh binary head, and re-train on Rule 30 for a short budget:

  A_body   = transformer body of modelA (pretrained -> arithmetic)
  B_body   = transformer body of modelB (random     -> arithmetic)
  scratch  = fully random body (the relearning floor)

Fair-comparison controls:
  - all three get the SAME freshly-seeded embedding/fc_out (only the body differs)
  - all three train on the SAME fixed Rule 30 data, same optimizer/LR
  - evaluated each epoch on the SAME fixed Rule 30 eval set

If A_body climbs Rule-30 accuracy faster than B_body, pretrained structure
survived the arithmetic phase. Needs GPU.
"""
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Transformer import Rule30Transformer
from Rule30Generator import Rule30Dataset

# --- config ---
A_AFTER = "modelA_phase5_final.pt"
B_AFTER = "modelB_phase5_final.pt"

D_MODEL, NHEAD, NUM_LAYERS, DIM_FF = 256, 8, 6, 1024
SEQ_LEN     = 128
BATCH       = 128
EPOCHS      = 40          # short budget — we care about SPEED, not final accuracy
N_TRAIN     = 5000
N_EVAL      = 2000
LR          = 5e-4
INIT_SEED   = 1234        # identical fresh head (and scratch body) across runs
DATA_SEED   = 7           # identical Rule30 data across runs
# --------------


def make_rule30(n, seq_len, seed):
    torch.manual_seed(seed)
    ds = Rule30Dataset(num_samples=n, seq_length=seq_len)
    xs, ys = [], []
    for i in range(n):
        x, y = ds[i]
        xs.append(x); ys.append(y)
    return TensorDataset(torch.stack(xs), torch.stack(ys))


def build(checkpoint, device):
    """Fresh seeded model; if checkpoint given, overwrite the body only."""
    torch.manual_seed(INIT_SEED)
    m = Rule30Transformer(vocab_size=2, d_model=D_MODEL, nhead=NHEAD,
                          num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)
    if checkpoint is not None:
        sd = torch.load(checkpoint, map_location=device)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        body = {k: v for k, v in sd.items()
                if k.startswith("transformer.") or k.startswith("final_norm.")}
        m.load_state_dict(body, strict=False)   # keep seeded embedding/fc_out
    return m


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for st, st1 in loader:
        st = st.to(device); st1 = st1.to(device)
        with autocast("cuda"):
            logits = model(st)
        shifted = torch.roll(st1, shifts=1, dims=1)
        preds = torch.argmax(logits[:, 2:, :], dim=-1)   # ignore 2 causally-blind tokens
        tgt = shifted[:, 2:]
        correct += (preds == tgt).sum().item()
        total += tgt.numel()
    return 100.0 * correct / total


def relearn(label, checkpoint, train_loader, eval_loader, device):
    model = build(checkpoint, device)
    opt = AdamW(model.parameters(), lr=LR)
    scaler = GradScaler("cuda")
    crit = nn.CrossEntropyLoss()
    curve = []
    print(f"\n--- relearning: {label} ---")
    for epoch in range(EPOCHS):
        model.train()
        for st, st1 in train_loader:
            st = st.to(device); st1 = st1.to(device)
            opt.zero_grad()
            with autocast("cuda"):
                logits = model(st)
                shifted = torch.roll(st1, shifts=1, dims=1)
                loss = crit(logits[:, 2:, :].reshape(-1, 2), shifted[:, 2:].reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        acc = accuracy(model, eval_loader, device)
        curve.append(acc)
        if epoch % 2 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch+1:2d}/{EPOCHS} | Rule30 acc {acc:6.2f}%")
    return curve


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Relearning-speed probe on: {device}")

    train_loader = DataLoader(make_rule30(N_TRAIN, SEQ_LEN, DATA_SEED),
                              batch_size=BATCH, shuffle=False)
    eval_loader = DataLoader(make_rule30(N_EVAL, SEQ_LEN, DATA_SEED + 1),
                             batch_size=BATCH)

    runs = {
        "A_body (pretrained->arith)": A_AFTER,
        "B_body (random->arith)":     B_AFTER,
        "scratch (random body)":      None,
    }
    curves = {label: relearn(label, ckpt, train_loader, eval_loader, device)
              for label, ckpt in runs.items()}

    # CSV
    with open("relearn_rule30.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + list(curves.keys()))
        for e in range(EPOCHS):
            w.writerow([e + 1] + [f"{curves[k][e]:.2f}" for k in curves])
    print("\nsaved -> relearn_rule30.csv")

    # plot
    colors = {"A_body (pretrained->arith)": "#534AB7",
              "B_body (random->arith)": "#1D9E75",
              "scratch (random body)": "#999999"}
    plt.figure(figsize=(8, 5))
    for label, c in curves.items():
        plt.plot(range(1, EPOCHS + 1), c, "o-", ms=3,
                 color=colors.get(label), label=label)
    plt.xlabel("relearning epoch")
    plt.ylabel("Rule 30 next-state accuracy %")
    plt.title("Relearning speed: does A's body remember Rule 30?")
    plt.ylim(45, 101)
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig("relearn_rule30.png", dpi=130)
    print("saved -> relearn_rule30.png")


if __name__ == "__main__":
    main()