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

# CONFIG
CHECKPOINT  = "rule30_rollout_pretrained.pt"
VOCAB_SIZE  = 3                 
D_MODEL, NHEAD, NUM_LAYERS, DIM_FF = 256, 8, 6, 1024
N           = 24                 
ROWS        = 8                  
BATCH_SIZE  = 128
ITERS_PER_EPOCH = 80
PROBE_EPOCHS = 25
LR = 3e-4
PROBE_LAYER = 5             
TARGET = "cell_above" 
RULE_LUT = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0], dtype=torch.long)


def make_batch(bs, device):
    row = torch.randint(0, 2, (bs, N), dtype=torch.long)
    grid = [row]
    for _ in range(ROWS - 1):
        left = torch.roll(row, 1, 1); right = torch.roll(row, -1, 1)
        row = RULE_LUT[left * 4 + row * 2 + right]
        grid.append(row)
    grid = torch.stack(grid, dim=1)                       
    seq = grid.reshape(bs, ROWS * N)                     

    g_prev = grid[:, :-1, :]                              
    left = torch.roll(g_prev, 1, 2); right = torch.roll(g_prev, -1, 2)
    if TARGET == "cell_above":
        tgt_rows = g_prev                                
        n_classes = 2
    else:
        tgt_rows = left * 4 + g_prev * 2 + right * 1      
        n_classes = 8

    full_tgt = torch.full((bs, ROWS * N), -1, dtype=torch.long)  
    full_tgt[:, N:] = tgt_rows.reshape(bs, (ROWS - 1) * N)        
    return seq.to(device), full_tgt.to(device), n_classes


def capture_layer(model, layer_idx):
    store = {}
    def hook(_m, _i, out):
        store["h"] = out.detach().float()
    handle = model.transformer.layers[layer_idx].register_forward_hook(hook)
    return store, handle


@torch.no_grad()
def feature_stats(model, store, device, n_batches=6):
    s = ssq = count = 0.0
    for _ in range(n_batches):
        seq, full_tgt, _ = make_batch(BATCH_SIZE, device)
        x = seq[:, :-1]
        with autocast("cuda"):
            _ = model(x)
        h = store["h"]                                   
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

    _, _, n_classes = make_batch(2, device)
    mean, std = feature_stats(model, store, device)

    probe = nn.Linear(D_MODEL, n_classes).to(device)     
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
            h = store["h"]                               
            targs = full_tgt[:, 1:]                      
            valid = (targs != -1).reshape(-1)
            f = (h.reshape(-1, D_MODEL)[valid] - mean) / std
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
        print(f"  Probe Epoch [{epoch+1:2d}/{PROBE_EPOCHS}] | Loss: {total_loss/ITERS_PER_EPOCH:.4f} "
              f"| Probe Accuracy: {final_acc:6.2f}% | classes predicted: {n_pred}/{n_classes}")

    handle.remove()
    return final_acc, chance


def load_rollout(device):
    m = GeneralTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD,
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

    rand = GeneralTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD,
                             num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)
    acc_random, _ = run_probe(rand, device, "RANDOM-init control")

    print(f"Long-range (row-above, offset N={N}) linear-probe @ layer {PROBE_LAYER} "
          f"| target={TARGET}")
    print(f"  rollout model : {acc_trained:6.2f}%")
    print(f"  random control: {acc_random:6.2f}%   (chance = {chance:.1f}%)")
    print(f"  gap           : {acc_trained - acc_random:+6.2f} pts")


if __name__ == "__main__":
    main()