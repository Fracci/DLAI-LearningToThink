import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast

from Transformer import GeneralTransformer
from CarryOnlyGenerator import sample_ab, assemble, VOCAB, IGNORE

# CONFIG
CHECKPOINT  = "carryonly_pretrained.pt"
D_MODEL, NHEAD, NUM_LAYERS, DIM_FF = 256, 8, 6, 1024
MIN_N, MAX_N = 8, 24
CHAIN_MAX = 12             
TARGET_ACTIVE = 0.25
MAX_LEN = 3 * MAX_N + 2
BATCH_SIZE  = 128
ITERS_PER_EPOCH = 80
PROBE_EPOCHS = 25
LR = 3e-4
PROBE_LAYER = 3
TARGET = "gen_dist"  # or "carry_in"            


def make_batch(bs, device):
    seqs, targs = [], []
    for _ in range(bs):
        import random
        n = random.randint(MIN_N, MAX_N)
        a, b = sample_ab(n, CHAIN_MAX, TARGET_ACTIVE)
        seq, tgt = assemble(a, b, MAX_LEN, latent=TARGET)
        seqs.append(seq); targs.append(tgt)
    return torch.stack(seqs).to(device), torch.stack(targs).to(device)


def capture_layer(model, layer_idx):
    store = {}
    def hook(_m, _i, out):
        store["h"] = out.detach().float()
    handle = model.transformer.layers[layer_idx].register_forward_hook(hook)
    return store, handle


def n_classes_for(target):
    return 2 if target == "carry_in" else 6


@torch.no_grad()
def feature_stats(model, store, device, n_batches=6):
    s = ssq = count = 0.0
    for _ in range(n_batches):
        seq, tgt = make_batch(BATCH_SIZE, device)
        with autocast("cuda"):
            _ = model(seq)
        valid = (tgt != IGNORE).reshape(-1)
        f = store["h"].reshape(-1, D_MODEL)[valid]
        s = s + f.sum(0); ssq = ssq + (f * f).sum(0); count += f.shape[0]
    mean = s / count
    std = torch.sqrt((ssq / count - mean ** 2).clamp_min(1e-6))
    return mean, std


def run_probe(model, device, tag):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    store, handle = capture_layer(model, PROBE_LAYER)
    nc = n_classes_for(TARGET)
    mean, std = feature_stats(model, store, device)

    probe = nn.Linear(D_MODEL, nc).to(device)
    opt = AdamW(probe.parameters(), lr=LR, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    chance = 100.0 / nc

    print(f"\n=== probing {tag} | layer {PROBE_LAYER} | target = {TARGET} "
          f"({nc}-way, chance {chance:.1f}%) ===")
    final_acc = 0.0
    for epoch in range(PROBE_EPOCHS):
        probe.train()
        total_loss = 0.0
        correct = total = 0
        for _ in range(ITERS_PER_EPOCH):
            seq, tgt = make_batch(BATCH_SIZE, device)
            with torch.no_grad(), autocast("cuda"):
                _ = model(seq)
            valid = (tgt != IGNORE).reshape(-1)
            f = (store["h"].reshape(-1, D_MODEL)[valid] - mean) / std
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
        print(f"  Probe Epoch [{epoch+1:2d}/{PROBE_EPOCHS}] | Loss: {total_loss/ITERS_PER_EPOCH:.4f} "
              f"| Probe Accuracy: {final_acc:6.2f}% | classes predicted: {n_pred}/{nc}")

    handle.remove()
    return final_acc, chance


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Carry-only world-model probe on {device}")

    try:
        model = GeneralTransformer(vocab_size=VOCAB, d_model=D_MODEL, nhead=NHEAD,
                                  num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)
        sd = torch.load(CHECKPOINT, map_location=device)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        print("Loaded carry-only pretrained weights.")
    except Exception as e:
        print(f"Error loading {CHECKPOINT}: {e}")
        return
    acc_trained, chance = run_probe(model, device, "CARRY model")

    rand = GeneralTransformer(vocab_size=VOCAB, d_model=D_MODEL, nhead=NHEAD,
                             num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)
    acc_random, _ = run_probe(rand, device, "RANDOM-init control")

    gap = acc_trained - acc_random
    print(f"Long-range carry latent ({TARGET}) linear-probe @ layer {PROBE_LAYER}")
    print(f"  carry model   : {acc_trained:6.2f}%")
    print(f"  random control: {acc_random:6.2f}%   <- empirical floor (NOT chance {chance:.1f}%)")
    print(f"  GAP           : {gap:+6.2f} pts   <- this is the result")


if __name__ == "__main__":
    main()