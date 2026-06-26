import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast

from Transformer import GeneralTransformer
from Rule30Generator import Rule30Dataset

# CONFIG
CHECKPOINT  = "rule30_pretrained_new.pt"
D_MODEL, NHEAD, NUM_LAYERS, DIM_FF = 256, 8, 6, 1024
SEQ_LENGTH  = 256
BATCH_SIZE  = 128
PROBE_EPOCHS = 25
LR = 3e-4
PROBE_LAYER = 3                 # intermediate layer


def neighborhood_targets(state_t):
    c0 = torch.roll(state_t, shifts=2, dims=1)
    c1 = torch.roll(state_t, shifts=1, dims=1)
    c2 = state_t                              

    return (c0 * 4 + c1 * 2 + c2 * 1).long()


def capture_layer(transformer, layer_idx):
    store = {}
    def hook(_m, _i, out):
        store["h"] = out.detach().float()
    handle = transformer.transformer.layers[layer_idx].register_forward_hook(hook)
    return store, handle


def run_probe(transformer, device, tag):
    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad = False

    store, handle = capture_layer(transformer, PROBE_LAYER)

    probe = nn.Linear(D_MODEL, 8).to(device)
    opt = AdamW(probe.parameters(), lr=LR, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()

    loader = DataLoader(Rule30Dataset(num_samples=10000, seq_length=SEQ_LENGTH),
                        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    print(f"\nprobing {tag} | layer {PROBE_LAYER}")
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
            f = store["h"][:, 2:, :]                  
            y = nbr[:, 2:]

            opt.zero_grad()
            logits = probe(f)                     
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

    transformer = GeneralTransformer(vocab_size=2, d_model=D_MODEL, nhead=NHEAD,
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
    rand = GeneralTransformer(vocab_size=2, d_model=D_MODEL, nhead=NHEAD,
                             num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)
    acc_random = run_probe(rand, device, "RANDOM-init control")

    print(f"Neighborhood linear-probe accuracy @ layer {PROBE_LAYER}")
    print(f"  trained model : {acc_trained:6.2f}%")
    print(f"  random control: {acc_random:6.2f}%   (chance = 12.5%)")
    print(f"  gap           : {acc_trained - acc_random:+6.2f} pts")


if __name__ == "__main__":
    main()