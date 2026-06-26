import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import time

from Transformer import GeneralTransformer
from RolloutGenerator import Rule30RolloutDataset, PAD_IDX


def train_rollout():
    VOCAB_SIZE = 3                       
    D_MODEL, NHEAD, NUM_LAYERS, DIM_FF = 256, 8, 6, 1024

    MIN_N, MAX_N = 16, 32                
    ROWS = 8                             

    BATCH_SIZE = 256                     
    EPOCHS = 300
    NUM_SAMPLES = 20000
    LR, WEIGHT_DECAY, GRAD_CLIP = 1e-3, 0.2, 1.0
    PRINT_EVERY = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"Rollout pre-training on {device} ({n_gpu} GPU(s)) | period {MIN_N}-{MAX_N} "
          f"| rows {ROWS} | max_len {MAX_N*ROWS} | batch {BATCH_SIZE}")

    model = GeneralTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD,
                              num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)
    if n_gpu > 1:
        print(f"Using {n_gpu} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    dataset = Rule30RolloutDataset(NUM_SAMPLES, MIN_N, MAX_N, ROWS, pad_idx=PAD_IDX)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        pin_memory=True, num_workers=2)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler("cuda")

    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = total = 0
        for x, y, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y_loss = torch.where(mask, y, torch.full_like(y, PAD_IDX))

            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(x)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), y_loss.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += ((preds == y) & mask).sum().item()
            total += mask.sum().item()

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
            elapsed = (time.time() - start) / 60
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Loss {total_loss/len(loader):.4f} "
                  f"| Rollout next-token acc {100.0*correct/total:6.2f}% | {elapsed:5.1f} min")

    to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(to_save, "rule30_rollout_pretrained.pt")
    print(f"\nDone in {(time.time()-start)/60:.1f} min. Saved rule30_rollout_pretrained.pt")


if __name__ == "__main__":
    train_rollout()