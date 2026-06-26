import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import time
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.Transformer import GeneralTransformer
from config import ModelConfig, EVAL_EVERY, ROLLOUT_WEIGHTS
from data_generation.RolloutGenerator import Rule30RolloutDataset, PAD_IDX


def train_rollout():
    VOCAB_SIZE = 3                       
    MIN_N, MAX_N = 16, 32                
    ROWS = 8                             

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"Rollout pre-training on {device} ({n_gpu} GPU(s)) | period {MIN_N}-{MAX_N} "
          f"| rows {ROWS} | max_len {MAX_N*ROWS} | batch {ModelConfig.batch_size}")

    model = GeneralTransformer(vocab_size=VOCAB_SIZE, d_model=ModelConfig.d_model, nhead=ModelConfig.n_heads,
                              num_layers=ModelConfig.n_layers, dim_feedforward=ModelConfig.dim_feedforward).to(device)
    if n_gpu > 1:
        print(f"Using {n_gpu} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    dataset = Rule30RolloutDataset(ModelConfig.num_samples, MIN_N, MAX_N, ROWS, pad_idx=PAD_IDX)
    loader = DataLoader(dataset, batch_size=ModelConfig.batch_size, shuffle=True,
                        pin_memory=True, num_workers=2)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = AdamW(model.parameters(), lr=ModelConfig.lr, weight_decay=ModelConfig.weight_decay)
    scaler = GradScaler("cuda")

    start = time.time()
    for epoch in range(ModelConfig.epochs):
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
            nn.utils.clip_grad_norm_(model.parameters(), ModelConfig.grad_clip)
            scaler.step(optimizer); scaler.update()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += ((preds == y) & mask).sum().item()
            total += mask.sum().item()

        if epoch % EVAL_EVERY == 0 or epoch == ModelConfig.epochs - 1:
            elapsed = (time.time() - start) / 60
            print(f"Epoch [{epoch+1:3d}/{ModelConfig.epochs}] | Loss {total_loss/len(loader):.4f} "
                  f"| Rollout next-token acc {100.0*correct/total:6.2f}% | {elapsed:5.1f} min")

    to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(to_save, ROLLOUT_WEIGHTS)
    print(f"\nDone in {(time.time()-start)/60:.1f} min. Saved {ROLLOUT_WEIGHTS}")


if __name__ == "__main__":
    train_rollout()