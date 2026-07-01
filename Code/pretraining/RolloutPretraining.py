"""
RolloutPretraining.py — pretrains GeneralTransformer on the Rollout task (the
"long-range but fixed-period" arm of the pretraining spectrum). Cells are generated
by rolling out a fixed local update rule over many rows; the model must predict each
next cell from a flattened row-major sequence, so long-range dependencies exist but
recur at a fixed period rather than the variable carry-distance seen in the target
task.
"""
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
    """Pretrain on fixed-period rollout sequences and save Model A's weights to ROLLOUT_WEIGHTS."""
    VOCAB_SIZE = 3                       # {0, 1, PAD} — binary cell state + pad, not the addition vocab
    MIN_N, MAX_N = 16, 32                # period range: row width sampled per-sequence in [MIN_N, MAX_N)
    ROWS = 8                             # rollout depth; max_len below assumes worst case MAX_N*ROWS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"Rollout pre-training on {device} ({n_gpu} GPU(s)) | period {MIN_N}-{MAX_N} "
          f"| rows {ROWS} | max_len {MAX_N*ROWS} | batch {ModelConfig.batch_size}")

    model = GeneralTransformer(vocab_size=VOCAB_SIZE, d_model=ModelConfig.d_model, nhead=ModelConfig.n_heads,
                              num_layers=ModelConfig.n_layers, dim_feedforward=ModelConfig.dim_feedforward).to(device)
    if n_gpu > 1:
        print(f"Using {n_gpu} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    # Rule30RolloutDataset handles the variable-period generation and returns a
    # boolean mask so loss/accuracy exclude first-row cells and right-padding.
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

            # Masked-out positions (first row / pad) are relabeled to PAD_IDX so
            # CrossEntropyLoss's ignore_index drops them from the loss too, not
            # just from the accuracy tally below.
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

    # Unwrap DataParallel before saving so downstream loaders (probes, transfer
    # scripts) get a plain state_dict without "module." key prefixes.
    to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(to_save, ROLLOUT_WEIGHTS)
    print(f"\nDone in {(time.time()-start)/60:.1f} min. Saved {ROLLOUT_WEIGHTS}")


if __name__ == "__main__":
    train_rollout()