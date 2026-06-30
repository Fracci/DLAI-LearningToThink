"""
Rule30PreTraining.py — pretrains the model on the single-step Rule 30 task.

Trains GeneralTransformer to predict the Rule 30 successor of each cell from a
random binary row, the LOCAL/compatible arm of the pretraining spectrum. The
resulting body weights are the initialization for arithmetic fine-tuning.

NOTE on target alignment (intentional, and consistent with Rule30Probe):
The dataset returns (state_t, state_t_plus_1) where state_t_plus_1[i] is the
successor of the neighborhood CENTERED at i. We roll the target by +1 so that
position i instead predicts the successor of the LEFT-anchored neighborhood
[i-2, i-1, i] (a valid relabeling of Rule 30).
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
from config import ModelConfig, EVAL_EVERY, RULE30_WEIGHTS
from data_generation.Rule30Generator import Rule30Dataset


def train():
    """Pretrain on single-step Rule 30 and save the transformer body for transfer."""
    VOCAB_SIZE = 2
    SEQ_LENGTH = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"Rule30 pre-training on {device} ({n_gpu} GPU(s)) | seq_len {SEQ_LENGTH} "
          f"| batch {ModelConfig.batch_size}")

    model = GeneralTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=ModelConfig.d_model,
        nhead=ModelConfig.n_heads,
        num_layers=ModelConfig.n_layers,
        dim_feedforward=ModelConfig.dim_feedforward,
    ).to(device)

    if n_gpu > 1:
        print(f"Using {n_gpu} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    dataset = Rule30Dataset(num_samples=ModelConfig.num_samples, seq_length=SEQ_LENGTH)
    loader = DataLoader(
        dataset,
        batch_size=ModelConfig.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=ModelConfig.lr, weight_decay=ModelConfig.weight_decay)
    scaler = GradScaler("cuda")

    start = time.time()

    for epoch in range(ModelConfig.epochs):
        model.train()
        total_loss = 0.0
        correct = total = 0

        for state_t, state_t_plus_1 in loader:
            state_t = state_t.to(device, non_blocking=True)
            state_t_plus_1 = state_t_plus_1.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(state_t)

                # left-anchored relabeling + drop first 2 positions
                shifted_targets = torch.roll(state_t_plus_1, shifts=1, dims=1)
                logits_valid = logits[:, 2:, :]
                targets_valid = shifted_targets[:, 2:]
                loss = criterion(logits_valid.reshape(-1, VOCAB_SIZE), targets_valid.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), ModelConfig.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = torch.argmax(logits_valid, dim=-1)
            correct += (preds == targets_valid).sum().item()
            total += targets_valid.numel()

        if epoch % EVAL_EVERY == 0 or epoch == ModelConfig.epochs - 1:
            acc = 100.0 * correct / max(total, 1)   # on-batch (train) accuracy, not held-out
            el = (time.time() - start) / 60
            print(f"Epoch [{epoch+1:3d}/{ModelConfig.epochs}] | Loss {total_loss/len(loader):.4f} "
                  f"| Rule30 next-state acc {acc:6.2f}% (train) | {el:5.1f} min")

    # Unwrap DataParallel before saving so checkpoint keys have no 'module.' prefix.
    to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(to_save, RULE30_WEIGHTS)
    print(f"\nDone in {(time.time()-start)/60:.1f} min. Saved {RULE30_WEIGHTS}")


if __name__ == "__main__":
    train()