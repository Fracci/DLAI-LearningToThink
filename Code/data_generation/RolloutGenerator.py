"""
RolloutGenerator.py — synthetic data for the rollout pretraining task.

Generates a multi-row Rule 30 evolution (a random first row, then ROWS-1
successor rows) and flattens it row-major into one autoregressive sequence. The
key latent is LONG-RANGE: predicting a cell requires the cell directly above it,
N positions back, where the period N VARIES per sample.
"""
import torch
from torch.utils.data import Dataset
import random

PAD_IDX = 2          # vocab is {0,1,PAD}; PAD doubles as the loss ignore_index


class Rule30RolloutDataset(Dataset):
    """Flattened multi-row Rule 30 rollouts with a mask marking the predictable (non-first-row) cells."""

    def __init__(self, num_samples, min_n, max_n, rows, pad_idx=PAD_IDX):
        self.num_samples = num_samples
        self.min_n = min_n
        self.max_n = max_n
        self.rows = rows
        self.pad_idx = pad_idx
        self.max_len = max_n * rows                       # longest possible flat length

        # Rule 30 lookup indexed by (left*4 + center*2 + right): 100,011,010,001 -> 1.
        self.rule_lookup = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0], dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Build one rollout: random first row, evolve ROWS-1 steps, flatten, pad, and mask."""

        # N (row width = the long-range period) varies per sample, forcing the model
        # to infer the dependency distance from content rather than memorizing it.
        N = random.randint(self.min_n, self.max_n)
        row = torch.randint(0, 2, (N,), dtype=torch.long)
        rows = [row]

        for _ in range(self.rows - 1):
            left = torch.roll(row, shifts=1, dims=0)
            right = torch.roll(row, shifts=-1, dims=0)
            nb = left * 4 + row * 2 + right                # 3-bit neighborhood index
            row = self.rule_lookup[nb]
            rows.append(row)

        flat = torch.cat(rows)                             # row-major flatten
        Lflat = flat.numel()

        seq = torch.full((self.max_len,), self.pad_idx, dtype=torch.long)
        seq[:Lflat] = flat
        x = seq[:-1]
        y = seq[1:]

        # Loss mask: train only on positions that are past the first row and real, not pad.
        pos = torch.arange(self.max_len - 1)
        mask = ((pos + 1) >= N) & ((pos + 1) < Lflat)
        return x, y, mask


if __name__ == "__main__":
    # Shape sanity check plus a tiny printed rollout to eyeball the Rule 30 evolution.
    ds = Rule30RolloutDataset(num_samples=1000, min_n=8, max_n=12, rows=5)
    x, y, mask = ds[0]

    print(f"max padded length: {ds.max_len}")
    print(f"x shape {x.shape} | y shape {y.shape} | mask shape {mask.shape}")
    print(f"trainable (mask True) positions in this sample: {int(mask.sum())}")

    N = 8
    row = torch.randint(0, 2, (N,))
    print("\ntiny rollout preview (N=8, 5 rows):")
    lut = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0])
    
    for r in range(5):
        print("".join(map(str, row.tolist())))
        nb = torch.roll(row, 1) * 4 + row * 2 + torch.roll(row, -1)
        row = lut[nb]