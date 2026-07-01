"""
Rule30Generator.py — synthetic data for the single-step Rule 30 pretraining task.

Generates one Rule 30 transition: a random binary row (state at time t) and its
one-step successor (state at t+1), where each cell's next value is a fixed function
of its 3-cell neighborhood. This is the LOCAL, structurally-compatible arm —
the dependency is short-range (immediate neighbors), unlike the long-range carry
and rollout tasks. Boundaries wrap around (toroidal) via torch.roll.
"""
import torch
from torch.utils.data import Dataset, DataLoader


class Rule30Dataset(Dataset):
    """Yields (state_t, state_t+1) pairs for one Rule 30 step over a random binary row."""

    def __init__(self, num_samples, seq_length):
        self.num_samples = num_samples
        self.seq_length = seq_length

        # Rule 30 successor table, indexed by (left*4 + center*2 + right):
        # 100,011,010,001 -> 1 ; 111,110,101,000 -> 0.
        self.rule_lookup = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0], dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Draw a random row and return it with its Rule 30 one-step successor."""
        state_t = torch.randint(0, 2, (self.seq_length,), dtype=torch.long)

        # Wrap-around (toroidal) neighbors; roll(+1)=left, roll(-1)=right.
        left_neighbors = torch.roll(state_t, shifts=1, dims=0)
        right_neighbors = torch.roll(state_t, shifts=-1, dims=0)

        # Encode each 3-cell neighborhood as a 0..7 index, then look up the successor.
        neighborhoods = (left_neighbors * 4) + (state_t * 2) + (right_neighbors * 1)
        state_t_plus_1 = self.rule_lookup[neighborhoods]

        return state_t, state_t_plus_1


if __name__ == "__main__":
    
    # Quick check: print a small batch of (t) inputs and their (t+1) targets.
    SEQ_LENGTH = 16
    BATCH_SIZE = 4

    dataset = Rule30Dataset(num_samples=10000, seq_length=SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    inputs, targets = next(iter(dataloader))

    print("Input Batch (t):")
    print(inputs)
    print("\nTarget Batch (t+1):")
    print(targets)