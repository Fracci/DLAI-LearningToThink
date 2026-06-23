"""
Multi-step Rule 30 ROLLOUT generator.

Single-step Rule 30 predicts state_{t+1} from state_t in one parallel pass, so
each output depends only on a LOCAL 3-cell window -> the model learns a local
circuit and never builds long-range routing. This dataset instead evolves a row
ROWS times under Rule 30 and flattens all rows row-major into ONE autoregressive
sequence:

    [ row0 (random seed) | row1 | row2 | ... | row_{T-1} ]   length = N * ROWS

To emit cell i of row r, the model must attend back ~N tokens to row r-1's cells
[i-1, i, i+1] -- a genuinely long-range retrieval. N (the period) VARIES per
sample, so the model cannot hardcode a fixed offset; it must learn content-based
induction / copy-with-transform -- the same machinery long-range arithmetic
retrieval (carry propagation, digit fetching) reuses.

Returns (x, y, mask):
  x, y  : next-token shifted (x = seq[:-1], y = seq[1:])
  mask  : True only where the prediction is a row>=1 token (row 0 is the
          unpredictable random seed; pad positions excluded). Loss/accuracy
          are computed on mask only.

Vocab: {0, 1, PAD=2}. The transfer pipeline reuses only transformer.* / final_norm.*
(it drops embedding/fc_out), so the vocab size here is independent of the
arithmetic vocab.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import random

PAD_IDX = 2


class Rule30RolloutDataset(Dataset):
    def __init__(self, num_samples, min_n, max_n, rows, pad_idx=PAD_IDX):
        self.num_samples = num_samples
        self.min_n = min_n
        self.max_n = max_n
        self.rows = rows
        self.pad_idx = pad_idx
        self.max_len = max_n * rows                       # padded sequence length
        self.rule_lookup = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0], dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        N = random.randint(self.min_n, self.max_n)        # variable period
        row = torch.randint(0, 2, (N,), dtype=torch.long)
        rows = [row]
        for _ in range(self.rows - 1):
            left = torch.roll(row, shifts=1, dims=0)
            right = torch.roll(row, shifts=-1, dims=0)
            nb = left * 4 + row * 2 + right                # periodic boundaries
            row = self.rule_lookup[nb]
            rows.append(row)
        flat = torch.cat(rows)                            # length N * rows
        Lflat = flat.numel()

        seq = torch.full((self.max_len,), self.pad_idx, dtype=torch.long)
        seq[:Lflat] = flat
        x = seq[:-1]
        y = seq[1:]
        pos = torch.arange(self.max_len - 1)
        # predict token at seq[k+1]; keep it iff that token is row>=1 and not pad
        mask = ((pos + 1) >= N) & ((pos + 1) < Lflat)
        return x, y, mask


if __name__ == "__main__":
    ds = Rule30RolloutDataset(num_samples=1000, min_n=8, max_n=12, rows=5)
    x, y, mask = ds[0]
    print(f"max padded length: {ds.max_len}")
    print(f"x shape {x.shape} | y shape {y.shape} | mask shape {mask.shape}")
    print(f"trainable (mask True) positions in this sample: {int(mask.sum())}")
    # show the rollout as rows for a tiny case
    N = 8
    row = torch.randint(0, 2, (N,))
    print("\n--- tiny rollout preview (N=8, 5 rows) ---")
    lut = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0])
    for r in range(5):
        print("".join(map(str, row.tolist())))
        nb = torch.roll(row, 1) * 4 + row * 2 + torch.roll(row, -1)
        row = lut[nb]