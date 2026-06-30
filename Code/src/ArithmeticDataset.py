"""
ArithmeticDataset.py — tokenizer and data for the scratchpad-addition target task.

The transfer target is multi-digit addition written out with an explicit
chain-of-thought "scratchpad": each digit position is added with its carry shown,
then the final answer. Training on this teaches the model to carry; the project
then tests whether it generalizes to LONGER numbers than seen in training.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import random


class CharTokenizer:
    """Character-level tokenizer for the arithmetic scratchpad; index 0 is <PAD>."""

    def __init__(self):
        # Vocab = PAD + the 16 characters that appear in a scratchpad string.
        # PAD is index 0 so it doubles as the ignore/pad id everywhere downstream.
        chars = list("0123456789+=C:,A")
        self.pad_token = "<PAD>"
        self.vocab = [self.pad_token] + chars

        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)            # 17 (1 pad + 16 chars)
        self.pad_idx = self.char_to_idx[self.pad_token]

    def encode(self, text, max_len=None):
        """Encode a string to a 1-D LongTensor, optionally right-padded to max_len."""
        indices = [self.char_to_idx[c] for c in text]
        if max_len is not None:
            # Hard boundary: a sequence longer than max_len is a fatal error rather
            # than silently truncated, since truncation would corrupt the answer.
            if len(indices) > max_len:
                raise ValueError(
                    f"Encoded sequence length {len(indices)} exceeds max_len={max_len}. "
                    "Increase max_seq_len or reduce the generation length."
                )
            
            padding = [self.pad_idx] * (max_len - len(indices))
            indices = indices + padding
            
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, tensor):
        """Decode a tensor of ids back to a string, dropping PAD tokens."""
        chars = [self.idx_to_char[idx.item()] for idx in tensor if idx.item() != self.pad_idx]
        return "".join(chars)


class ScratchpadAdditionDataset(Dataset):
    """Yields (x, y) next-token pairs of 'n1+n2=<scratchpad>,A:<answer>' strings."""

    def __init__(self, num_samples, min_digits, max_digits, tokenizer, max_seq_len):
        self.num_samples = num_samples
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.num_samples

    def generate_scratchpad(self, n1, n2):
        """Build the full 'n1+n2=...,A:answer' string with per-digit carry steps (LSB first)."""
        # Reverse both operands so index 0 is the least-significant digit, and
        # zero-pad the shorter one so the column loop is uniform.
        s1, s2 = str(n1)[::-1], str(n2)[::-1]
        max_len = max(len(s1), len(s2))
        s1 = s1.ljust(max_len, '0')
        s2 = s2.ljust(max_len, '0')

        carry = 0
        steps = []
        # One scratchpad step per digit column: "C<carry_in>:d1+d2=<remainder>".
        for i in range(max_len):
            d1 = int(s1[i])
            d2 = int(s2[i])
            total = d1 + d2 + carry
            current_carry = carry
            carry = total // 10
            remainder = total % 10
            steps.append(f"C{current_carry}:{d1}+{d2}={remainder}")

        # Final carry out of the most-significant column gets its own step.
        if carry > 0:
            steps.append(f"C{carry}:0+0={carry}")

        final_answer = str(n1 + n2)
        target_str = ",".join(steps) + f",A:{final_answer}"
        input_str = f"{n1}+{n2}="
        return input_str + target_str

    def __getitem__(self, idx):
        """Sample a random addition problem and return (x, y) shifted by one token."""
        # Operand lengths are drawn INDEPENDENTLY in [min_digits, max_digits],
        # so the dataset is a mixture of digit-length combinations (documented).
        n1 = random.randint(10**(self.min_digits-1), 10**self.max_digits - 1)
        n2 = random.randint(10**(self.min_digits-1), 10**self.max_digits - 1)

        full_str = self.generate_scratchpad(n1, n2)
        tensor_seq = self.tokenizer.encode(full_str, max_len=self.max_seq_len)

        # Next-token prediction: predict token t+1 from tokens up to t.
        x = tensor_seq[:-1]
        y = tensor_seq[1:]
        return x, y


if __name__ == "__main__":
    # Quick visual sanity check of one scratchpad string and a batch's shapes.
    tokenizer = CharTokenizer()

    dataset = ScratchpadAdditionDataset(
        num_samples=1000,
        min_digits=3,
        max_digits=4,
        tokenizer=tokenizer,
        max_seq_len=64
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    x, y = next(iter(dataloader))

    print("RAW STRING PREVIEW:")
    sample_n1 = 456
    sample_n2 = 129
    print(f"Adding {sample_n1} and {sample_n2}:")
    print(dataset.generate_scratchpad(sample_n1, sample_n2))

    print("\nBATCH TENSOR PREVIEW:")
    print(f"X shape: {x.shape} (Input tokens)")
    print(f"Y shape: {y.shape} (Target tokens shifted by 1)")
    print("\nDecoded Batch 0 (X):")
    print(tokenizer.decode(x[0]))