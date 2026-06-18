import torch
from torch.utils.data import Dataset, DataLoader
import random

class CharTokenizer:
    """
    A simple character-level tokenizer for the mathematical scratchpad.
    Converts strings like 'C0:5+3=8,A:168' into PyTorch tensors.
    """
    def __init__(self):
        # Define the exact vocabulary needed for the scratchpad
        chars = list("0123456789+=C:,A")
        self.pad_token = "<PAD>"
        self.vocab = [self.pad_token] + chars
        
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_idx = self.char_to_idx[self.pad_token]

    def encode(self, text, max_len=None):
        indices = [self.char_to_idx[c] for c in text]
        if max_len is not None:
            # Pad sequences to ensure uniform tensor shapes for batching
            padding = [self.pad_idx] * (max_len - len(indices))
            indices = indices + padding
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, tensor):
        chars = [self.idx_to_char[idx.item()] for idx in tensor if idx.item() != self.pad_idx]
        return "".join(chars)


class ScratchpadAdditionDataset(Dataset):
    def __init__(self, num_samples, min_digits, max_digits, tokenizer, max_seq_len):
        """
        Args:
            num_samples (int): Number of virtual samples per epoch.
            min_digits (int): Minimum length of the numbers to add.
            max_digits (int): Maximum length of the numbers to add.
            tokenizer (CharTokenizer): The tokenizer to encode strings.
            max_seq_len (int): The padded length for the tensor outputs.
        """
        self.num_samples = num_samples
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.num_samples

    def generate_scratchpad(self, n1, n2):
        # Convert to strings and reverse them to process right-to-left (ones, tens, hundreds)
        s1, s2 = str(n1)[::-1], str(n2)[::-1]
        max_len = max(len(s1), len(s2))
        
        # Pad the shorter number with zeros for column alignment
        s1 = s1.ljust(max_len, '0')
        s2 = s2.ljust(max_len, '0')
        
        carry = 0
        steps = []
        
        # Build the step-by-step causal trace
        for i in range(max_len):
            d1 = int(s1[i])
            d2 = int(s2[i])
            
            total = d1 + d2 + carry
            current_carry = carry
            carry = total // 10
            remainder = total % 10
            
            # Format: C[carry_in]:[digit1]+[digit2]=[remainder]
            steps.append(f"C{current_carry}:{d1}+{d2}={remainder}")
            
        if carry > 0:
            steps.append(f"C{carry}:0+0={carry}")
            
        final_answer = str(n1 + n2)
        
        # Join steps with commas and append the Answer token 'A'
        target_str = ",".join(steps) + f",A:{final_answer}"
        input_str = f"{n1}+{n2}="
        
        # In a standard autoregressive model, the full sequence is "InputTarget"
        full_sequence = input_str + target_str
        return full_sequence

    def __getitem__(self, idx):
        # On-the-fly generation to prevent memorization
        n1 = random.randint(10**(self.min_digits-1), 10**self.max_digits - 1)
        n2 = random.randint(10**(self.min_digits-1), 10**self.max_digits - 1)
        
        full_str = self.generate_scratchpad(n1, n2)
        
        # Encode and pad the sequence
        tensor_seq = self.tokenizer.encode(full_str, max_len=self.max_seq_len)
        
        x = tensor_seq[:-1]
        y = tensor_seq[1:]
        
        return x, y


if __name__ == "__main__":
    tokenizer = CharTokenizer()
    
    # Let's generate 3-digit and 4-digit addition problems
    dataset = ScratchpadAdditionDataset(
        num_samples=1000, 
        min_digits=3, 
        max_digits=4, 
        tokenizer=tokenizer, 
        max_seq_len=64
    )
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    x, y = next(iter(dataloader))
    
    print("--- RAW STRING PREVIEW ---")
    # Generating a raw string to visualize the scratchpad logic
    sample_n1 = 456
    sample_n2 = 129
    print(f"Adding {sample_n1} and {sample_n2}:")
    print(dataset.generate_scratchpad(sample_n1, sample_n2))
    
    print("\n--- BATCH TENSOR PREVIEW ---")
    print(f"X shape: {x.shape} (Input tokens)")
    print(f"Y shape: {y.shape} (Target tokens shifted by 1)")
    print("\nDecoded Batch 0 (X):")
    print(tokenizer.decode(x[0]))