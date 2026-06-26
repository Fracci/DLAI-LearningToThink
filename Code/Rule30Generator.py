import torch
from torch.utils.data import Dataset, DataLoader

class Rule30Dataset(Dataset):
    def __init__(self, num_samples, seq_length):
        self.num_samples = num_samples
        self.seq_length = seq_length
        
        # Rule 30 mapping:
        # 111 (7) -> 0, 110 (6) -> 0, 101 (5) -> 0, 100 (4) -> 1
        # 011 (3) -> 1, 010 (2) -> 1, 001 (1) -> 1, 000 (0) -> 0
        self.rule_lookup = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0], dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. On-the-Fly Initialization: Generate a random binary vector for step t
        state_t = torch.randint(0, 2, (self.seq_length,), dtype=torch.long)
        
        # 2. Periodic Boundary Conditions (Wrap-around)
        left_neighbors = torch.roll(state_t, shifts=1, dims=0)
        right_neighbors = torch.roll(state_t, shifts=-1, dims=0)
        
        # 3. Calculate Neighborhood Binary Value (0 to 7)
        neighborhoods = (left_neighbors * 4) + (state_t * 2) + (right_neighbors * 1)
        
        # 4. Apply Rule 30
        state_t_plus_1 = self.rule_lookup[neighborhoods]
        
        return state_t, state_t_plus_1


if __name__ == "__main__":
    SEQ_LENGTH = 16
    BATCH_SIZE = 4
    
    dataset = Rule30Dataset(num_samples=10000, seq_length=SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    inputs, targets = next(iter(dataloader))
    
    print("Input Batch (t):")
    print(inputs)
    print("\nTarget Batch (t+1):")
    print(targets)