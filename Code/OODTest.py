import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# Import your custom modules
from Transformer import Rule30Transformer
from Rule30Generator import Rule30Dataset

def run_ood_length_test():
    # ---------------------------------------------------------
    # 1. Hyperparameters
    # ---------------------------------------------------------
    D_MODEL = 128
    TRAIN_SEQ_LENGTH = 256          # What the model was trained on
    TEST_SEQ_LENGTH = 512           # The OOD Length (Double the size!)
    BATCH_SIZE = 128
    TEST_SAMPLES = 5000             # Let's test on 5000 completely new sequences
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing OOD Length Test on device: {device}")
    print(f"Testing on Sequence Length: {TEST_SEQ_LENGTH}")

    # ---------------------------------------------------------
    # 2. Load the Pre-Trained Model
    # ---------------------------------------------------------
    model = Rule30Transformer(vocab_size=2, d_model=D_MODEL, nhead=4, num_layers=4).to(device)
    
    checkpoint_path = "kaggle/working/rule30_pretrained_gpu.pt"
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Successfully loaded pre-trained Transformer weights.")
    except Exception as e:
        print(f"Error loading weights: {e}. Cannot run test without the trained model.")
        return

    # Set to evaluation mode (turns off dropout, etc.)
    model.eval()

    # ---------------------------------------------------------
    # 3. Generate OOD Data and Evaluate
    # ---------------------------------------------------------
    test_dataset = Rule30Dataset(num_samples=TEST_SAMPLES, seq_length=TEST_SEQ_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)
    
    total_correct = 0
    total_preds = 0
    
    print("\nRunning Forward Passes on Length 512...")
    
    with torch.no_grad():
        for state_t, state_t_plus_1 in test_loader:
            state_t = state_t.to(device, non_blocking=True)
            state_t_plus_1 = state_t_plus_1.to(device, non_blocking=True)
            
            with autocast():
                logits = model(state_t)
                shifted_targets = torch.roll(state_t_plus_1, shifts=1, dims=1)
                
                # Ignore the first two causally blind tokens
                logits_valid = logits[:, 2:, :]
                targets_valid = shifted_targets[:, 2:]
                
                preds = torch.argmax(logits_valid, dim=-1)
                total_correct += (preds == targets_valid).sum().item()
                total_preds += targets_valid.numel()
                
    accuracy = (total_correct / total_preds) * 100
    
    # ---------------------------------------------------------
    # 4. Final Verdict
    # ---------------------------------------------------------
    print("\n==================================================")
    print(f"OOD LENGTH GENERALIZATION ACCURACY: {accuracy:.2f}%")
    print("==================================================")
    
    if accuracy > 95.0:
        print("\nSUCCESS! Your ALiBi linear slopes and Rule 30 training worked perfectly.")
        print("The model maintained its causal logic flawlessly on sequences it has never seen.")
    else:
        print("\nFAILURE. The model's logic degraded over longer sequences. Check your ALiBi slopes.")

if __name__ == "__main__":
    run_ood_length_test()