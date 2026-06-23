"""
Pre-training on the carry-only task (multi-GPU).

Same architecture as the other arms (clean transfer swap). The model reads out
the carry-out at each query position from the bit segments -- forced to trace the
variable-distance carry chain. Saves carryonly_pretrained.pt for SeedSweep's
build_A (point PRETRAINED at it).

Metrics (raw accuracy alone can hide failure even at balanced labels):
  acc      = overall query accuracy
  bal_acc  = mean of carry==0 and carry==1 recalls
  rec1     = recall on carry==1 positions
  long_acc = accuracy on LONG-chain positions (gen_dist >= LONG_THRESH)
             -- the load-bearing long-range cases; this is the number that
             proves the long-range circuit actually formed.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import time

from Transformer import Rule30Transformer
from CarryOnlyGenerator import (CarryOnlyDataset, compute_carry, assemble,
                                 sample_ab, VOCAB, IGNORE)
import random


def make_eval_batch(bs, min_n, max_n, chain_max, target_active, max_len, device):
    """Eval batch that also returns per-query gen_dist (for chain-length metrics)."""
    seqs, targs, dists = [], [], []
    for _ in range(bs):
        n = random.randint(min_n, max_n)
        a, b = sample_ab(n, chain_max, target_active)
        seq, tgt = assemble(a, b, max_len, latent="carry_out")
        _, _, dist = compute_carry(a, b)
        dful = torch.full((max_len,), -1, dtype=torch.long)
        dful[2 * n + 2:2 * n + 2 + n] = dist
        seqs.append(seq); targs.append(tgt); dists.append(dful)
    return (torch.stack(seqs).to(device), torch.stack(targs).to(device),
            torch.stack(dists).to(device))


@torch.no_grad()
def evaluate(model, device, cfg, n_batches=8, long_thresh=5):
    model.eval()
    tp1 = fn1 = tn0 = fp0 = 0
    correct = total = 0
    long_correct = long_total = 0
    for _ in range(n_batches):
        seq, tgt, dist = make_eval_batch(cfg["bs"], cfg["min_n"], cfg["max_n"],
                                         cfg["chain_max"], cfg["target_active"],
                                         cfg["max_len"], device)
        with autocast("cuda"):
            preds = torch.argmax(model(seq), dim=-1)
        q = tgt != IGNORE
        p = preds[q]; y = tgt[q]; d = dist[q]
        correct += (p == y).sum().item(); total += y.numel()
        tp1 += ((p == 1) & (y == 1)).sum().item()
        fn1 += ((p != 1) & (y == 1)).sum().item()
        tn0 += ((p == 0) & (y == 0)).sum().item()
        fp0 += ((p != 0) & (y == 0)).sum().item()
        lm = d >= long_thresh
        long_correct += (p[lm] == y[lm]).sum().item(); long_total += int(lm.sum())
    acc = 100.0 * correct / max(total, 1)
    rec1 = 100.0 * tp1 / max(tp1 + fn1, 1)
    rec0 = 100.0 * tn0 / max(tn0 + fp0, 1)
    bal = 0.5 * (rec1 + rec0)
    long_acc = 100.0 * long_correct / max(long_total, 1)
    return acc, bal, rec1, long_acc


def train_carry():
    D_MODEL, NHEAD, NUM_LAYERS, DIM_FF = 256, 8, 6, 1024
    MIN_N, MAX_N = 8, 24
    CHAIN_MAX = 12
    TARGET_ACTIVE = 0.25
    LONG_THRESH = 5

    BATCH_SIZE = 256
    EPOCHS = 300
    NUM_SAMPLES = 20000
    LR, WEIGHT_DECAY, GRAD_CLIP = 1e-3, 0.2, 1.0
    PRINT_EVERY = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    max_len = 3 * MAX_N + 2
    print(f"Carry-only pre-training on {device} ({n_gpu} GPU) | n {MIN_N}-{MAX_N} "
          f"| chain_max {CHAIN_MAX} | active {TARGET_ACTIVE} | batch {BATCH_SIZE}")

    model = Rule30Transformer(vocab_size=VOCAB, d_model=D_MODEL, nhead=NHEAD,
                              num_layers=NUM_LAYERS, dim_feedforward=DIM_FF).to(device)
    if n_gpu > 1:
        print(f"Using {n_gpu} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    ds = CarryOnlyDataset(NUM_SAMPLES, MIN_N, MAX_N, CHAIN_MAX, TARGET_ACTIVE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        pin_memory=True, num_workers=2)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler("cuda")

    eval_cfg = dict(bs=256, min_n=MIN_N, max_n=MAX_N, chain_max=CHAIN_MAX,
                    target_active=TARGET_ACTIVE, max_len=max_len)

    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for seq, target in loader:
            seq = seq.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(seq)
                loss = criterion(logits.reshape(-1, VOCAB), target.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item()

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
            acc, bal, rec1, long_acc = evaluate(model, device, eval_cfg,
                                                long_thresh=LONG_THRESH)
            el = (time.time() - start) / 60
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Loss {total_loss/len(loader):.4f} "
                  f"| acc {acc:5.1f} | bal {bal:5.1f} | rec1 {rec1:5.1f} "
                  f"| long(>= {LONG_THRESH}) {long_acc:5.1f} | {el:4.1f}m")

    to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(to_save, "carryonly_pretrained.pt")
    print(f"\nDone in {(time.time()-start)/60:.1f} min. Saved carryonly_pretrained.pt")
    print("Check: 'long' accuracy near 'acc' => the long-range carry circuit formed.")


if __name__ == "__main__":
    train_carry()