import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import time
import random
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.Transformer import GeneralTransformer
from config import ModelConfig, EVAL_EVERY, CARRYONLY_WEIGHTS
from data_generation.CarryOnlyGenerator import (CarryOnlyDataset, compute_carry, assemble,
                                 sample_ab, VOCAB, IGNORE, TARGET_ACTIVE)


def make_eval_batch(bs, min_n, max_n, chain_max, target_active, max_len, device):
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
    MIN_N, MAX_N = 8, 24
    CHAIN_MAX = 12
    LONG_THRESH = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    max_len = 3 * MAX_N + 2
    print(f"Carry-only pre-training on {device} ({n_gpu} GPU) | n {MIN_N}-{MAX_N} "
          f"| chain_max {CHAIN_MAX} | active {TARGET_ACTIVE} | batch {ModelConfig.batch_size}")

    model = GeneralTransformer(vocab_size=VOCAB, d_model=ModelConfig.d_model, nhead=ModelConfig.n_heads,
                              num_layers=ModelConfig.n_layers, dim_feedforward=ModelConfig.dim_feedforward).to(device)
    if n_gpu > 1:
        print(f"Using {n_gpu} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    ds = CarryOnlyDataset(ModelConfig.num_samples, MIN_N, MAX_N, CHAIN_MAX, TARGET_ACTIVE)
    loader = DataLoader(ds, batch_size=ModelConfig.batch_size, shuffle=True,
                        pin_memory=True, num_workers=2)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    optimizer = AdamW(model.parameters(), lr=ModelConfig.lr, weight_decay=ModelConfig.weight_decay)
    scaler = GradScaler("cuda")

    eval_cfg = dict(bs=256, min_n=MIN_N, max_n=MAX_N, chain_max=CHAIN_MAX,
                    target_active=TARGET_ACTIVE, max_len=max_len)

    start = time.time()
    for epoch in range(ModelConfig.epochs):
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
            nn.utils.clip_grad_norm_(model.parameters(), ModelConfig.grad_clip)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item()

        if epoch % EVAL_EVERY == 0 or epoch == ModelConfig.epochs - 1:
            acc, bal, rec1, long_acc = evaluate(model, device, eval_cfg,
                                                long_thresh=LONG_THRESH)
            el = (time.time() - start) / 60
            print(f"Epoch [{epoch+1:3d}/{ModelConfig.epochs}] | Loss {total_loss/len(loader):.4f} "
                  f"| acc {acc:5.1f} | bal {bal:5.1f} | rec1 {rec1:5.1f} "
                  f"| long(>= {LONG_THRESH}) {long_acc:5.1f} | {el:4.1f}m")

    to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(to_save, CARRYONLY_WEIGHTS)
    print(f"\nDone in {(time.time()-start)/60:.1f} min. Saved {CARRYONLY_WEIGHTS}")

if __name__ == "__main__":
    train_carry()