"""
Carry-only pretraining task: variable-distance, carry-like long-range structure.

Third point on the structural-overlap spectrum:
  single-step Rule 30 = local, compatible          (best transfer so far)
  multi-step rollout   = long-range, FIXED period   (mismatched -> weak transfer)
  carry-only (this)    = long-range, VARIABLE dist  (matched to addition's carry)

Task. Two binary numbers a, b (LSB-first). At each position the model emits the
CARRY-OUT bit. A position is generate (a=b=1 -> start a carry), propagate
(a+b=1 -> pass the incoming carry) or kill (a=b=0 -> stop it). A carry can ride
an arbitrarily long run of propagate positions, so the carry at position i
depends on the nearest GENERATE below it -- a variable, input-dependent distance.

Layout (LSB-first):
    [ a_0..a_{n-1} | SEP | b_0..b_{n-1} | SEP | Q Q ... Q ]
The n query tokens Q are CONSTANT (no carry info), so the model cannot use a
neighbouring carry as a stepping stone -- it must recompute each carry from the
bits by tracing back to the nearest generate. Target at query i = carry-out of i;
loss on query positions only.

KEY DESIGN (vs the naive i.i.d. version): carry chains are PLANTED explicitly.
A naive sampler starves generates (chains never seed) and yields almost-all-zero
carry labels with no long-range gen_dist. Here we lay down generate seeds and
deliberate propagate runs of controlled length, so that (a) carry labels are
roughly balanced and (b) long chains (large gen_dist) are guaranteed present.
CHAIN_MAX sets the longest planted chain; TARGET_ACTIVE the fraction of positions
under an active carry.

Vocab {0,1,SEP=2,QUERY=3,PAD=4}; transfer reuses only transformer.*/final_norm.*.
"""
import torch
from torch.utils.data import Dataset
import random

ZERO, ONE, SEP, QUERY, PAD = 0, 1, 2, 3, 4
VOCAB = 5
IGNORE = -100


def sample_ab(n, chain_max=12, target_active=0.25):
    """Build LSB-first a, b by PLANTING carry chains.

    We sweep positions left to right (LSB->MSB). At each step we either:
      - start/continue a carry chain: place a generate, then a run of propagates
        of a length drawn up to chain_max (creating gen_dist up to that length),
      - or place 'quiet' kill/propagate-without-carry positions.
    target_active tunes how much of the string is under an active carry, which
    balances the carry-out labels around ~50%.
    """
    a = torch.zeros(n, dtype=torch.long)
    b = torch.zeros(n, dtype=torch.long)

    def set_generate(i): a[i] = 1; b[i] = 1            # a=b=1
    def set_propagate(i):
        if random.random() < 0.5: a[i] = 1
        else: b[i] = 1                                  # exactly one bit
    def set_kill(i): pass                               # both 0

    i = 0
    while i < n:
        if random.random() < target_active:
            # plant a chain: one generate seed, then a propagate run
            set_generate(i); i += 1
            run = random.randint(0, chain_max)
            for _ in range(run):
                if i >= n: break
                set_propagate(i); i += 1
            # optionally terminate with a kill (cheap, keeps lengths varied)
            if i < n and random.random() < 0.5:
                set_kill(i); i += 1
        else:
            # quiet region: a few non-seeding positions (kill or lone propagate)
            quiet = random.randint(1, 3)
            for _ in range(quiet):
                if i >= n: break
                if random.random() < 0.5: set_kill(i)
                else: set_propagate(i)                  # propagate w/o incoming carry -> passes 0
                i += 1
    return a, b


def compute_carry(a, b):
    """Return (carry_out, carry_in, gen_dist) per position, LSB-first.
    gen_dist = distance back to the generate that started the current carry-out
    (0 when no carry is leaving)."""
    n = len(a)
    cout = torch.zeros(n, dtype=torch.long)
    cin = torch.zeros(n, dtype=torch.long)
    dist = torch.zeros(n, dtype=torch.long)
    carry, origin = 0, -1
    for i in range(n):
        cin[i] = carry
        s = int(a[i]) + int(b[i])
        if s == 2:                 # generate
            c = 1; origin = i
        elif s == 1:               # propagate
            c = carry; origin = origin if carry == 1 else -1
        else:                      # kill
            c = 0; origin = -1
        cout[i] = c
        dist[i] = (i - origin) if c == 1 else 0
        carry = c
    return cout, cin, dist


def assemble(a, b, max_len, latent="carry_out"):
    """Padded sequence + target at query positions.
    latent: 'carry_out' (pretraining), 'carry_in' or 'gen_dist' (probing)."""
    n = len(a)
    cout, cin, dist = compute_carry(a, b)
    seq = torch.full((max_len,), PAD, dtype=torch.long)
    seq[0:n] = a
    seq[n] = SEP
    seq[n + 1:2 * n + 1] = b
    seq[2 * n + 1] = SEP
    qstart = 2 * n + 2
    seq[qstart:qstart + n] = QUERY

    if latent == "carry_out":   vals = cout
    elif latent == "carry_in":  vals = cin
    elif latent == "gen_dist":  vals = torch.clamp(dist, max=5)   # 6 classes 0..4,5+
    else: raise ValueError(latent)

    target = torch.full((max_len,), IGNORE, dtype=torch.long)
    target[qstart:qstart + n] = vals
    return seq, target


class CarryOnlyDataset(Dataset):
    def __init__(self, num_samples, min_n, max_n, chain_max=12, target_active=0.25):
        self.num_samples = num_samples
        self.min_n, self.max_n = min_n, max_n
        self.chain_max = chain_max
        self.target_active = target_active
        self.max_len = 3 * max_n + 2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        n = random.randint(self.min_n, self.max_n)
        a, b = sample_ab(n, self.chain_max, self.target_active)
        seq, target = assemble(a, b, self.max_len, latent="carry_out")
        return seq, target


def _balance(vals):
    ones = int((vals == 1).sum()); tot = len(vals)
    return ones, tot


if __name__ == "__main__":
    print("Sanity check: label balance and chain lengths over 2000 samples\n")
    tot_ones = tot = 0
    maxdist = 0
    long_chains = 0
    for _ in range(2000):
        n = random.randint(8, 24)
        a, b = sample_ab(n, chain_max=12, target_active=0.25)
        cout, cin, dist = compute_carry(a, b)
        o, t = _balance(cout); tot_ones += o; tot += t
        maxdist = max(maxdist, int(dist.max()))
        if int(dist.max()) >= 5: long_chains += 1
    print(f"carry-out == 1 fraction : {100.0*tot_ones/tot:5.1f}%  (want ~30-60%)")
    print(f"max gen_dist seen        : {maxdist}  (want clearly > 1)")
    print(f"samples with a long chain (gen_dist>=5): {100.0*long_chains/2000:4.1f}%\n")

    n = 16
    a, b = sample_ab(n, chain_max=12, target_active=0.25)
    cout, cin, dist = compute_carry(a, b)
    print("a (LSB->):", a.tolist())
    print("b (LSB->):", b.tolist())
    print("carry_out:", cout.tolist())
    print("carry_in :", cin.tolist())
    print("gen_dist :", dist.tolist())