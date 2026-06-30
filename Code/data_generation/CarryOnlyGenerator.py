"""
CarryOnlyGenerator.py — synthetic data for the carry-only pretraining task.

Isolates the ONE operation addition needs to generalize across lengths: carry
propagation. Each sample is two bit-strings a,b; from them we compute the carry
chain and ask the model to predict a latent (carry_out / carry_in / gen_dist) at
query positions. Chains are explicitly PLANTED so carries propagate over long,
VARIABLE distances — the matched long-range structure the project tests transfer
from. 
"""
import torch
from torch.utils.data import Dataset
import random

# Token ids and the special targets/clamp used throughout.
ZERO, ONE, SEP, QUERY, PAD = 0, 1, 2, 3, 4
VOCAB = 5
IGNORE = -100             # CrossEntropy ignore_index for non-query positions
TARGET_ACTIVE = 0.25      # probability of starting a planted carry chain
GEN_DIST_MAX = 24         # gen_dist labels clamped to [0, GEN_DIST_MAX] -> 13 classes


def sample_ab(n, chain_max=12, target_active=TARGET_ACTIVE):
    """Build two bit-strings a,b whose addition contains planted, variable-length carry chains."""
    a = torch.zeros(n, dtype=torch.long)
    b = torch.zeros(n, dtype=torch.long)

    # A column generates a carry if a=b=1, propagates an incoming carry if exactly
    # one of a,b is 1, and kills it if both are 0.
    def set_generate(i): a[i] = 1; b[i] = 1

    def set_propagate(i):
        if random.random() < 0.5: a[i] = 1
        else: b[i] = 1

    def set_kill(i): pass                              # leave both 0

    # Without planting, i.i.d. bits almost never produce long chains; we explicitly
    # seed a generate then a run of propagates so long-range carries actually occur.
    i = 0
    while i < n:

        if random.random() < target_active:
            set_generate(i); i += 1
            run = random.randint(0, chain_max)         # variable chain length

            for _ in range(run):
                if i >= n: break
                set_propagate(i); i += 1

            if i < n and random.random() < 0.5:        # sometimes end with a kill
                set_kill(i); i += 1

        else:
            quiet = random.randint(1, 3)               # short non-seeding gap
            for _ in range(quiet):
                if i >= n: break
                if random.random() < 0.5: set_kill(i)
                else: set_propagate(i)
                i += 1

    return a, b


def compute_carry(a, b):
    """Return (carry_out, carry_in, gen_dist) per column; gen_dist = distance back to the carry's origin."""
    n = len(a)
    cout = torch.zeros(n, dtype=torch.long)
    cin = torch.zeros(n, dtype=torch.long)
    dist = torch.zeros(n, dtype=torch.long)

    # `origin` tracks the column index where the currently-live carry was generated,
    # so gen_dist can measure how far a carry has propagated. -1 means no live carry.
    carry, origin = 0, -1

    for i in range(n):
        cin[i] = carry
        s = int(a[i]) + int(b[i])

        if s == 2:                 # generate: a fresh carry starts here
            c = 1; origin = i
        elif s == 1:               # propagate: keep carry AND its origin (reset if none)
            c = carry; origin = origin if carry == 1 else -1
        else:                      # kill: no carry, no origin
            c = 0; origin = -1

        cout[i] = c
        dist[i] = (i - origin) if c == 1 else 0
        carry = c

    return cout, cin, dist


def assemble(a, b, max_len, latent="carry_out", gen_dist_max=GEN_DIST_MAX):
    """Pack a,b into the [a|SEP|b|SEP|QUERY...] sequence and build the latent target at query positions."""
    n = len(a)
    cout, cin, dist = compute_carry(a, b)

    seq = torch.full((max_len,), PAD, dtype=torch.long)
    seq[0:n] = a
    seq[n] = SEP
    seq[n + 1:2 * n + 1] = b
    seq[2 * n + 1] = SEP
    qstart = 2 * n + 2

    # Query tokens are CONSTANT (all QUERY): the model can't use a neighboring
    # carry as a stepping-stone, forcing a true variable-distance lookback.
    seq[qstart:qstart + n] = QUERY

    if latent == "carry_out":   vals = cout
    elif latent == "carry_in":  vals = cin

    # gen_dist is clamped to GEN_DIST_MAX -> (GEN_DIST_MAX+1) probe classes. The
    # clamp caps the long-range signal, so keep it >= the real chain lengths.
    elif latent == "gen_dist":  vals = torch.clamp(dist, max=gen_dist_max)
    else: raise ValueError(latent)

    target = torch.full((max_len,), IGNORE, dtype=torch.long)
    target[qstart:qstart + n] = vals      # supervise only the query span
    return seq, target


class CarryOnlyDataset(Dataset):
    """Dataset of carry-only sequences; pretraining target is carry_out at the query positions."""

    def __init__(self, num_samples, min_n, max_n, chain_max=12, target_active=TARGET_ACTIVE):
        self.num_samples = num_samples
        self.min_n, self.max_n = min_n, max_n
        self.chain_max = chain_max
        self.target_active = target_active
        self.max_len = 3 * max_n + 2          # a + SEP + b + SEP + query, at n=max_n

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Sample a variable length n, plant chains, and return (seq, carry_out target)."""
        n = random.randint(self.min_n, self.max_n)
        a, b = sample_ab(n, self.chain_max, self.target_active)
        seq, target = assemble(a, b, self.max_len, latent="carry_out")
        return seq, target


def _balance(vals):
    """Count carry-out==1 vs total (used by the sanity check to verify label balance)."""
    ones = int((vals == 1).sum()); tot = len(vals)
    return ones, tot


if __name__ == "__main__":
    # Sanity check: confirm chains are balanced and reach long distances
    # Used also to fix the target_active parameter to produce a reasonable label balance

    print("Sanity check: label balance and chain lengths over 2000 samples\n")
    tot_ones = tot = 0
    maxdist = 0
    long_chains = 0

    for _ in range(2000):
        n = random.randint(8, 24)
        a, b = sample_ab(n, chain_max=12, target_active=TARGET_ACTIVE)
        cout, cin, dist = compute_carry(a, b)
        o, t = _balance(cout); tot_ones += o; tot += t
        maxdist = max(maxdist, int(dist.max()))
        if int(dist.max()) >= 5: long_chains += 1

    print(f"carry-out == 1 fraction : {100.0*tot_ones/tot:5.1f}% ")
    print(f"max gen_dist seen        : {maxdist}")
    print(f"samples with a long chain (gen_dist>=5): {100.0*long_chains/2000:4.1f}%\n")

    n = 16
    a, b = sample_ab(n, chain_max=12, target_active=TARGET_ACTIVE)
    cout, cin, dist = compute_carry(a, b)
    print("a (LSB->):", a.tolist())
    print("b (LSB->):", b.tolist())
    print("carry_out:", cout.tolist())
    print("carry_in :", cin.tolist())
    print("gen_dist :", dist.tolist())