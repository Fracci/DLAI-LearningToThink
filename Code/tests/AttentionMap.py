import os
import math
import torch
import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.Transformer import GeneralTransformer
from config import ModelConfig, RULE30_WEIGHTS, ROLLOUT_WEIGHTS, CARRYONLY_WEIGHTS
from src.ArithmeticDataset import CharTokenizer

# CONFIG
LAYER = 0   # change to change the layer
EXPR  = "456+129=C0:6+9=5,C1:5+2=8,C0:4+1=5,A:585"

RUNS = [
    (RULE30_WEIGHTS,      True,  "A_before.png"),   
    ("Weights/Rule30_seed0_modelA.pt",    False, "A_after.png"),   
    ("Weights/seed0_modelB.pt",    False, "B_after.png"),
]


def load_model(checkpoint_path, vocab_size, device, pretrained_rule30=False):
    model = GeneralTransformer(
        vocab_size=vocab_size, d_model=ModelConfig.d_model, nhead=ModelConfig.n_heads,
        num_layers=ModelConfig.n_layers, dim_feedforward=ModelConfig.dim_feedforward,
    ).to(device)

    sd = torch.load(checkpoint_path, map_location=device)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    if pretrained_rule30:
        sd = {k: v for k, v in sd.items()
              if not k.startswith("embedding.") and not k.startswith("fc_out.")}
        model.load_state_dict(sd, strict=False)
    else:
        model.load_state_dict(sd)

    model.eval()
    return model


@torch.no_grad()
def get_attention(model, ids, device, layer_idx=0):
    L = ids.shape[1]
    x = model.embedding(ids) * math.sqrt(model.d_model)

    layer = model.transformer.layers[layer_idx]
    normed = layer.norm1(x)                                 
    mask = model._get_alibi_causal_mask(L, 1, device)       

    _, attn = layer.self_attn(
        normed, normed, normed,
        attn_mask=mask,
        need_weights=True,
        average_attn_weights=False,
    )                                                       
    return attn[0].cpu()


def plot_heads(attn, eq_col, title, out_path):
    nhead = attn.shape[0]
    fig, axes = plt.subplots(2, nhead // 2, figsize=(4 * (nhead // 2), 8))
    for h, ax in enumerate(axes.flat):
        ax.imshow(attn[h], cmap="viridis", aspect="auto")
        ax.set_title(f"head {h}")
        if eq_col is not None:
            ax.axvline(eq_col + 0.5, color="red", lw=1, alpha=0.7) 
        ax.set_xlabel("key (looked-at)")
        ax.set_ylabel("query (writing)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved -> {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = CharTokenizer()
    ids = tok.encode(EXPR).unsqueeze(0).to(device)
    eq_col = EXPR.index("=") if "=" in EXPR else None

    for ckpt, is_raw, out in RUNS:
        if not os.path.exists(ckpt):
            print(f"skip (not found): {ckpt}")
            continue
        model = load_model(ckpt, tok.vocab_size, device, pretrained_rule30=is_raw)
        attn = get_attention(model, ids, device, layer_idx=LAYER)
        plot_heads(attn, eq_col, title=f"{ckpt}  (layer {LAYER})", out_path=out)


if __name__ == "__main__":
    main()