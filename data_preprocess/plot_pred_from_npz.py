#!/usr/bin/env python3
import argparse
import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import esm


def load_alphabet(model_name: str):
    """
    Load ESM alphabet and build idx -> token map.
    Returns:
      idx2tok: dict[int, str]
      pad_idx: int
    """
    _, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
    char2idx = alphabet.tok_to_idx
    pad_idx = alphabet.padding_idx
    idx2tok = {v: k for k, v in char2idx.items()}
    return idx2tok, pad_idx


def load_sequence_tokens(seq_npz_path, key=None):
    """
    Load the first row (query) of the combined sequence from NPZ.
    If key is None, use the first key in the NPZ.
    Returns:
      tokens: np.ndarray[int] of shape (L,)
      used_key: str
    """
    data = np.load(seq_npz_path, allow_pickle=True)
    keys = list(data.keys())
    if not keys:
        raise ValueError(f"No keys found in NPZ: {seq_npz_path}")

    if key is None:
        used_key = keys[0]
    else:
        if key in data:
            used_key = key
        elif key.lower() in data:
            used_key = key.lower()
        elif key.upper() in data:
            used_key = key.upper()
        else:
            raise KeyError(
                f"Key '{key}' not found in NPZ. Available keys (first 10): {keys[:10]}"
            )

    arr = data[used_key]  # shape (depth, L)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected array shape for key={used_key}: {arr.shape}")

    tokens = arr[0].astype(int)  # first row = query sequence
    return tokens, used_key


def untokenize(tokens, idx2tok, pad_idx):
    """
    Convert token indices to characters.
    - PAD -> '.'
    - Special tokens (e.g. '<cls>') -> '*'
    - Normal AA / 'X' etc -> as-is
    """
    chars = []
    for t in tokens:
        t = int(t)
        if t == pad_idx:
            chars.append(".")
        else:
            tok = idx2tok.get(t, "?")
            if len(tok) != 1:
                chars.append("*")
            else:
                chars.append(tok)
    return np.array(chars)  # array of shape (L,)


def load_probs(pred_csv, sample_idx=0, key=None):
    """
    Load per-position probabilities from prediction CSV.

    CSV columns:
      SampleIdx,Key,ChainType,Position,PredLabel,Probability

    Returns:
      probs_dict: dict[int, float]  (pos -> prob)
    """
    probs_dict = {}
    with open(pred_csv) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if int(row["SampleIdx"]) != sample_idx:
                continue
            if key is not None and row["Key"] != key:
                continue
            pos = int(row["Position"])
            pr = float(row["Probability"])
            probs_dict[pos] = pr

    if not probs_dict:
        raise ValueError(
            f"No predictions found in {pred_csv} "
            f"for SampleIdx={sample_idx}"
            + (f", Key={key}" if key is not None else "")
        )
    return probs_dict


def split_chains(tokens, eoc_id, chain_names=None):
    """
    Split a combined token sequence into chains using EOC token id.

    tokens: 1D np.array of ints (length L)
    eoc_id: int, token id marking end-of-chain

    Returns:
      chain_slices: list[(start, end)]   # inclusive start, exclusive end
      chain_labels: list[str]
    """
    eocs = np.where(tokens == eoc_id)[0]
    starts = [0] + eocs.tolist()
    ends = eocs.tolist() + [len(tokens)]

    # default names: L, H, AG0, AG1, ...
    if chain_names is None:
        labels = ["Lchain", "Hchain"] + [f"AGchain_{i}" for i in range(max(0, len(starts) - 2))]
    else:
        labels = chain_names

    labels = labels[: len(starts)]
    return list(zip(starts, ends)), labels


def build_chain_views(tokens, chars, probs_dict, eoc_id, pad_idx):
    """
    Build per-chain sequences and probs.

    Returns:
      chain_labels: list[str]
      chain_chars:  list[np.ndarray[str]]   # per-chain residue chars
      chain_probs:  list[np.ndarray[float]]
    """
    slices, labels = split_chains(tokens, eoc_id)

    chain_labels = []
    chain_chars = []
    chain_probs = []

    for (start, end), label in zip(slices, labels):
        idxs = np.arange(start, end)

        # remove EOC positions
        mask = tokens[idxs] != eoc_id
        idxs = idxs[mask]

        if len(idxs) == 0:
            continue

        c = chars[idxs]
        p = np.array([probs_dict.get(int(i), 0.0) for i in idxs], dtype=float)

        # skip pure PAD segments
        if np.all(tokens[idxs] == pad_idx):
            continue

        chain_labels.append(label)
        chain_chars.append(c)
        chain_probs.append(p)

    return chain_labels, chain_chars, chain_probs


def plot_chain_grid(chain_labels, chain_chars, chain_probs, out_png, vmax=1.0):
    """
    Plot one row per chain, each residue as colored square with letter.
    Adds residue-number labels every 10 residues (per chain numbering).
    """
    n_chains = len(chain_labels)
    if n_chains == 0:
        raise ValueError("No non-empty chains to plot.")

    max_len = max(len(c) for c in chain_chars)

    fig_height = max(2.0, n_chains * 0.8)
    fig_width = max(10.0, max_len / 8.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # ------------------- tiles + residue letters -------------------
    for j, (label, chars, probs) in enumerate(zip(chain_labels, chain_chars, chain_probs)):
        L = len(chars)
        for i in range(L):
            x = i
            y = j
            pr = probs[i]

            ax.add_patch(
                plt.Rectangle(
                    (x, y), 1.0, 1.0,
                    linewidth=0.5,
                    edgecolor="black",
                    facecolor=plt.cm.viridis(min(max(pr, 0.0), vmax) / vmax),
                )
            )

            ax.text(
                x + 0.5,
                y + 0.5,
                chars[i],
                ha="center",
                va="center",
                fontsize=6,
                color="white",
            )

    # ------------------- residue numbers every 10 residues -------------------
    for j, chars in enumerate(chain_chars):
        L = len(chars)
        # indices 9, 19, 29, ... (0-based) -> residue numbers 10, 20, 30, ...
        for pos in range(9, L, 10):
            ax.text(
                pos + 0.5,        # x center of tile
                j - 0.25,         # a bit above the tile row
                str(pos + 1),     # 1-based residue index
                ha="center",
                va="bottom",
                fontsize=6,
                color="black",
                fontweight="bold",
            )

    # ------------------- axes & colorbar -------------------
    ax.set_xlim(0, max_len)
    ax.set_ylim(-0.7, n_chains)   # leave space on top for numbers
    ax.invert_yaxis()             # first chain at top

    ax.set_yticks(np.arange(n_chains) + 0.5)
    ax.set_yticklabels(chain_labels)
    ax.set_xticks([])

    ax.set_xlabel("Residue position (per chain)")
    ax.set_title("Paratope prediction (probability heatmap)")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Predicted probability (class 1)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[ok] Saved plot to {out_png}")


def main():
    ap = argparse.ArgumentParser(
        description="Make a colored grid plot of paratope predictions, "
                    "using seq NPZ (untokenized) and prediction CSV."
    )
    ap.add_argument("--seq-npz", required=True, help="Sequence NPZ file.")
    ap.add_argument("--pred-csv", required=True, help="Prediction CSV.")
    ap.add_argument(
        "--model-name",
        default="esm2_t6_8M_UR50D",
        help="ESM model name [default: esm2_t6_8M_UR50D].",
    )
    ap.add_argument(
        "--eoc-id",
        type=int,
        default=24,
        help="Token id for end-of-chain (EOC) [default: 24].",
    )
    ap.add_argument(
        "--key",
        help="Key (PDB id) to use from NPZ and CSV 'Key' column. "
             "If omitted, first NPZ key is used and CSV Key is not filtered.",
    )
    ap.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="SampleIdx in prediction CSV [default: 0].",
    )
    ap.add_argument(
        "--out",
        default="pred_grid.png",
        help="Output PNG filename [default: pred_grid.png].",
    )
    args = ap.parse_args()

    idx2tok, pad_idx = load_alphabet(args.model_name)
    tokens, used_key = load_sequence_tokens(args.seq_npz, key=args.key)
    chars = untokenize(tokens, idx2tok, pad_idx)

    print(f"[info] using key={used_key}, length={len(tokens)}")

    csv_key = used_key if args.key is not None else None
    probs_dict = load_probs(args.pred_csv, sample_idx=args.sample_idx, key=csv_key)
    print(f"[info] loaded predictions for {len(probs_dict)} positions")

    chain_labels, chain_chars, chain_probs = build_chain_views(
        tokens, chars, probs_dict, eoc_id=args.eoc_id, pad_idx=pad_idx
    )
    print("[info] chains:", list(zip(chain_labels, [len(c) for c in chain_chars])))

    plot_chain_grid(chain_labels, chain_chars, chain_probs, args.out)


if __name__ == "__main__":
    main()

