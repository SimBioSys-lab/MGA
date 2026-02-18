#!/usr/bin/env python3
import os
import sys
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast

from Dataloader_pred import SequenceParatopeDataset  # prediction-only version (seq + edge + key)
from Models_fullnew import ClassificationModel


# ------------------------------------------------------------- CONFIG
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# Fixed model path
MODEL_FILE = "/projects/SimBioSys/Xing/itf_pred/data_preprocess/isicisiParamodelnewesm_l1_g10_i5_do0.30_dpr0.25_lr0.0001_fold10.pth"

# EOC token id (from ESM vocabulary)
EOC_ID = 24

# ------------------------------------------------------------- ARGS
if len(sys.argv) != 3:
    print("Usage:")
    print("  python predict_paratope_msa.py <sequence_npz> <edge_npz>")
    sys.exit(1)

sequence_file = sys.argv[1]
edge_file = sys.argv[2]

if not os.path.exists(sequence_file):
    raise FileNotFoundError(sequence_file)
if not os.path.exists(edge_file):
    raise FileNotFoundError(edge_file)

print(f"Sequence file: {sequence_file}")
print(f"Edge file    : {edge_file}")
print(f"Model        : {MODEL_FILE}")

# ------------------------------------------------------------- TEST CONFIG
test_config = {
    "batch_size": 1,
    "max_len": 1600,
    "vocab_size": 31,
    "num_classes": 2,
}

# -------------------------------------------------------- collate function
def custom_collate_fn(batch):
    """
    batch: list of (sequence_tensor, edges_tensor, key)
        sequence_tensor: (64, L)  # MSA depth x length
        edges_tensor   : (E, 2)
        key            : str

    Returns:
        padded_edges : (B, 2, Emax)
        sequences    : (B, 64, L)
        keys         : list[str]
    """
    sequences, edges, keys = zip(*batch)

    sequence_tensor = torch.stack(sequences)  # (B, 64, L)

    max_edges = max(edge.shape[0] for edge in edges)
    padded_edges = []
    for edge_index in edges:
        edge_pad = -torch.ones((2, max_edges), dtype=torch.long)
        edge_pad[:, :edge_index.shape[0]] = edge_index.T.clone().detach()
        padded_edges.append(edge_pad)

    padded_edges = torch.stack(padded_edges)  # (B, 2, Emax)

    return padded_edges, sequence_tensor, list(keys)


# -------------------------------------------------------- dataset & dataloader
test_dataset = SequenceParatopeDataset(
    sequence_file=sequence_file,
    edge_file=edge_file,
    max_len=test_config["max_len"],
)

test_loader = DataLoader(
    test_dataset,
    batch_size=test_config["batch_size"],
    collate_fn=custom_collate_fn,
    shuffle=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")
use_amp = (device.type == "cuda")

# -------------------------------------------------------- build model
parts = os.path.basename(MODEL_FILE).split("_")
# isicisiParamodelnewesm_l1_g10_i5_do0.30_dpr0.25_lr0.0001_fold10.pth
num_layers     = int(parts[1][1:])       # l1  -> 1
num_gnn_layers = int(parts[2][1:])       # g10 -> 10
num_int_layers = int(parts[3][1:])       # i5  -> 5
dropout        = float(parts[4][2:])     # do0.30 -> 0.30
drop_path_rate = float(parts[5][3:])     # dpr0.25 -> 0.25

# Use your usual settings
num_heads = 16
embed_dim = 256

print("\nModel hyperparameters:")
print(f"  num_layers     = {num_layers}")
print(f"  num_gnn_layers = {num_gnn_layers}")
print(f"  num_int_layers = {num_int_layers}")
print(f"  dropout        = {dropout}")
print(f"  drop_path_rate = {drop_path_rate}")
print(f"  num_heads      = {num_heads}")
print(f"  embed_dim      = {embed_dim}")

model = ClassificationModel(
    vocab_size=test_config["vocab_size"],
    seq_len=test_config["max_len"],
    embed_dim=embed_dim,
    num_heads=num_heads,
    dropout=dropout,
    drop_path_rate=drop_path_rate,
    num_layers=num_layers,
    num_gnn_layers=num_gnn_layers,
    num_int_layers=num_int_layers,
    num_classes=test_config["num_classes"],
).to(device)

print(f"\nLoading checkpoint: {MODEL_FILE}")
checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=True)
checkpoint.pop("n_averaged", None)
checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}

missing, unexpected = model.load_state_dict(checkpoint, strict=False)
if missing:
    print("Missing keys:", missing)
if unexpected:
    print("Unexpected keys:", unexpected)

model.eval()

# -------------------------------------------------------- prediction loop
out_csv = os.path.basename(sequence_file).replace(".npz", "_msa_pred.csv")
print(f"\nSaving predictions to: {out_csv}")

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["SampleIdx", "Key", "ChainType", "Position", "PredLabel", "Probability"])

    sample_idx = 0
    with torch.no_grad():
        for padded_edges, sequence_tensor, keys in test_loader:
            padded_edges = padded_edges.to(device)        # (B, 2, Emax)
            sequence_tensor = sequence_tensor.to(device)  # (B, 64, L)
            key = keys[0]                                 # batch_size = 1

            # Forward pass
            if use_amp:
                with autocast("cuda"):
                    outputs, _ = model(
                        sequences=sequence_tensor,
                        padded_edges=padded_edges,
                        return_attention=True,
                    )
            else:
                outputs, _ = model(
                    sequences=sequence_tensor,
                    padded_edges=padded_edges,
                    return_attention=True,
                )

            # outputs shape depends on your model; earlier you did squeeze(1)
            # so we keep that convention:
            outputs = outputs.squeeze(1)            # (B, L, 2)
            probs   = torch.softmax(outputs, dim=-1)
            preds   = torch.argmax(probs, dim=-1)   # (B, L)

            # Use the first row of the MSA (query) to find EOCs
            seq_np = sequence_tensor[0, 0, :].detach().cpu().numpy()
            EOCs = np.where(seq_np == EOC_ID)[0]

            if len(EOCs) < 2:
                print(f"[warn] sample {sample_idx}, key={key}: only {len(EOCs)} EOCs found.")

            # Chain segments: [L, H, AG1, AG2, ...]
            chain_types  = ["Lchain", "Hchain"] + [f"AGchain_{i}" for i in range(len(EOCs) - 2)]
            chain_splits = [0] + EOCs.tolist()

            for i, chain in enumerate(chain_types):
                if i + 1 >= len(chain_splits):
                    break

                start, end = chain_splits[i], chain_splits[i + 1]
                if end <= start:
                    continue

                chain_preds = preds[0, start:end].cpu().numpy()
                chain_probs = probs[0, start:end, 1].cpu().numpy()
                positions   = np.arange(start, end)

                for pos, p, pr in zip(positions, chain_preds, chain_probs):
                    writer.writerow([
                        sample_idx,
                        key,
                        chain,
                        int(pos),
                        int(p),
                        float(pr),
                    ])

            sample_idx += 1

print("\n Prediction finished successfully.")

