import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
import numpy as np
from Dataloader_itf import SequenceParatopeDataset
from Models_fullnew_prenorm import ClassificationModel
from torch.amp import autocast
import csv
import os
import sys
# Configuration
torch.backends.cudnn.benchmark = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
inp_file = sys.argv[1]
# Read model filenames
with open(inp_file, 'r') as f:
    models = [line.strip() for line in f if line.strip()]

# Test configuration
test_config = {
    'batch_size': 1,
    'sequence_file': 'para_test_esmsequences_1600.npz',
    'data_file': 'para_test_esminterfaces_1600.npz',
    'edge_file': 'para_test_esmedges_1600.npz',
    'max_len': 1600,
    'vocab_size': 31,
    'num_classes': 2
}
# Custom collate function
def custom_collate_fn(batch):
    sequences, pt, edge = zip(*batch)
    sequence_tensor = torch.stack(sequences)
    pt_tensor = torch.tensor(np.array(pt), dtype=torch.long)
    max_edges = max(edge_index.shape[0] for edge_index in edge)
    padded_edges = []
    for edge_index in edge:
        edge_pad = -torch.ones((2, max_edges), dtype=torch.long)
        edge_pad[:, :edge_index.shape[0]] = edge_index.T.clone().detach()
        padded_edges.append(edge_pad)
    padded_edges = torch.stack(padded_edges)
    return padded_edges, sequence_tensor, pt_tensor

# Load the dataset
test_dataset = SequenceParatopeDataset(
    data_file=test_config['data_file'],
    sequence_file=test_config['sequence_file'],
    edge_file=test_config['edge_file'],
    max_len=test_config['max_len']
)
test_loader = DataLoader(test_dataset, batch_size=test_config['batch_size'],
                           collate_fn=custom_collate_fn, shuffle=False)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the interfaces NPZ file and get its keys.
# (We assume that the keys in this file correspond one-to-one with the samples in test_loader.)
interfaces_npz = np.load(test_config['data_file'], allow_pickle=True)
interface_keys = list(interfaces_npz.keys())

# Test each model
for model_file in models:
    torch.cuda.empty_cache()
    print(f"\nTesting model: {model_file}")
    # Parse model configuration from filename
    parts = model_file.split('_')
    num_layers = int(parts[1][1:])
    num_gnn_layers = int(parts[2][1:])
    num_int_layers = int(parts[3][1:])
    embed_dim = 256
    num_heads = int(parts[7][5:])
    dropout = float(parts[4][2:])
    drop_path_rate = float(parts[5][3:])
    # Initialize model
    model = ClassificationModel(
        vocab_size=test_config['vocab_size'],
        seq_len=test_config['max_len'],
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        num_layers=num_layers,
        num_gnn_layers=num_gnn_layers,
        num_int_layers=num_int_layers,
        num_classes=test_config['num_classes']
    )
    model = model.to(device)
#    # Load model weights
#    checkpoint = torch.load(model_file, map_location=device)
#    if 'module.' in list(checkpoint.keys())[0]:
#        checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
#    model.load_state_dict(checkpoint)
    # ─── Load checkpoint and normalise keys ───────────────────────────
    checkpoint = torch.load(model_file, map_location=device,weights_only=True)
    
    # 1) remove the SWA buffer that isn’t a real parameter
    checkpoint.pop("n_averaged", None)
    
    # 2) strip leading "module." (DataParallel / DDP wrapper)
    checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
    checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
    
    # 3) load into model
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    
    if missing:
        print("Missing keys :", missing)
    if unexpected:
        print("Unexpected  :", unexpected)
    model.eval()

    # Initialize metrics for each chain and the combined antibody and antigen
    metrics = {
        "Lchain": {"true": [], "pred": [], "probs": []},
        "Hchain": {"true": [], "pred": [], "probs": []},
        "Antibody": {"true": [], "pred": [], "probs": []},
        "Antigen": {"true": [], "pred": [], "probs": []}
    }

    # Prepare to save predictions to CSV
    with open(f"{model_file}_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Chain Type", "True Label", "Predicted Label", "Probability"])

        # This dictionary will store the last_attention matrices using keys from the interfaces NPZ.
        last_attention_dict = {}
        sample_idx = 0  # To index into interface_keys

        with torch.no_grad():
            for padded_edges, sequence_tensor, pt_tensor in test_loader:
                # Move tensors to device
                padded_edges = padded_edges.to(device)
                sequence_tensor = sequence_tensor.to(device)
                pt_tensor = pt_tensor.to(device)

                # Perform inference and get last_attention from the model.
                with autocast("cuda"):
                    outputs, last_attention = model(sequences=sequence_tensor, padded_edges=padded_edges,
                                                    return_attention=True)
#                print('attn_shape:', last_attention.shape)
                outputs = outputs.squeeze(1)
                probs = torch.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)
#                print('probshape', probs.shape)
#                print('predshape', preds.shape)
#                pos_probs = probs[:, 1]                # probability for class 1
#                preds = (pos_probs > 0.4).long()         # 1 if pos_prob > 0.4, else 0

#                # Store the last_attention matrix for this sample
#                # (batch_size==1 so we use sample_idx from interface_keys)
#                if sample_idx < len(interface_keys):
#                    last_attention_dict[interface_keys[sample_idx]] = last_attention.cpu().numpy()
#                else:
#                    last_attention_dict[f"sample_{sample_idx}"] = last_attention.cpu().numpy()
#                sample_idx += 1

                # Extract EOCs from the sequence (assumes channel 0 holds the sequence integers)
                EOCs = np.where(sequence_tensor[0, 0, :].cpu().numpy() == 24)[0]

                # Define chain types and splits
                chain_types = ["Lchain", "Hchain"] + [f"AGchain_{i}" for i in range(len(EOCs) - 2)]
                chain_splits = [0] + EOCs.tolist()
                # Ensure metrics are initialized for AGchains if not already present
                for chain_type in chain_types[2:]:
                    if chain_type not in metrics:
                        metrics[chain_type] = {"true": [], "pred": [], "probs": []}

                # Process metrics for each chain type
                for i, chain_type in enumerate(chain_types):
                    start, end = chain_splits[i], chain_splits[i + 1]
                    mask = pt_tensor[0, start:end] >= 0

                    if mask.sum().item() == 0:
                        print(f"No valid positions for {chain_type}. Skipping...")
                        continue
                    true_labels = pt_tensor[0, start:end][mask].cpu().numpy()
                    predicted_labels = preds[0, start:end][mask].cpu().numpy()
                    probabilities = probs[0, start:end, 1][mask].cpu().numpy()

                    # Log metrics for this chain
                    metrics[chain_type]["true"].extend(true_labels)
                    metrics[chain_type]["pred"].extend(predicted_labels)
                    metrics[chain_type]["probs"].extend(probabilities)

                    # Combine antibody (Lchain, Hchain) and antigen metrics
                    if chain_type in ["Lchain", "Hchain"]:
                        metrics["Antibody"]["true"].extend(true_labels)
                        metrics["Antibody"]["pred"].extend(predicted_labels)
                        metrics["Antibody"]["probs"].extend(probabilities)
                    else:
                        metrics["Antigen"]["true"].extend(true_labels)
                        metrics["Antigen"]["pred"].extend(predicted_labels)
                        metrics["Antigen"]["probs"].extend(probabilities)

                    # Write predictions for this chain to CSV
                    for t, p, pr in zip(true_labels, predicted_labels, probabilities):
                        writer.writerow([chain_type, t, p, pr])

    # After processing all batches, calculate and print final metrics
    for chain_type, data in metrics.items():
        true = data["true"]
        pred = data["pred"]
        probs = data["probs"]

        if len(true) == 0:
            print(f"No data available for {chain_type}")
            continue

        accuracy = np.mean(np.array(true) == np.array(pred))
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        auc_roc = roc_auc_score(true, probs)
        precision_vals, recall_vals, _ = precision_recall_curve(true, probs)
        auc_pr = auc(recall_vals, precision_vals)

        print(f"\nMetrics for {chain_type}:")
        print(f"  Accuracy: {accuracy * 100:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  AUC-ROC: {auc_roc:.4f}")
        print(f"  AUC-PR: {auc_pr:.4f}")

#    # Save the last_attention matrices to an NPZ file.
#    np.savez(f"{model_file}.npz", **last_attention_dict)
#    print(f"Saved last_attention matrices to {model_file}.npz")

