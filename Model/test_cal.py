import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np
from Dataloader_itf import SequenceParatopeDataset
from Models_newesm import ClassificationModel
from torch.amp import autocast
import csv
import os
import sys


# ───────────────────────────────────────── CONFIG / SETUP
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

inp_file = sys.argv[1]

# Optional threshold mode:
#   fixed            : use fixed threshold below
#   best_f1          : choose best-F1 threshold separately for each metric group on this eval set
#   antibody_antigen : choose one threshold for Antibody and one for Antigen, then apply to subchains
#THRESHOLD_MODE = "best_f1"
#FIXED_THRESHOLD = 0.5
THRESHOLD_MODE = "fixed"
FIXED_THRESHOLD = 0.38
with open(inp_file, "r") as f:
    models = [line.strip() for line in f if line.strip()]


test_config = {
    "batch_size": 1,
    "sequence_file": "para_test_esmsequences_1600.npz",
    "data_file": "para_test_esminterfaces_1600.npz",
    "edge_file": "para_test_esmedges_1600.npz",
    "max_len": 1600,
    "vocab_size": 31,
    "num_classes": 2,
}


def custom_collate_fn(batch):
    sequences, pt, edge = zip(*batch)

    sequence_tensor = torch.stack(sequences)
    pt_tensor = torch.tensor(np.array(pt), dtype=torch.long)

    max_edges = max(edge_index.shape[0] for edge_index in edge)
    padded_edges = []

    for edge_index in edge:
        edge_pad = -torch.ones((2, max_edges), dtype=torch.long)
        edge_pad[:, : edge_index.shape[0]] = edge_index.T.clone().detach()
        padded_edges.append(edge_pad)

    padded_edges = torch.stack(padded_edges)
    return padded_edges, sequence_tensor, pt_tensor


def squeeze_query_logits(outputs):
    if outputs.dim() == 4 and outputs.shape[1] == 1:
        outputs = outputs.squeeze(1)
    return outputs


def strip_checkpoint_keys(checkpoint):
    checkpoint.pop("n_averaged", None)

    clean = {}
    for k, v in checkpoint.items():
        while k.startswith("module."):
            k = k[len("module.") :]
        clean[k] = v
    return clean


def best_f1_threshold(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    mask = y_true >= 0
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.0, 0.0, 0.0

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0

    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.nanargmax(f1[:-1]))

    return (
        float(thresholds[best_idx]),
        float(precision[best_idx]),
        float(recall[best_idx]),
        float(f1[best_idx]),
    )


def safe_auc_roc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")

    return roc_auc_score(y_true, y_prob)


def safe_auc_pr(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall_vals, precision_vals)


def compute_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    valid = y_true >= 0
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    pred = (y_prob >= threshold).astype(int)

    accuracy = np.mean(y_true == pred) if len(y_true) else float("nan")
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    auc_roc = safe_auc_roc(y_true, y_prob)
    auc_pr = safe_auc_pr(y_true, y_prob)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "pred": pred,
    }


def choose_thresholds(metrics, mode="best_f1", fixed_threshold=0.5):
    thresholds = {}

    if mode == "fixed":
        for chain_type in metrics:
            thresholds[chain_type] = fixed_threshold
        return thresholds

    if mode == "best_f1":
        for chain_type, data in metrics.items():
            thr, _, _, _ = best_f1_threshold(data["true"], data["probs"])
            thresholds[chain_type] = thr
        return thresholds

    if mode == "antibody_antigen":
        ab_thr, _, _, _ = best_f1_threshold(metrics["Antibody"]["true"], metrics["Antibody"]["probs"])
        ag_thr, _, _, _ = best_f1_threshold(metrics["Antigen"]["true"], metrics["Antigen"]["probs"])

        for chain_type in metrics:
            if chain_type in ("Lchain", "Hchain", "Antibody"):
                thresholds[chain_type] = ab_thr
            else:
                thresholds[chain_type] = ag_thr

        return thresholds

    raise ValueError(f"Unknown threshold mode: {mode}")


def parse_model_config(model_file):
    base = os.path.basename(model_file)
    parts = base.split("_")

    num_layers = int(parts[2][1:])
    num_gnn_layers = int(parts[3][1:])
    num_int_layers = int(parts[4][1:])
    dropout = float(parts[5][2:])
    drop_path_rate = float(parts[6][3:])
    num_heads = int(parts[8][5:])

    return num_layers, num_gnn_layers, num_int_layers, dropout, drop_path_rate, num_heads


test_dataset = SequenceParatopeDataset(
    data_file=test_config["data_file"],
    sequence_file=test_config["sequence_file"],
    edge_file=test_config["edge_file"],
    max_len=test_config["max_len"],
)

test_loader = DataLoader(
    test_dataset,
    batch_size=test_config["batch_size"],
    collate_fn=custom_collate_fn,
    shuffle=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

interfaces_npz = np.load(test_config["data_file"], allow_pickle=True)
interface_keys = list(interfaces_npz.keys())


for model_file in models:
    torch.cuda.empty_cache()
    print(f"\nTesting model: {model_file}")
    print(f"Threshold mode: {THRESHOLD_MODE}")

    num_layers, num_gnn_layers, num_int_layers, dropout, drop_path_rate, num_heads = parse_model_config(model_file)

    model = ClassificationModel(
        vocab_size=test_config["vocab_size"],
        seq_len=test_config["max_len"],
        embed_dim=256,
        num_heads=num_heads,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        num_layers=num_layers,
        num_gnn_layers=num_gnn_layers,
        num_int_layers=num_int_layers,
        num_classes=test_config["num_classes"],
    )

    model = model.to(device)

    checkpoint = torch.load(model_file, map_location=device, weights_only=True)
    checkpoint = strip_checkpoint_keys(checkpoint)

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing:
        print("Missing keys :", missing)
    if unexpected:
        print("Unexpected  :", unexpected)

    model.eval()

    metrics = {
        "Lchain": {"true": [], "probs": [], "records": []},
        "Hchain": {"true": [], "probs": [], "records": []},
        "Antibody": {"true": [], "probs": [], "records": []},
        "Antigen": {"true": [], "probs": [], "records": []},
    }

    sample_idx = 0

    with torch.no_grad():
        for padded_edges, sequence_tensor, pt_tensor in test_loader:
            padded_edges = padded_edges.to(device)
            sequence_tensor = sequence_tensor.to(device)
            pt_tensor = pt_tensor.to(device)

            with autocast("cuda", enabled=(device.type == "cuda")):
                outputs, last_attention = model(
                    sequences=sequence_tensor,
                    padded_edges=padded_edges,
                    return_attention=True,
                )

            outputs = squeeze_query_logits(outputs)
            probs = torch.softmax(outputs, dim=-1)
            pos_probs = probs[..., 1]

            eoc_positions = np.where(sequence_tensor[0, 0, :].detach().cpu().numpy() == 24)[0]

            if len(eoc_positions) < 2:
                print(f"[Warning] sample {sample_idx}: fewer than 2 EOC positions. Skipping.")
                sample_idx += 1
                continue

            chain_types = ["Lchain", "Hchain"] + [f"AGchain_{i}" for i in range(len(eoc_positions) - 2)]
            chain_splits = [0] + eoc_positions.tolist()

            for chain_type in chain_types[2:]:
                if chain_type not in metrics:
                    metrics[chain_type] = {"true": [], "probs": [], "records": []}

            sample_key = interface_keys[sample_idx] if sample_idx < len(interface_keys) else f"sample_{sample_idx}"

            for i, chain_type in enumerate(chain_types):
                start, end = chain_splits[i], chain_splits[i + 1]

                mask = pt_tensor[0, start:end] >= 0
                if mask.sum().item() == 0:
                    print(f"No valid positions for {chain_type}. Skipping...")
                    continue

                true_labels = pt_tensor[0, start:end][mask].detach().cpu().numpy()
                probabilities = pos_probs[0, start:end][mask].detach().cpu().numpy()
                local_positions = np.arange(start, end)[mask.detach().cpu().numpy()]

                metrics[chain_type]["true"].extend(true_labels.tolist())
                metrics[chain_type]["probs"].extend(probabilities.tolist())

                for pos, t, pr in zip(local_positions, true_labels, probabilities):
                    metrics[chain_type]["records"].append({
                        "sample_key": sample_key,
                        "position": int(pos),
                        "chain_type": chain_type,
                        "true": int(t),
                        "prob": float(pr),
                    })

                if chain_type in ["Lchain", "Hchain"]:
                    metrics["Antibody"]["true"].extend(true_labels.tolist())
                    metrics["Antibody"]["probs"].extend(probabilities.tolist())
                    for pos, t, pr in zip(local_positions, true_labels, probabilities):
                        metrics["Antibody"]["records"].append({
                            "sample_key": sample_key,
                            "position": int(pos),
                            "chain_type": "Antibody",
                            "true": int(t),
                            "prob": float(pr),
                        })
                else:
                    metrics["Antigen"]["true"].extend(true_labels.tolist())
                    metrics["Antigen"]["probs"].extend(probabilities.tolist())
                    for pos, t, pr in zip(local_positions, true_labels, probabilities):
                        metrics["Antigen"]["records"].append({
                            "sample_key": sample_key,
                            "position": int(pos),
                            "chain_type": "Antigen",
                            "true": int(t),
                            "prob": float(pr),
                        })

            sample_idx += 1

    thresholds = choose_thresholds(
        metrics,
        mode=THRESHOLD_MODE,
        fixed_threshold=FIXED_THRESHOLD,
    )

    pred_csv = f"{model_file}_predictions_threshold_{THRESHOLD_MODE}.csv"
    with open(pred_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sample",
            "Position",
            "Chain Type",
            "True Label",
            "Predicted Label",
            "Probability",
            "Threshold",
        ])

        for chain_type, data in metrics.items():
            if chain_type in ("Antibody", "Antigen"):
                continue

            thr = thresholds[chain_type]
            for rec in data["records"]:
                pred = int(rec["prob"] >= thr)
                writer.writerow([
                    rec["sample_key"],
                    rec["position"],
                    chain_type,
                    rec["true"],
                    pred,
                    rec["prob"],
                    thr,
                ])

    print(f"Saved predictions: {pred_csv}")

    for chain_type, data in metrics.items():
        true = data["true"]
        probs_chain = data["probs"]

        if len(true) == 0:
            print(f"No data available for {chain_type}")
            continue

        thr = thresholds[chain_type]
        res = compute_metrics(true, probs_chain, thr)
        best_thr, best_p, best_r, best_f1 = best_f1_threshold(true, probs_chain)

        print(f"\nMetrics for {chain_type}:")
        print(f"  Threshold used: {thr:.4f}")
        print(f"  Accuracy: {res['accuracy'] * 100:.2f}%")
        print(f"  Precision: {res['precision']:.4f}")
        print(f"  Recall: {res['recall']:.4f}")
        print(f"  F1: {res['f1']:.4f}")
        print(f"  AUC-ROC: {res['auc_roc']:.4f}")
        print(f"  AUC-PR: {res['auc_pr']:.4f}")
        print(f"  Best-F1 threshold on this set: {best_thr:.4f}")
        print(f"  Best-F1 Precision/Recall/F1: {best_p:.4f} / {best_r:.4f} / {best_f1:.4f}")

