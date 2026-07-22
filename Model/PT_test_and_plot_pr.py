import argparse
import csv
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
)
from torch.amp import autocast
from torch.utils.data import DataLoader

from Dataloader_itf import SequenceParatopeDataset
from Models_fullnew import ClassificationModel


# =============================================================================
# Plot styling
# =============================================================================
def set_plot_style(scale: float = 1.9):
    base = 10.0

    plt.rcParams.update(
        {
            "font.size": base * scale,
            "axes.titlesize": base * scale * 1.02,
            "axes.labelsize": base * scale,
            "xtick.labelsize": base * scale * 0.82,
            "ytick.labelsize": base * scale * 0.82,
            "legend.fontsize": base * scale * 0.55,
            "lines.linewidth": 2.4,
            "axes.linewidth": 1.2,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "savefig.dpi": 400,
            "figure.dpi": 140,
        }
    )


def style_axes(ax):
    ax.grid(
        True,
        linestyle="--",
        alpha=0.22,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(
    ax,
    label: str,
    x: float = -0.20,
    y: float = 1.02,
    fontsize: int = 22,
):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
    )


# =============================================================================
# Input utilities
# =============================================================================
def read_checkpoint_list(list_file: str) -> List[str]:
    checkpoints = []

    with open(list_file, "r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()

            if not value:
                continue

            if value.startswith("#"):
                continue

            checkpoints.append(value)

    if not checkpoints:
        raise ValueError(
            f"No checkpoint paths were found in {list_file}"
        )

    return checkpoints


def safe_filename_stem(path: str) -> str:
    basename = os.path.basename(path)

    for suffix in (".pth", ".pt", ".ckpt"):
        if basename.endswith(suffix):
            basename = basename[: -len(suffix)]
            break

    return basename


# =============================================================================
# Model configuration parsing
# =============================================================================
def parse_model_config_from_filename(
    checkpoint_path: str,
    default_embed_dim: int = 256,
) -> Dict:
    """
    Expected checkpoint filename components resemble:

        model_l1_g10_i5_do0.10_dpr0.10_lr0.0001_heads16_fold1.pth

    The parser searches the basename and does not rely on fixed underscore
    positions.
    """
    basename = os.path.basename(checkpoint_path)

    patterns = {
        "num_layers": r"(?:^|_)l(\d+)(?:_|$)",
        "num_gnn_layers": r"(?:^|_)g(\d+)(?:_|$)",
        "num_int_layers": r"(?:^|_)i(\d+)(?:_|$)",
        "dropout": r"(?:^|_)do([0-9]*\.?[0-9]+)(?:_|$)",
        "drop_path_rate": r"(?:^|_)dpr([0-9]*\.?[0-9]+)(?:_|$)",
        "num_heads": r"(?:^|_)heads(\d+)(?:_|$)",
    }

    parsed = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, basename)

        if match is None:
            raise ValueError(
                f"Could not parse '{key}' from checkpoint filename:\n"
                f"  {basename}\n\n"
                "Expected filename components such as:\n"
                "  _l1_g10_i5_do0.10_dpr0.10_heads16_\n\n"
                "You can instead use --manual_model_config and provide "
                "the architecture arguments explicitly."
            )

        parsed[key] = match.group(1)

    return {
        "num_layers": int(parsed["num_layers"]),
        "num_gnn_layers": int(parsed["num_gnn_layers"]),
        "num_int_layers": int(parsed["num_int_layers"]),
        "dropout": float(parsed["dropout"]),
        "drop_path_rate": float(parsed["drop_path_rate"]),
        "num_heads": int(parsed["num_heads"]),
        "embed_dim": default_embed_dim,
    }


def get_model_config(
    checkpoint_path: str,
    args,
) -> Dict:
    if args.manual_model_config:
        return {
            "num_layers": args.num_layers,
            "num_gnn_layers": args.num_gnn_layers,
            "num_int_layers": args.num_int_layers,
            "dropout": args.dropout,
            "drop_path_rate": args.drop_path_rate,
            "num_heads": args.num_heads,
            "embed_dim": args.embed_dim,
        }

    return parse_model_config_from_filename(
        checkpoint_path,
        default_embed_dim=args.embed_dim,
    )


# =============================================================================
# Checkpoint loading
# =============================================================================
def extract_state_dict(checkpoint) -> Dict:
    """
    Supports:
      - raw state dictionaries
      - model_state_dict containers
      - state_dict containers
      - model containers
      - SWA checkpoints
      - repeated module.module prefixes
    """
    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model_state",
            "model",
        ):
            if (
                key in checkpoint
                and isinstance(checkpoint[key], dict)
            ):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The checkpoint does not contain a valid model state dictionary."
        )

    checkpoint.pop("n_averaged", None)

    normalized = {}

    for key, value in checkpoint.items():
        while key.startswith("module."):
            key = key[len("module."):]

        normalized[key] = value

    return normalized


def load_model(
    checkpoint_path: str,
    model_config: Dict,
    test_config: Dict,
    device: torch.device,
) -> ClassificationModel:
    print("Model configuration:")

    for key, value in model_config.items():
        print(f"  {key}: {value}")

    model = ClassificationModel(
        vocab_size=test_config["vocab_size"],
        seq_len=test_config["max_len"],
        embed_dim=model_config["embed_dim"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
        drop_path_rate=model_config["drop_path_rate"],
        num_layers=model_config["num_layers"],
        num_gnn_layers=model_config["num_gnn_layers"],
        num_int_layers=model_config["num_int_layers"],
        num_classes=test_config["num_classes"],
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    state_dict = extract_state_dict(checkpoint)

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing:
        print(f"Missing keys ({len(missing)}):")
        print(missing[:20])

    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}):")
        print(unexpected[:20])

    model = model.to(device)
    model.eval()

    return model


# =============================================================================
# Dataset and collate
# =============================================================================
def custom_collate_fn(batch):
    sequences, labels, edges = zip(*batch)

    sequence_tensor = torch.stack(sequences)

    label_tensor = torch.as_tensor(
        np.array(labels),
        dtype=torch.long,
    )

    max_edges = max(
        edge_index.shape[0]
        for edge_index in edges
    )

    padded_edges = []

    for edge_index in edges:
        if torch.is_tensor(edge_index):
            edge_tensor = (
                edge_index.T
                .clone()
                .detach()
                .long()
            )
        else:
            edge_tensor = torch.as_tensor(
                np.asarray(edge_index).T,
                dtype=torch.long,
            )

        edge_pad = -torch.ones(
            (2, max_edges),
            dtype=torch.long,
        )

        edge_pad[:, :edge_tensor.shape[1]] = edge_tensor
        padded_edges.append(edge_pad)

    padded_edges = torch.stack(padded_edges)

    return (
        padded_edges,
        sequence_tensor,
        label_tensor,
    )


# =============================================================================
# Chain handling
# =============================================================================
def get_chain_boundaries(
    sequence_tokens: np.ndarray,
    eoc_token: int = 24,
) -> List[Dict]:
    """
    Follows the same chain splitting behavior as the original PT_test.py.

    Chain order:
        Lchain
        Hchain
        AGchain_0
        AGchain_1
        ...

    EOC positions are treated as exclusive end indices, so the EOC token itself
    is not included in the saved residue predictions.
    """
    eoc_positions = np.where(
        sequence_tokens == eoc_token
    )[0]

    if len(eoc_positions) < 2:
        raise ValueError(
            "Fewer than two EOC tokens were found. "
            "Cannot identify light and heavy chains."
        )

    chain_names = (
        ["Lchain", "Hchain"]
        + [
            f"AGchain_{index}"
            for index in range(len(eoc_positions) - 2)
        ]
    )

    chain_splits = [0] + eoc_positions.tolist()

    boundaries = []

    for chain_index, chain_name in enumerate(chain_names):
        if chain_index + 1 >= len(chain_splits):
            break

        boundaries.append(
            {
                "chain_type": chain_name,
                "chain_index": chain_index,
                "start": chain_splits[chain_index],
                "end": chain_splits[chain_index + 1],
            }
        )

    return boundaries


# =============================================================================
# Prediction
# =============================================================================
def run_checkpoint_inference(
    model: ClassificationModel,
    checkpoint_path: str,
    step_number: int,
    test_loader: DataLoader,
    interface_keys: List[str],
    device: torch.device,
    output_csv: str,
    num_classes: int,
):
    fieldnames = [
        "Step",
        "Model",
        "Sample Index",
        "Sample ID",
        "Chain Type",
        "Chain Index",
        "Global Residue Index",
        "Chain Residue Index",
        "Chain Residue Number",
        "True Label",
        "Predicted Label",
        "Probability",
    ]

    number_saved = 0

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        with torch.no_grad():
            for sample_index, (
                padded_edges,
                sequence_tensor,
                label_tensor,
            ) in enumerate(test_loader):

                padded_edges = padded_edges.to(
                    device,
                    non_blocking=True,
                )

                sequence_tensor = sequence_tensor.to(
                    device,
                    non_blocking=True,
                )

                label_tensor = label_tensor.to(
                    device,
                    non_blocking=True,
                )

                if sample_index < len(interface_keys):
                    sample_id = str(interface_keys[sample_index])
                else:
                    sample_id = f"sample_{sample_index}"

                with autocast(
                    device_type="cuda",
                    enabled=torch.cuda.is_available(),
                ):
                    outputs, _ = model(
                        sequences=sequence_tensor,
                        padded_edges=padded_edges,
                        return_attention=True,
                    )

                # Expected:
                #   [B, 1, L, C] -> squeeze dimension 1
                # or
                #   [B, L, C]
                if (
                    outputs.ndim == 4
                    and outputs.shape[1] == 1
                ):
                    outputs = outputs.squeeze(1)

                if outputs.ndim != 3:
                    raise ValueError(
                        "Unexpected model output shape. Expected "
                        "[B, L, C] or [B, 1, L, C], but received "
                        f"{tuple(outputs.shape)}"
                    )

                if outputs.shape[-1] != num_classes:
                    raise ValueError(
                        f"Expected {num_classes} output classes, "
                        f"but output shape is {tuple(outputs.shape)}"
                    )

                probabilities = torch.softmax(
                    outputs.float(),
                    dim=-1,
                )

                positive_probabilities = probabilities[..., 1]

                predicted_labels = torch.argmax(
                    probabilities,
                    dim=-1,
                )

                sequence_tokens = (
                    sequence_tensor[0, 0, :]
                    .detach()
                    .cpu()
                    .numpy()
                )

                boundaries = get_chain_boundaries(
                    sequence_tokens
                )

                for boundary in boundaries:
                    chain_type = boundary["chain_type"]
                    chain_index = boundary["chain_index"]
                    start = boundary["start"]
                    end = boundary["end"]

                    chain_labels = label_tensor[
                        0,
                        start:end,
                    ]

                    valid_mask = chain_labels >= 0

                    valid_relative_indices = (
                        torch.nonzero(
                            valid_mask,
                            as_tuple=False,
                        )
                        .flatten()
                    )

                    if valid_relative_indices.numel() == 0:
                        continue

                    true_labels = (
                        chain_labels[valid_mask]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    pred_labels = (
                        predicted_labels[
                            0,
                            start:end,
                        ][valid_mask]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    pos_probs = (
                        positive_probabilities[
                            0,
                            start:end,
                        ][valid_mask]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    relative_indices = (
                        valid_relative_indices
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    global_indices = relative_indices + start

                    for (
                        relative_index,
                        global_index,
                        true_label,
                        predicted_label,
                        probability,
                    ) in zip(
                        relative_indices,
                        global_indices,
                        true_labels,
                        pred_labels,
                        pos_probs,
                    ):
                        writer.writerow(
                            {
                                "Step": step_number,
                                "Model": checkpoint_path,
                                "Sample Index": sample_index,
                                "Sample ID": sample_id,
                                "Chain Type": chain_type,
                                "Chain Index": chain_index,
                                "Global Residue Index": int(
                                    global_index
                                ),
                                "Chain Residue Index": int(
                                    relative_index
                                ),
                                "Chain Residue Number": int(
                                    relative_index
                                ) + 1,
                                "True Label": int(true_label),
                                "Predicted Label": int(
                                    predicted_label
                                ),
                                "Probability": float(probability),
                            }
                        )

                        number_saved += 1

    print(
        f"Saved {number_saved} residue predictions to:\n"
        f"  {output_csv}"
    )


# =============================================================================
# PR/AUPR computation
# =============================================================================
def compute_pr_metrics(
    y_true,
    y_score,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    y_true = pd.Series(y_true).astype(int).values
    y_score = pd.Series(y_score).astype(float).values

    unique_classes = np.unique(y_true)

    if len(unique_classes) < 2:
        raise ValueError(
            "Only one class is present in y_true; "
            "PR/AUPR is not defined."
        )

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_score,
    )

    aupr_trapz = auc(
        recall,
        precision,
    )

    average_precision = average_precision_score(
        y_true,
        y_score,
    )

    return (
        precision,
        recall,
        aupr_trapz,
        average_precision,
    )


def subset_rows(
    dataframe: pd.DataFrame,
    subset_name: str,
) -> pd.DataFrame:
    chain_column = (
        dataframe["Chain Type"]
        .astype(str)
        .str.strip()
    )

    if subset_name == "overall":
        return dataframe.copy()

    if subset_name == "paratope":
        mask = chain_column.str.startswith(
            ("Lchain", "Hchain")
        )

        return dataframe.loc[mask].copy()

    if subset_name == "epitope":
        mask = chain_column.str.startswith(
            "AGchain"
        )

        return dataframe.loc[mask].copy()

    raise ValueError(
        f"Unknown subset name: {subset_name}"
    )


def evaluate_dataframe_subset(
    dataframe: pd.DataFrame,
    subset_name: str,
    step_number: int,
    checkpoint_path: str,
) -> Dict:
    subset = subset_rows(
        dataframe,
        subset_name,
    )

    if len(subset) == 0:
        raise ValueError(
            f"No rows were found for subset '{subset_name}'"
        )

    y_true = subset["True Label"]
    y_score = subset["Probability"]

    (
        precision,
        recall,
        aupr_trapz,
        average_precision,
    ) = compute_pr_metrics(
        y_true,
        y_score,
    )

    positive_fraction = float(
        pd.Series(y_true).mean()
    )

    return {
        "file": checkpoint_path,
        "step": step_number,
        "label": f"Step {step_number}",
        "subset": subset_name,
        "n_samples": len(subset),
        "positive_fraction": positive_fraction,
        "AUPR": aupr_trapz,
        "AveragePrecision": average_precision,
        "precision": precision,
        "recall": recall,
    }


def collect_results(
    prediction_records: List[Dict],
    subset_name: str,
) -> List[Dict]:
    results = []

    for record in prediction_records:
        prediction_csv = record["prediction_csv"]
        step_number = record["step"]
        checkpoint_path = record["checkpoint"]

        dataframe = pd.read_csv(
            prediction_csv
        )

        dataframe.columns = [
            column.strip()
            for column in dataframe.columns
        ]

        required_columns = [
            "Chain Type",
            "True Label",
            "Probability",
        ]

        missing = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"{prediction_csv} is missing columns: {missing}"
            )

        result = evaluate_dataframe_subset(
            dataframe=dataframe,
            subset_name=subset_name,
            step_number=step_number,
            checkpoint_path=checkpoint_path,
        )

        results.append(result)

        print(
            f"[{subset_name}] Step {step_number}: "
            f"AUPR={result['AUPR']:.6f}, "
            f"AP={result['AveragePrecision']:.6f}, "
            f"positive_fraction="
            f"{result['positive_fraction']:.6f}, "
            f"n={result['n_samples']}"
        )

    return sorted(
        results,
        key=lambda item: item["step"],
    )


# =============================================================================
# Plotting
# =============================================================================
def get_colors(number: int):
    palette = [
        "0.55",
        "tab:blue",
        "tab:orange",
        "tab:red",
        "tab:green",
        "tab:purple",
        "tab:brown",
        "tab:pink",
    ]

    if number <= len(palette):
        return palette[:number]

    return [
        f"C{index}"
        for index in range(number)
    ]


def get_step1_aupr(
    results: List[Dict],
) -> float:
    sorted_results = sorted(
        results,
        key=lambda item: item["step"],
    )

    for result in sorted_results:
        if result["step"] == 1:
            return result["AUPR"]

    return sorted_results[0]["AUPR"]


def plot_pr_full(
    ax,
    results: List[Dict],
    title: str,
    legend_loc: str = "lower left",
):
    colors = get_colors(
        len(results)
    )

    baseline = float(
        np.mean(
            [
                result["positive_fraction"]
                for result in results
            ]
        )
    )

    step1_aupr = get_step1_aupr(
        results
    )

    best_step = max(
        results,
        key=lambda item: item["AUPR"],
    )["step"]

    ax.axhline(
        y=baseline,
        linestyle="--",
        linewidth=1.4,
        color="black",
        alpha=0.75,
        label=f"Baseline (≈ {baseline:.3f})",
        zorder=1,
    )

    for color, result in zip(
        colors,
        results,
    ):
        gain = (
            result["AUPR"]
            - step1_aupr
        )

        if result["step"] == 1:
            legend = (
                f"{result['label']} "
                f"({result['AUPR']:.3f})"
            )
        else:
            legend = (
                f"{result['label']} "
                f"({result['AUPR']:.3f}, "
                f"{gain:+.3f})"
            )

        is_best = (
            result["step"]
            == best_step
        )

        ax.plot(
            result["recall"],
            result["precision"],
            color=color,
            linewidth=2.8 if is_best else 2.0,
            alpha=1.0 if is_best else 0.9,
            label=legend,
        )

    ax.set_title(
        title,
        pad=8,
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    style_axes(ax)

    ax.legend(
        loc=legend_loc,
        frameon=False,
        handlelength=1.8,
        labelspacing=0.25,
        borderpad=0.2,
    )


def plot_pr_zoom(
    ax,
    results: List[Dict],
    title: str,
    zoom_xlim: float = 0.6,
    zoom_ylim_min: float = 0.7,
    legend_loc: str = "upper right",
):
    colors = get_colors(
        len(results)
    )

    step1_aupr = get_step1_aupr(
        results
    )

    best_step = max(
        results,
        key=lambda item: item["AUPR"],
    )["step"]

    for color, result in zip(
        colors,
        results,
    ):
        gain = (
            result["AUPR"]
            - step1_aupr
        )

        if result["step"] == 1:
            legend = (
                f"{result['label']} "
                f"({result['AUPR']:.3f})"
            )
        else:
            legend = (
                f"{result['label']} "
                f"({result['AUPR']:.3f}, "
                f"{gain:+.3f})"
            )

        is_best = (
            result["step"]
            == best_step
        )

        ax.plot(
            result["recall"],
            result["precision"],
            color=color,
            linewidth=2.8 if is_best else 2.0,
            alpha=1.0 if is_best else 0.9,
            label=legend,
        )

    ax.set_title(
        title,
        pad=8,
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, zoom_xlim)
    ax.set_ylim(
        zoom_ylim_min,
        1.01,
    )

    style_axes(ax)

    ax.legend(
        loc=legend_loc,
        frameon=False,
        handlelength=1.8,
        labelspacing=0.25,
        borderpad=0.2,
    )


def plot_aupr_bar(
    ax,
    results: List[Dict],
    title: str,
):
    labels = [
        result["label"]
        for result in results
    ]

    auprs = [
        result["AUPR"]
        for result in results
    ]

    step1_aupr = get_step1_aupr(
        results
    )

    gains = [
        value - step1_aupr
        for value in auprs
    ]

    colors = get_colors(
        len(results)
    )

    best_index = int(
        np.argmax(auprs)
    )

    minimum = max(
        0.0,
        min(auprs) - 0.02,
    )

    maximum = min(
        1.0,
        max(auprs) + 0.07,
    )

    bars = ax.bar(
        range(len(results)),
        auprs,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )

    bars[best_index].set_linewidth(
        1.8
    )

    for index, (
        bar,
        value,
        gain,
    ) in enumerate(
        zip(
            bars,
            auprs,
            gains,
        )
    ):
        if index == 0:
            annotation = f"{value:.3f}"
        else:
            annotation = (
                f"{value:.3f}\n"
                f"({gain:+.3f})"
            )

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.004,
            annotation,
            ha="center",
            va="bottom",
            fontsize=(
                plt.rcParams["font.size"]
                * 0.42
            ),
        )

    ax.set_title(
        title,
        pad=8,
    )

    ax.set_ylabel("AUPR")

    ax.set_xticks(
        range(len(results))
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylim(
        minimum,
        maximum,
    )

    style_axes(ax)


def save_metrics_summary(
    all_results: Dict[str, List[Dict]],
    output_csv: str,
):
    rows = []

    for subset_name, results in all_results.items():
        for result in results:
            rows.append(
                {
                    "Subset": subset_name,
                    "Step": result["step"],
                    "Checkpoint": result["file"],
                    "Residues": result["n_samples"],
                    "Positive Fraction": (
                        result["positive_fraction"]
                    ),
                    "AUPR": result["AUPR"],
                    "Average Precision": (
                        result["AveragePrecision"]
                    ),
                }
            )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        output_csv,
        index=False,
    )

    print(
        f"Saved metric summary to:\n"
        f"  {output_csv}"
    )


def generate_figure(
    all_results: Dict[str, List[Dict]],
    output_png: str,
    font_scale: float,
    zoom_xlim: float,
    zoom_ylim_min: float,
):
    set_plot_style(
        scale=font_scale
    )

    subsets = [
        ("overall", "Overall", "A"),
        ("paratope", "Paratope", "B"),
        ("epitope", "Epitope", "C"),
    ]

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(17, 14),
    )

    for row_index, (
        subset_name,
        pretty_name,
        panel_label,
    ) in enumerate(subsets):

        results = all_results[
            subset_name
        ]

        full_axis = axes[
            row_index,
            0,
        ]

        zoom_axis = axes[
            row_index,
            1,
        ]

        bar_axis = axes[
            row_index,
            2,
        ]

        if subset_name == "epitope":
            full_legend_location = "upper right"
        else:
            full_legend_location = "lower left"

        plot_pr_full(
            full_axis,
            results,
            f"{pretty_name}: Full PR",
            legend_loc=full_legend_location,
        )

        plot_pr_zoom(
            zoom_axis,
            results,
            f"{pretty_name}: Zoomed PR",
            zoom_xlim=zoom_xlim,
            zoom_ylim_min=zoom_ylim_min,
            legend_loc="upper right",
        )

        plot_aupr_bar(
            bar_axis,
            results,
            f"{pretty_name}: AUPR",
        )

        add_panel_label(
            full_axis,
            panel_label,
        )

    plt.tight_layout(
        w_pad=2.0,
        h_pad=2.6,
    )

    plt.savefig(
        output_png,
        bbox_inches="tight",
    )

    print(
        f"Saved PR/AUPR figure to:\n"
        f"  {output_png}"
    )

    plt.show()


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run residue-level inference for an ordered list of CTSR "
            "checkpoints and generate overall, paratope, and epitope "
            "PR/AUPR plots."
        )
    )

    parser.add_argument(
        "--pth_list",
        type=str,
        required=True,
        help=(
            "Text file containing checkpoint paths in order: "
            "Step 1, Step 2, Step 3, Step 4."
        ),
    )

    parser.add_argument(
        "--sequence_file",
        type=str,
        default="para_test_esmsequences_1600.npz",
    )

    parser.add_argument(
        "--data_file",
        type=str,
        default="para_test_esminterfaces_1600.npz",
    )

    parser.add_argument(
        "--edge_file",
        type=str,
        default="para_test_esmedges_1600.npz",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="ctsr_pr_results",
    )

    parser.add_argument(
        "--out_png",
        type=str,
        default="ctsr_pr_curve_grid.png",
    )

    parser.add_argument(
        "--metrics_csv",
        type=str,
        default="ctsr_pr_metrics.csv",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=1600,
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=31,
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--font_scale",
        type=float,
        default=1.9,
    )

    parser.add_argument(
        "--zoom_xlim",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--zoom_ylim_min",
        type=float,
        default=0.7,
    )

    # By default, architecture values are parsed from each checkpoint filename.
    parser.add_argument(
        "--manual_model_config",
        action="store_true",
        help=(
            "Use the architecture arguments below instead of parsing "
            "them from checkpoint filenames."
        ),
    )

    parser.add_argument(
        "--embed_dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num_heads",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num_gnn_layers",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--num_int_layers",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--reuse_predictions",
        action="store_true",
        help=(
            "Skip inference when the expected residue prediction CSV "
            "already exists."
        ),
    )

    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Save the plot without opening a plot window.",
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    torch.backends.cudnn.benchmark = True

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    checkpoint_paths = read_checkpoint_list(
        args.pth_list
    )

    print(
        f"Loaded {len(checkpoint_paths)} checkpoints."
    )

    for step_number, checkpoint_path in enumerate(
        checkpoint_paths,
        start=1,
    ):
        print(
            f"  Step {step_number}: "
            f"{checkpoint_path}"
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint does not exist: "
                f"{checkpoint_path}"
            )

    test_config = {
        "sequence_file": args.sequence_file,
        "data_file": args.data_file,
        "edge_file": args.edge_file,
        "max_len": args.max_len,
        "vocab_size": args.vocab_size,
        "num_classes": args.num_classes,
    }

    test_dataset = SequenceParatopeDataset(
        data_file=test_config["data_file"],
        sequence_file=test_config["sequence_file"],
        edge_file=test_config["edge_file"],
        max_len=test_config["max_len"],
    )

    print(
        f"Number of test samples: "
        f"{len(test_dataset)}"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    interfaces_npz = np.load(
        test_config["data_file"],
        allow_pickle=True,
    )

    interface_keys = list(
        interfaces_npz.keys()
    )

    if len(interface_keys) != len(test_dataset):
        print(
            "WARNING: interface-key count does not match "
            "dataset length:"
        )
        print(
            f"  Interface keys: {len(interface_keys)}"
        )
        print(
            f"  Dataset samples: {len(test_dataset)}"
        )

    prediction_records = []

    for step_number, checkpoint_path in enumerate(
        checkpoint_paths,
        start=1,
    ):
        print("\n" + "=" * 80)
        print(
            f"Processing CTSR Step {step_number}"
        )
        print(
            f"Checkpoint: {checkpoint_path}"
        )
        print("=" * 80)

        model_stem = safe_filename_stem(
            checkpoint_path
        )

        prediction_csv = os.path.join(
            args.output_dir,
            (
                f"step{step_number}_"
                f"{model_stem}_"
                f"residue_predictions.csv"
            ),
        )

        if (
            args.reuse_predictions
            and os.path.exists(prediction_csv)
        ):
            print(
                "Using existing residue predictions:\n"
                f"  {prediction_csv}"
            )
        else:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model_config = get_model_config(
                checkpoint_path,
                args,
            )

            model = load_model(
                checkpoint_path=checkpoint_path,
                model_config=model_config,
                test_config=test_config,
                device=device,
            )

            run_checkpoint_inference(
                model=model,
                checkpoint_path=checkpoint_path,
                step_number=step_number,
                test_loader=test_loader,
                interface_keys=interface_keys,
                device=device,
                output_csv=prediction_csv,
                num_classes=test_config["num_classes"],
            )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        prediction_records.append(
            {
                "step": step_number,
                "checkpoint": checkpoint_path,
                "prediction_csv": prediction_csv,
            }
        )

    subsets = [
        "overall",
        "paratope",
        "epitope",
    ]

    all_results = {
        subset_name: collect_results(
            prediction_records,
            subset_name,
        )
        for subset_name in subsets
    }

    metrics_output_path = os.path.join(
        args.output_dir,
        args.metrics_csv,
    )

    save_metrics_summary(
        all_results,
        metrics_output_path,
    )

    figure_output_path = os.path.join(
        args.output_dir,
        args.out_png,
    )

    set_plot_style(
        scale=args.font_scale
    )

    subset_plot_info = [
        ("overall", "Overall", "A"),
        ("paratope", "Paratope", "B"),
        ("epitope", "Epitope", "C"),
    ]

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(17, 14),
    )

    for row_index, (
        subset_name,
        pretty_name,
        panel_label,
    ) in enumerate(subset_plot_info):

        results = all_results[
            subset_name
        ]

        full_axis = axes[
            row_index,
            0,
        ]

        zoom_axis = axes[
            row_index,
            1,
        ]

        bar_axis = axes[
            row_index,
            2,
        ]

        full_legend_location = (
            "upper right"
            if subset_name == "epitope"
            else "lower left"
        )

        plot_pr_full(
            full_axis,
            results,
            f"{pretty_name}: Full PR",
            legend_loc=full_legend_location,
        )

        plot_pr_zoom(
            zoom_axis,
            results,
            f"{pretty_name}: Zoomed PR",
            zoom_xlim=args.zoom_xlim,
            zoom_ylim_min=args.zoom_ylim_min,
            legend_loc="upper right",
        )

        plot_aupr_bar(
            bar_axis,
            results,
            f"{pretty_name}: AUPR",
        )

        add_panel_label(
            full_axis,
            panel_label,
        )

    plt.tight_layout(
        w_pad=2.0,
        h_pad=2.6,
    )

    plt.savefig(
        figure_output_path,
        bbox_inches="tight",
    )

    print(
        f"Saved PR/AUPR figure to:\n"
        f"  {figure_output_path}"
    )

    if args.no_show:
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
