import pandas as pd
import sys
import os
import re

file_name = sys.argv[1]

results = []


def parse_model_name(model_path):
    """
    Robustly parse fold/checkpoint type from .pth filename.

    Examples:
        fold1.pth
        fold1_best_aupr.pth
        fold1_best_loss.pth
        fold1_balanced.pth
        fold1_best_balanced.pth
        MGA_fold1_best_aupr.pth
    """

    base = os.path.basename(model_path)
    stem = base.replace(".pth", "")

    # Fold number: fold1, fold_1, Fold10, etc.
    fold_match = re.search(r"fold[_-]?(\d+)", stem, flags=re.IGNORECASE)
    fold = fold_match.group(1) if fold_match else "NA"

    # Checkpoint type
    lower = stem.lower()

    if "best_balanced" in lower:
        checkpoint_type = "best_balanced"
    elif "balanced" in lower:
        checkpoint_type = "balanced"
    elif "best_aupr" in lower or "best_aucpr" in lower:
        checkpoint_type = "best_aupr"
    elif "best_loss" in lower:
        checkpoint_type = "best_loss"
    elif "best" in lower:
        checkpoint_type = "best"
    else:
        checkpoint_type = "main"

    # Structure/dropout parsing from old naming style, if available
    parts = stem.split("_")

    structure = "NA"
    dropout = None

    # Old format expected something like:
    # model_x_x_x_do0.1_x_x_fold1.pth
    try:
        structure = "_".join(parts[1:7])
    except Exception:
        structure = "NA"

    # Find dropout token like do0.1, dropout0.1, dp0.1
    for p in parts:
        m = re.match(r"(do|dropout|dp)([0-9.]+)", p.lower())
        if m:
            try:
                dropout = float(m.group(2))
            except Exception:
                dropout = None
            break

    return fold, structure, dropout, checkpoint_type


with open(file_name, "r") as file:
    current_model = None
    current_fold = None
    model_structure = None
    dropout = None
    checkpoint_type = None

    category = None
    auc_pr = None
    auc_roc = None

    lines = file.readlines()

    for i in range(len(lines)):
        line = lines[i].strip()

        # Detect tested model/checkpoint
        if line.startswith("Testing model:"):
            current_model = line.split("Testing model:", 1)[1].strip()

            current_fold, model_structure, dropout, checkpoint_type = parse_model_name(
                current_model
            )

            category = None
            auc_pr = None
            auc_roc = None

        # Detect category
        if line.startswith("Metrics for "):
            category = line.split("Metrics for ", 1)[1].rstrip(":").strip()
            auc_pr = None
            auc_roc = None

        # Skip per-chain categories if desired
        if category in {
            "AGchain_0",
            "AGchain_1",
            "AGchain_2",
            "AGchain_3",
            "Lchain",
            "Hchain",
        }:
            continue

        # Extract metrics
        if "AUC-ROC:" in line:
            try:
                auc_roc = float(line.split("AUC-ROC:")[-1].strip())
            except ValueError:
                auc_roc = None

        if "AUC-PR:" in line:
            try:
                auc_pr = float(line.split("AUC-PR:")[-1].strip())
            except ValueError:
                auc_pr = None

        # Save result when complete
        if (
            current_model is not None
            and category is not None
            and auc_pr is not None
            and auc_roc is not None
        ):
            results.append(
                {
                    "Model": current_model,
                    "Fold": current_fold,
                    "Checkpoint_Type": checkpoint_type,
                    "Structure": model_structure,
                    "Dropout": dropout,
                    "Category": category,
                    "AUC-ROC": auc_roc,
                    "AUC-PR": auc_pr,
                }
            )

            auc_pr = None
            auc_roc = None


df = pd.DataFrame(results)

if df.empty:
    print("No metrics found. Check that the output contains lines like:")
    print("  Testing model: fold1_best_aupr.pth")
    print("  Metrics for Antibody:")
    print("  AUC-ROC: 0.XXXX")
    print("  AUC-PR: 0.XXXX")
    sys.exit(1)


# Detailed metrics
df.to_csv("metrics_summary.csv", index=False)

# Average across folds, separately for each checkpoint type
avg_metrics = (
    df.groupby(["Checkpoint_Type", "Structure", "Dropout", "Category"], dropna=False)[
        ["AUC-ROC", "AUC-PR"]
    ]
    .mean()
    .reset_index()
)

# Maximum across folds, separately for each checkpoint type
max_metrics = (
    df.groupby(["Checkpoint_Type", "Structure", "Dropout", "Category"], dropna=False)[
        ["AUC-ROC", "AUC-PR"]
    ]
    .max()
    .reset_index()
)

# Fold-level summary
fold_metrics = (
    df.groupby(["Checkpoint_Type", "Fold", "Category"], dropna=False)[
        ["AUC-ROC", "AUC-PR"]
    ]
    .mean()
    .reset_index()
)

# Best checkpoint type by average AUC-PR
best_by_aupr = (
    avg_metrics.sort_values("AUC-PR", ascending=False)
    .groupby(["Category"], dropna=False)
    .head(1)
    .reset_index(drop=True)
)

avg_metrics.to_csv("avg_metrics_summary.csv", index=False)
max_metrics.to_csv("max_metrics_summary.csv", index=False)
fold_metrics.to_csv("fold_metrics_summary.csv", index=False)
best_by_aupr.to_csv("best_checkpoint_by_aupr.csv", index=False)

print("Detailed Metrics:")
print(df)

print("\nAverage Metrics:")
print(avg_metrics)

print("\nMaximum Metrics:")
print(max_metrics)

print("\nFold Metrics:")
print(fold_metrics)

print("\nBest Checkpoint Type by AUC-PR:")
print(best_by_aupr)
