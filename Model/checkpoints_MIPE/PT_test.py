#!/usr/bin/env python3
"""
PT_test_parareg_stats_clean.py

Clean evaluator for MGA/iPara paratope checkpoints.

Starting point: Model/PT_test_parareg.py, but with cleaner output:
  - evaluates all .pth paths listed in --checkpoint_list
  - parses l/g/i/do/dpr/lr/heads/fold from checkpoint filenames
  - reports only selected categories by default: Antibody, Antigen, All, macro_ab_ag
  - saves grouped summaries:
      per_checkpoint_metrics.csv
      grouped_by_config_category.csv        # l+g+i+do+dpr
      grouped_by_do_dpr_category.csv        # do+dpr only
      grouped_by_arch_category.csv          # l+g+i only
      best_checkpoints.csv                  # best .pth per category
      summary.txt                           # readable compact report
  - optionally copies best .pth files into out_dir/best_pth/

Example:
  python PT_test_parareg_stats_clean.py \
    --checkpoint_list pth_list.txt \
    --model_module Models_fullnew \
    --out_dir test_stats_fullnew_clean \
    --best_category Antigen \
    --copy_best

If you want chain-level details too:
  python PT_test_parareg_stats_clean.py ... --categories Antibody Antigen All macro_ab_ag Lchain Hchain
"""

import argparse
import csv
import importlib
import inspect
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    accuracy_score,
)

from Dataloader_itf import SequenceParatopeDataset


MAIN_CATEGORIES = ["Antibody", "Antigen", "All", "macro_ab_ag"]
CHAIN_CATEGORIES = ["Lchain", "Hchain"]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--checkpoint_list", "--inp_file", dest="checkpoint_list", required=True,
                   help="Text file containing one .pth path per line.")
    p.add_argument("--model_module", default="Models_fullnew",
                   help="Module containing ClassificationModel.")
    p.add_argument("--out_dir", default="pt_test_stats_clean")
    p.add_argument("--best_category", default="Antigen",
                   choices=["Antigen", "Antibody", "All", "macro_ab_ag", "Lchain", "Hchain"])
    p.add_argument("--categories", nargs="+", default=MAIN_CATEGORIES,
                   choices=["Antibody", "Antigen", "All", "macro_ab_ag", "Lchain", "Hchain"],
                   help="Categories to keep in final CSV/report. Default: Antibody Antigen All macro_ab_ag")
    p.add_argument("--copy_best", action="store_true")
    p.add_argument("--save_predictions", action="store_true")

    # Data
    p.add_argument("--sequence_file", default="MIPE_test_esmsequences_1600.npz")
    p.add_argument("--data_file", default="MIPE_test_esminterfaces_1600.npz")
    p.add_argument("--edge_file", default="MIPE_test_esmedges_1600.npz")
    p.add_argument("--max_len", type=int, default=1600)
    p.add_argument("--vocab_size", type=int, default=31)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)

    # Defaults if missing from filename
    p.add_argument("--default_embed_dim", type=int, default=256)
    p.add_argument("--default_heads", type=int, default=16)
    p.add_argument("--default_lr", type=float, default=np.nan)

    # ESM args, passed only if constructor accepts them
    p.add_argument("--use_esm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--esm_model_name", default="esm2_t6_8M_UR50D")
    p.add_argument("--esm_repr_layer", type=int, default=None)
    p.add_argument("--esm_alpha_init", type=float, default=0.0)
    p.add_argument("--esm_adapter_dropout", type=float, default=0.10)
    p.add_argument("--esm_freeze", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--esm_max_len", type=int, default=1022)

    # Special tokens
    p.add_argument("--pad_id", type=int, default=1)
    p.add_argument("--eoc_id", type=int, default=24)

    p.add_argument("--strict_load", action="store_true")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--quiet", action="store_true", help="Print less per-checkpoint detail.")

    return p.parse_args()


def custom_collate_fn(batch):
    sequences, pt, edge = zip(*batch)
    sequence_tensor = torch.stack([torch.as_tensor(s).long() for s in sequences])
    pt_tensor = torch.tensor(np.array(pt), dtype=torch.long)

    max_edges = max(edge_index.shape[0] for edge_index in edge)
    padded_edges = []
    for edge_index in edge:
        edge_index = torch.as_tensor(edge_index).long()
        edge_pad = -torch.ones((2, max_edges), dtype=torch.long)
        edge_pad[:, :edge_index.shape[0]] = edge_index.T.clone().detach()
        padded_edges.append(edge_pad)

    return torch.stack(padded_edges), sequence_tensor, pt_tensor


def read_checkpoint_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def parse_checkpoint_config(model_file, args):
    name = Path(model_file).name

    def grab(pattern, cast, default=np.nan):
        m = re.search(pattern, name)
        if not m:
            return default
        try:
            return cast(m.group(1))
        except Exception:
            return default

    cfg = {
        "checkpoint": model_file,
        "checkpoint_name": name,
        "l": grab(r"(?:^|_)l(\d+)(?:_|\.|$)", int),
        "g": grab(r"(?:^|_)g(\d+)(?:_|\.|$)", int),
        "i": grab(r"(?:^|_)i(\d+)(?:_|\.|$)", int),
        "do": grab(r"(?:^|_)do([0-9]*\.?[0-9]+)(?:_|\.|$)", float),
        "dpr": grab(r"(?:^|_)dpr([0-9]*\.?[0-9]+)(?:_|\.|$)", float),
        "lr": grab(r"(?:^|_)lr([0-9.eE+-]+)(?:_|\.|$)", float, args.default_lr),
        "heads": grab(r"(?:^|_)heads(\d+)(?:_|\.|$)", int, args.default_heads),
        "fold": grab(r"(?:^|_)fold(\d+)(?:_|\.pth$|$)", int),
    }

    missing = [k for k in ["l", "g", "i", "do", "dpr"] if pd.isna(cfg[k])]
    if missing:
        raise ValueError(
            f"Could not parse {missing} from filename: {name}. "
            "Expected checkpoint name containing l#, g#, i#, do#, dpr#."
        )
    return cfg


def first_pad_valid_len(tokens, pad_id):
    pad_pos = (tokens == pad_id).nonzero(as_tuple=True)[0]
    return int(pad_pos[0].item()) if pad_pos.numel() > 0 else int(tokens.numel())


def clean_chain_spans(query_tokens, eoc_id=24, pad_id=1):
    """
    Clean metric split. EOC and PAD are excluded from metric spans.

      L ... EOC H ... EOC AG1 ... EOC AG2 ... PAD
      -> L, H, AG1, AG2
    """
    query_tokens = query_tokens.detach().cpu().long()
    valid_len = first_pad_valid_len(query_tokens, pad_id)
    toks = query_tokens[:valid_len]
    eocs = (toks == eoc_id).nonzero(as_tuple=True)[0].tolist()

    spans = []
    start = 0
    for eoc in eocs:
        eoc = int(eoc)
        if eoc > start:
            spans.append((start, eoc))
        start = eoc + 1
    if start < valid_len:
        spans.append((start, valid_len))
    return spans


def safe_binary_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    if y_true.size == 0:
        return dict(N=0, Pos=0, Neg=0, Accuracy=np.nan, Precision=np.nan,
                    Recall=np.nan, AUC_ROC=np.nan, AUC_PR=np.nan)

    out = {
        "N": int(y_true.size),
        "Pos": int((y_true == 1).sum()),
        "Neg": int((y_true == 0).sum()),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "AUC_ROC": np.nan,
        "AUC_PR": np.nan,
    }
    if np.unique(y_true).size >= 2:
        out["AUC_ROC"] = float(roc_auc_score(y_true, y_prob))
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
        out["AUC_PR"] = float(auc(recall_vals, precision_vals))
    return out


def add_to_store(metrics, category, true_labels, pred_labels, probs):
    if category not in metrics:
        metrics[category] = {"true": [], "pred": [], "probs": []}
    metrics[category]["true"].extend(true_labels)
    metrics[category]["pred"].extend(pred_labels)
    metrics[category]["probs"].extend(probs)


def strip_module_prefix(state):
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}


def extract_state_dict(obj):
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint is not a dict: {type(obj)}")
    for key in ["model_state_dict", "state_dict", "model", "model_state"]:
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
    return obj


def load_checkpoint(model, model_file, device, strict=False):
    try:
        ckpt = torch.load(model_file, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(model_file, map_location=device)

    state = strip_module_prefix(extract_state_dict(ckpt))
    state.pop("n_averaged", None)

    if strict:
        model.load_state_dict(state, strict=True)
        return {"loaded": len(state), "missing": 0, "unexpected": 0, "shape_skipped": 0}

    model_state = model.state_dict()
    loaded = {}
    unexpected = []
    skipped_shape = []
    for k, v in state.items():
        if k not in model_state:
            unexpected.append(k)
            continue
        if tuple(v.shape) != tuple(model_state[k].shape):
            skipped_shape.append(k)
            continue
        loaded[k] = v

    model_state.update(loaded)
    missing = sorted(set(model_state.keys()) - set(loaded.keys()))
    model.load_state_dict(model_state, strict=True)
    return {
        "loaded": len(loaded),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "shape_skipped": len(skipped_shape),
    }


def build_model(cfg, args, device):
    mod = importlib.import_module(args.model_module)
    ClassificationModel = getattr(mod, "ClassificationModel")

    kwargs = dict(
        vocab_size=args.vocab_size,
        seq_len=args.max_len,
        embed_dim=args.default_embed_dim,
        num_heads=int(cfg["heads"]),
        dropout=float(cfg["do"]),
        drop_path_rate=float(cfg["dpr"]),
        num_layers=int(cfg["l"]),
        num_gnn_layers=int(cfg["g"]),
        num_int_layers=int(cfg["i"]),
        num_classes=args.num_classes,
    )

    sig = inspect.signature(ClassificationModel.__init__)
    allowed = set(sig.parameters.keys())
    optional = dict(
        use_esm=args.use_esm,
        esm_model_name=args.esm_model_name,
        esm_repr_layer=args.esm_repr_layer,
        esm_alpha_init=args.esm_alpha_init,
        esm_adapter_dropout=args.esm_adapter_dropout,
        esm_freeze=args.esm_freeze,
        esm_max_len=args.esm_max_len,
        pad_id=args.pad_id,
        eoc_id=args.eoc_id,
    )
    for k, v in optional.items():
        if k in allowed:
            kwargs[k] = v

    return ClassificationModel(**kwargs).to(device)


def forward_model(model, sequence_tensor, padded_edges, args):
    use_amp = bool(args.amp and torch.cuda.is_available())
    with torch.amp.autocast("cuda", enabled=use_amp):
        try:
            out = model(sequences=sequence_tensor, padded_edges=padded_edges, return_attention=True)
        except TypeError:
            out = model(sequence_tensor, padded_edges, return_attention=True)
    if isinstance(out, tuple):
        outputs = out[0]
    else:
        outputs = out
    return outputs.squeeze(1)


def evaluate_checkpoint(model_file, test_loader, args, device, out_dir):
    cfg = parse_checkpoint_config(model_file, args)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.quiet:
        print("\n" + "=" * 100)
        print(f"Testing: {model_file}")
        print(f"Config: l={cfg['l']} g={cfg['g']} i={cfg['i']} do={cfg['do']} dpr={cfg['dpr']} fold={cfg['fold']}")

    model = build_model(cfg, args, device)
    load_info = load_checkpoint(model, model_file, device, strict=args.strict_load)
    model.eval()

    metrics = {
        "Lchain": {"true": [], "pred": [], "probs": []},
        "Hchain": {"true": [], "pred": [], "probs": []},
        "Antibody": {"true": [], "pred": [], "probs": []},
        "Antigen": {"true": [], "pred": [], "probs": []},
        "All": {"true": [], "pred": [], "probs": []},
    }

    pred_writer = None
    pred_f = None
    if args.save_predictions:
        pred_path = out_dir / f"{Path(model_file).name}_predictions.csv"
        pred_f = open(pred_path, "w", newline="")
        pred_writer = csv.writer(pred_f)
        pred_writer.writerow(["sample_index", "chain_type", "position", "true_label", "predicted_label", "probability"])

    sample_idx = 0
    with torch.no_grad():
        for padded_edges, sequence_tensor, pt_tensor in test_loader:
            padded_edges = padded_edges.to(device)
            sequence_tensor = sequence_tensor.to(device)
            pt_tensor = pt_tensor.to(device)

            outputs = forward_model(model, sequence_tensor, padded_edges, args)
            probs = torch.softmax(outputs, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            B = sequence_tensor.shape[0]
            for b in range(B):
                spans = clean_chain_spans(sequence_tensor[b, 0, :], eoc_id=args.eoc_id, pad_id=args.pad_id)

                for span_idx, (start, end) in enumerate(spans):
                    if span_idx == 0:
                        chain_type = "Lchain"
                        group_type = "Antibody"
                    elif span_idx == 1:
                        chain_type = "Hchain"
                        group_type = "Antibody"
                    else:
                        chain_type = f"AGchain_{span_idx - 1}"
                        group_type = "Antigen"

                    mask = pt_tensor[b, start:end] >= 0
                    if mask.sum().item() == 0:
                        continue

                    pos = np.arange(start, end)[mask.detach().cpu().numpy()]
                    true_labels = pt_tensor[b, start:end][mask].detach().cpu().numpy()
                    pred_labels = preds[b, start:end][mask].detach().cpu().numpy()
                    prob_values = probs[b, start:end, 1][mask].detach().cpu().numpy()

                    add_to_store(metrics, chain_type, true_labels, pred_labels, prob_values)
                    add_to_store(metrics, group_type, true_labels, pred_labels, prob_values)
                    add_to_store(metrics, "All", true_labels, pred_labels, prob_values)

                    if pred_writer is not None:
                        for p0, t, prd, prob in zip(pos, true_labels, pred_labels, prob_values):
                            pred_writer.writerow([sample_idx, chain_type, int(p0), int(t), int(prd), float(prob)])
                sample_idx += 1

    if pred_f is not None:
        pred_f.close()

    rows = []
    for category, data in metrics.items():
        m = safe_binary_metrics(data["true"], data["pred"], data["probs"])
        row = dict(cfg)
        row["Category"] = category
        row.update(m)
        row.update(load_info)
        rows.append(row)

    row_ab = next((r for r in rows if r["Category"] == "Antibody"), None)
    row_ag = next((r for r in rows if r["Category"] == "Antigen"), None)
    if row_ab is not None and row_ag is not None:
        macro = dict(cfg)
        macro["Category"] = "macro_ab_ag"
        for key in ["Accuracy", "Precision", "Recall", "AUC_ROC", "AUC_PR"]:
            macro[key] = float(np.nanmean([row_ab[key], row_ag[key]]))
        macro["N"] = int(row_ab["N"] + row_ag["N"])
        macro["Pos"] = int(row_ab["Pos"] + row_ag["Pos"])
        macro["Neg"] = int(row_ab["Neg"] + row_ag["Neg"])
        macro.update(load_info)
        rows.append(macro)

    if not args.quiet:
        keep = set(args.categories)
        compact = pd.DataFrame([r for r in rows if r["Category"] in keep])
        cols = ["Category", "N", "Pos", "AUC_ROC", "AUC_PR", "Accuracy", "Precision", "Recall"]
        print(compact[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def group_summary(df, cols):
    return (
        df.groupby(cols, dropna=False)
          .agg(
              n_ckpt=("checkpoint", "count"),
              mean_AUC_PR=("AUC_PR", "mean"),
              std_AUC_PR=("AUC_PR", "std"),
              max_AUC_PR=("AUC_PR", "max"),
              mean_AUC_ROC=("AUC_ROC", "mean"),
              std_AUC_ROC=("AUC_ROC", "std"),
              max_AUC_ROC=("AUC_ROC", "max"),
              mean_Accuracy=("Accuracy", "mean"),
              mean_Precision=("Precision", "mean"),
              mean_Recall=("Recall", "mean"),
          )
          .reset_index()
          .sort_values(["Category", "mean_AUC_PR"], ascending=[True, False])
    )


def best_checkpoints(df):
    rows = []
    for category, sub in df.groupby("Category"):
        sub = sub.dropna(subset=["AUC_PR"])
        if len(sub) == 0:
            continue
        best = sub.sort_values("AUC_PR", ascending=False).iloc[0].to_dict()
        best["Best_For_Category"] = category
        rows.append(best)
    return pd.DataFrame(rows)


def format_table(df, cols, max_rows=None):
    if max_rows is not None:
        df = df.head(max_rows)
    cols = [c for c in cols if c in df.columns]
    if len(df) == 0:
        return "[empty]"
    return df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}")


def summarize_results(df, args, out_dir):
    # Keep output categories only. This makes every CSV cleaner.
    keep = set(args.categories)
    df_keep = df[df["Category"].isin(keep)].copy()

    grouped_config = group_summary(df_keep, ["l", "g", "i", "do", "dpr", "Category"])
    grouped_do_dpr = group_summary(df_keep, ["do", "dpr", "Category"])
    grouped_arch = group_summary(df_keep, ["l", "g", "i", "Category"])
    best_df = best_checkpoints(df_keep)

    # Save clean CSVs.
    df_keep.to_csv(out_dir / "per_checkpoint_metrics.csv", index=False)
    grouped_config.to_csv(out_dir / "grouped_by_config_category.csv", index=False)
    grouped_do_dpr.to_csv(out_dir / "grouped_by_do_dpr_category.csv", index=False)
    grouped_arch.to_csv(out_dir / "grouped_by_arch_category.csv", index=False)
    best_df.to_csv(out_dir / "best_checkpoints.csv", index=False)

    primary = best_df[best_df["Best_For_Category"] == args.best_category]
    if len(primary) == 0 and len(best_df) > 0:
        primary = best_df.sort_values("AUC_PR", ascending=False).head(1)

    if args.copy_best and len(best_df) > 0:
        best_dir = out_dir / "best_pth"
        best_dir.mkdir(parents=True, exist_ok=True)
        for _, row in best_df.iterrows():
            src = Path(row["checkpoint"])
            if src.exists():
                dst = best_dir / f"best_{row['Best_For_Category']}_{src.name}"
                shutil.copy2(src, dst)
        for _, row in primary.iterrows():
            src = Path(row["checkpoint"])
            if src.exists():
                dst = best_dir / f"PRIMARY_best_{args.best_category}_{src.name}"
                shutil.copy2(src, dst)

    # Write readable summary.
    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append("PRIMARY BEST CHECKPOINT")
    summary_lines.append("=" * 100)
    summary_lines.append(format_table(
        primary,
        ["Best_For_Category", "checkpoint", "l", "g", "i", "do", "dpr", "fold", "AUC_PR", "AUC_ROC", "Accuracy"],
    ))

    summary_lines.append("\n" + "=" * 100)
    summary_lines.append("BEST CHECKPOINT BY CATEGORY")
    summary_lines.append("=" * 100)
    summary_lines.append(format_table(
        best_df.sort_values(["Best_For_Category"]),
        ["Best_For_Category", "checkpoint", "l", "g", "i", "do", "dpr", "fold", "AUC_PR", "AUC_ROC", "Accuracy"],
    ))

    if "Antigen" in keep:
        ag_do_dpr = grouped_do_dpr[grouped_do_dpr["Category"] == "Antigen"].sort_values("mean_AUC_PR", ascending=False)
        ag_config = grouped_config[grouped_config["Category"] == "Antigen"].sort_values("mean_AUC_PR", ascending=False)
        ag_arch = grouped_arch[grouped_arch["Category"] == "Antigen"].sort_values("mean_AUC_PR", ascending=False)

        summary_lines.append("\n" + "=" * 100)
        summary_lines.append("ANTIGEN: GROUPED BY do + dpr")
        summary_lines.append("=" * 100)
        summary_lines.append(format_table(
            ag_do_dpr,
            ["do", "dpr", "n_ckpt", "mean_AUC_PR", "std_AUC_PR", "max_AUC_PR", "mean_AUC_ROC", "max_AUC_ROC"],
        ))

        summary_lines.append("\n" + "=" * 100)
        summary_lines.append("ANTIGEN: TOP FULL CONFIGS l + g + i + do + dpr")
        summary_lines.append("=" * 100)
        summary_lines.append(format_table(
            ag_config,
            ["l", "g", "i", "do", "dpr", "n_ckpt", "mean_AUC_PR", "std_AUC_PR", "max_AUC_PR", "mean_AUC_ROC"],
            max_rows=30,
        ))

        summary_lines.append("\n" + "=" * 100)
        summary_lines.append("ANTIGEN: GROUPED BY ARCHITECTURE l + g + i")
        summary_lines.append("=" * 100)
        summary_lines.append(format_table(
            ag_arch,
            ["l", "g", "i", "n_ckpt", "mean_AUC_PR", "std_AUC_PR", "max_AUC_PR", "mean_AUC_ROC"],
        ))

    # Add antibody do+dpr only if requested.
    if "Antibody" in keep:
        ab_do_dpr = grouped_do_dpr[grouped_do_dpr["Category"] == "Antibody"].sort_values("mean_AUC_PR", ascending=False)
        summary_lines.append("\n" + "=" * 100)
        summary_lines.append("ANTIBODY: GROUPED BY do + dpr")
        summary_lines.append("=" * 100)
        summary_lines.append(format_table(
            ab_do_dpr,
            ["do", "dpr", "n_ckpt", "mean_AUC_PR", "std_AUC_PR", "max_AUC_PR", "mean_AUC_ROC", "max_AUC_ROC"],
        ))

    summary_lines.append("\n" + "=" * 100)
    summary_lines.append("SAVED FILES")
    summary_lines.append("=" * 100)
    for name in [
        "per_checkpoint_metrics.csv",
        "grouped_by_config_category.csv",
        "grouped_by_do_dpr_category.csv",
        "grouped_by_arch_category.csv",
        "best_checkpoints.csv",
        "summary.txt",
    ]:
        summary_lines.append(str(out_dir / name))

    summary_text = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(summary_text)

    print("\n" + summary_text)


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = read_checkpoint_list(args.checkpoint_list)
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {args.checkpoint_list}")

    print(f"Found {len(ckpts)} checkpoints")
    print(f"Model module: {args.model_module}")
    print(f"Categories kept: {args.categories}")
    print(f"Output dir: {out_dir}")

    dataset = SequenceParatopeDataset(
        data_file=args.data_file,
        sequence_file=args.sequence_file,
        edge_file=args.edge_file,
        max_len=args.max_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    all_rows = []
    failures = []
    for ckpt in ckpts:
        try:
            all_rows.extend(evaluate_checkpoint(ckpt, loader, args, device, out_dir))
        except Exception as exc:
            print("\n" + "!" * 100)
            print(f"FAILED: {ckpt}")
            print(f"ERROR: {repr(exc)}")
            print("!" * 100)
            failures.append({"checkpoint": ckpt, "error": repr(exc)})

    if not all_rows:
        raise RuntimeError("No checkpoints were successfully evaluated.")

    df = pd.DataFrame(all_rows)
    summarize_results(df, args, out_dir)

    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failed_checkpoints.csv", index=False)
        print(f"Failures: {len(failures)}. Saved {out_dir / 'failed_checkpoints.csv'}")


if __name__ == "__main__":
    main()

