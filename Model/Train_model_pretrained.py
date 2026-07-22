#!/usr/bin/env python3
"""
Train_model_pretrained.py

Balanced MGA/iPara training script with optional pretrained checkpoint loading.

Features:
  - --model_module: Models_fullnew / Models_newesm / Models_newesm_separate_adapters
  - --pretrained_ckpt or --pretrained_ckpt_pattern with {fold}
  - partial checkpoint loading by matching key + shape
  - strips DataParallel module. prefix
  - token_drop_p defaults to 0.0; no PAD token-drop corruption
  - EOC/PAD excluded from antibody/antigen metric/loss masks
  - warmup + cosine LR scheduler
  - scheduler-safe DyM unfreezing
"""

import os
import copy
import math
import random
import argparse
import contextlib
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score

from Dataloader_itf import SequenceParatopeDataset

try:
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler
    _NEW_AMP = True
except Exception:
    from torch.cuda.amp import autocast as _autocast
    from torch.cuda.amp import GradScaler as _GradScaler
    _NEW_AMP = False


def amp_autocast(enabled=True):
    if not enabled or not torch.cuda.is_available():
        return contextlib.nullcontext()
    if _NEW_AMP:
        return _autocast("cuda", enabled=True)
    return _autocast(enabled=True)


def make_grad_scaler(enabled=True):
    enabled = bool(enabled and torch.cuda.is_available())
    if _NEW_AMP:
        return _GradScaler("cuda", enabled=enabled)
    return _GradScaler(enabled=enabled)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def custom_collate_fn(batch):
    seqs, labels, edges = zip(*batch)
    seqs = torch.stack([torch.as_tensor(s).long() for s in seqs])
    labels = torch.tensor(np.array(labels), dtype=torch.long)

    max_e = max(e.shape[0] for e in edges)
    pads = []
    for e in edges:
        e = torch.as_tensor(e).long()
        pad = -torch.ones((2, max_e), dtype=torch.long)
        pad[:, : e.shape[0]] = e.T.clone().detach()
        pads.append(pad)
    edges = torch.stack(pads)
    return edges, seqs, labels


def get_ab_ag_masks(seqs, labels, eoc_id=24, pad_id=1):
    """
    Query row layout:
      L ... EOC H ... EOC AG1 ... [EOC AG2 ...] PAD

    antibody = L + H residues only
    antigen  = residues after second EOC, excluding later EOCs
    valid    = labels >= 0 and not PAD/EOC
    """
    B, L = labels.shape
    device = labels.device
    q = seqs[:, 0, :]

    valid = (labels >= 0) & (q != pad_id) & (q != eoc_id)
    ab_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    ag_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    ar = torch.arange(L, device=device)

    for b in range(B):
        pad_pos = torch.where(q[b] == pad_id)[0]
        valid_len = int(pad_pos[0].item()) if pad_pos.numel() > 0 else L
        eocs = torch.where((q[b] == eoc_id) & (ar < valid_len))[0]

        if eocs.numel() >= 2:
            e0 = int(eocs[0].item())
            e1 = int(eocs[1].item())
            ab_mask[b, :e0] = True
            ab_mask[b, e0 + 1:e1] = True
            ag_mask[b, e1 + 1:valid_len] = True
        else:
            ab_mask[b, :valid_len] = True

        special = (q[b] == eoc_id) | (q[b] == pad_id)
        ab_mask[b, special] = False
        ag_mask[b, special] = False

    ab_mask &= valid
    ag_mask &= valid
    return valid, ab_mask, ag_mask


def safe_aupr_np(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def precision_at_k(y_true, y_prob, group_ids, k=5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    group_ids = np.asarray(group_ids)
    if y_true.size == 0:
        return float("nan")
    vals = []
    for gid in np.unique(group_ids):
        m = group_ids == gid
        if not m.any():
            continue
        yt = y_true[m]
        yp = y_prob[m]
        kk = min(k, len(yt))
        idx = np.argsort(-yp)[:kk]
        vals.append(float(np.mean(yt[idx])))
    return float(np.mean(vals)) if vals else float("nan")


def balanced_selection_metric(val_m, ab_weight=0.5, ag_weight=0.5):
    ab = val_m["aupr_ab"]
    ag = val_m["aupr_ag"]
    all_ = val_m["aupr_all"]
    if not np.isnan(ab) and not np.isnan(ag):
        return float(ab_weight * ab + ag_weight * ag)
    if not np.isnan(all_):
        return float(all_)
    if not np.isnan(ab):
        return float(ab)
    if not np.isnan(ag):
        return float(ag)
    return float("nan")


def dropedge_padded(padded_edges, p):
    if p <= 0:
        return padded_edges
    out = padded_edges.clone()
    B, _, E = out.shape
    valid = (out[:, 0, :] >= 0) & (out[:, 1, :] >= 0)
    drop = (torch.rand((B, E), device=out.device) < p) & valid
    out[:, 0, :][drop] = -1
    out[:, 1, :][drop] = -1
    return out


def token_dropout(seqs, labels, p, pad_id=1, eoc_id=24, replace_id=None):
    """Safe token dropout. Do not replace residues with PAD."""
    if p <= 0 or replace_id is None:
        return seqs
    if replace_id in (pad_id, eoc_id):
        raise ValueError(f"replace_id must not be PAD/EOC, got {replace_id}")
    out = seqs.clone()
    q = out[:, 0, :]
    valid = labels >= 0
    can_drop = valid & (q != pad_id) & (q != eoc_id)
    drop = (torch.rand_like(q.float()) < p) & can_drop
    out[:, 0, :][drop] = int(replace_id)
    return out


def graph_smooth_loss(logits, padded_edges, labels):
    probs = torch.softmax(logits, dim=-1)[..., 1]
    B, L = probs.shape
    total = logits.new_tensor(0.0)
    count = 0
    for b in range(B):
        src = padded_edges[b, 0]
        dst = padded_edges[b, 1]
        valid = (src >= 0) & (dst >= 0) & (src < L) & (dst < L)
        if valid.any():
            s = src[valid].long()
            d = dst[valid].long()
            ok = (labels[b, s] >= 0) & (labels[b, d] >= 0)
            if ok.any():
                total = total + (probs[b, s[ok]] - probs[b, d[ok]]).pow(2).mean()
                count += 1
    return total / count if count > 0 else logits.new_tensor(0.0)


def soft_pr_auc_loss(logits, labels, mask, bins=64, temp=0.05):
    y = labels[mask].float()
    if y.numel() == 0 or y.sum() == 0:
        return logits.new_tensor(0.0)
    scores = torch.softmax(logits, dim=-1)[..., 1][mask]
    thresholds = torch.linspace(0, 1, bins, device=logits.device)
    pred_pos = torch.sigmoid((scores[:, None] - thresholds[None, :]) / temp)
    tp = (pred_pos * y[:, None]).sum(dim=0)
    fp = (pred_pos * (1.0 - y[:, None])).sum(dim=0)
    fn = ((1.0 - pred_pos) * y[:, None]).sum(dim=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    order = torch.argsort(recall)
    area = torch.trapz(precision[order], recall[order]).clamp(0.0, 1.0)
    return 1.0 - area


def squeeze_logits(logits):
    if logits.dim() == 4 and logits.shape[1] == 1:
        return logits.squeeze(1)
    return logits


def set_dym_trainable(model, trainable):
    for name, p in model.named_parameters():
        if "dym" in name.lower():
            p.requires_grad = trainable


def get_model_state(model):
    if isinstance(model, nn.DataParallel):
        return copy.deepcopy(model.module.state_dict())
    return copy.deepcopy(model.state_dict())


def strip_module_prefix(state):
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }


def load_pretrained_partial(model, ckpt_path, device):
    if ckpt_path is None or str(ckpt_path).strip() == "":
        print("[pretrained] No pretrained checkpoint specified.")
        return
    ckpt_path = str(ckpt_path)
    if not os.path.exists(ckpt_path):
        print(f"[pretrained] WARNING: checkpoint not found: {ckpt_path}")
        return

    print(f"[pretrained] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        elif "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            state = ckpt["model_state"]
        else:
            state = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    state = strip_module_prefix(state)
    model_state = model.state_dict()
    loaded = {}
    skipped_missing = []
    skipped_shape = []

    for k, v in state.items():
        if k not in model_state:
            skipped_missing.append(k)
            continue
        if tuple(v.shape) != tuple(model_state[k].shape):
            skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        loaded[k] = v

    model_state.update(loaded)
    model.load_state_dict(model_state, strict=True)

    current_keys = set(model.state_dict().keys())
    loaded_keys = set(loaded.keys())
    not_loaded = sorted(current_keys - loaded_keys)
    esm_not_loaded = [k for k in not_loaded if "esm_query_fusion" in k]

    print(f"[pretrained] loaded keys: {len(loaded)}")
    print(f"[pretrained] skipped missing-in-current keys: {len(skipped_missing)}")
    print(f"[pretrained] skipped shape-mismatch keys: {len(skipped_shape)}")
    print(f"[pretrained] current model keys not loaded: {len(not_loaded)}")
    if not_loaded:
        print("[pretrained] current model keys not loaded:")
        for k in not_loaded:
            print("  ", k)
    if esm_not_loaded:
        print(f"[pretrained] ESM/fusion keys initialized from scratch: {len(esm_not_loaded)}")
        for k in esm_not_loaded[:20]:
            print("  ", k)
    if skipped_missing[:10]:
        print("[pretrained] first skipped missing-in-current keys:")
        for k in skipped_missing[:10]:
            print("  ", k)
    if skipped_shape[:10]:
        print("[pretrained] first shape mismatches:")
        for k, src, dst in skipped_shape[:10]:
            print(f"  {k}: ckpt={src}, model={dst}")


def ramp_value(epoch, start, length, final_value):
    if epoch < start:
        return 0.0
    if length <= 0:
        return final_value
    return final_value * min(1.0, (epoch - start + 1) / float(length))


def update_metric_lists(labels, probs, valid_mask, ab_mask, ag_mask, stores):
    B, L = labels.shape
    group = torch.arange(B, device=labels.device).view(B, 1).expand(B, L)
    for name, mask in [("all", valid_mask), ("ab", ab_mask), ("ag", ag_mask)]:
        if mask.any():
            stores[name]["true"].extend(labels[mask].detach().cpu().numpy().tolist())
            stores[name]["prob"].extend(probs[mask].detach().cpu().numpy().tolist())
            stores[name]["group"].extend(group[mask].detach().cpu().numpy().tolist())


def summarize_stores(stores):
    return {
        "aupr_all": safe_aupr_np(stores["all"]["true"], stores["all"]["prob"]),
        "aupr_ab": safe_aupr_np(stores["ab"]["true"], stores["ab"]["prob"]),
        "aupr_ag": safe_aupr_np(stores["ag"]["true"], stores["ag"]["prob"]),
        "p5_all": precision_at_k(stores["all"]["true"], stores["all"]["prob"], stores["all"]["group"], k=5),
        "p10_all": precision_at_k(stores["all"]["true"], stores["all"]["prob"], stores["all"]["group"], k=10),
        "p5_ab": precision_at_k(stores["ab"]["true"], stores["ab"]["prob"], stores["ab"]["group"], k=5),
        "p10_ab": precision_at_k(stores["ab"]["true"], stores["ab"]["prob"], stores["ab"]["group"], k=10),
        "p5_ag": precision_at_k(stores["ag"]["true"], stores["ag"]["prob"], stores["ag"]["group"], k=5),
        "p10_ag": precision_at_k(stores["ag"]["true"], stores["ag"]["prob"], stores["ag"]["group"], k=10),
    }


def empty_stores():
    return {
        "all": {"true": [], "prob": [], "group": []},
        "ab": {"true": [], "prob": [], "group": []},
        "ag": {"true": [], "prob": [], "group": []},
    }


def train_one_epoch(model, loader, optimizer, scaler, device, config, epoch, class_weights):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    n_steps = 0
    stores = empty_stores()
    token_p = config["token_drop_p"]
    dropedge_p = float(np.random.uniform(config["dropedge_min"], config["dropedge_max"]))
    auc_alpha = ramp_value(epoch, config["aucpr_ramp_start"], config["aucpr_ramp_len"], config["aucpr_alpha"])

    for step, (edges, seqs, labels) in enumerate(loader):
        edges = edges.to(device, non_blocking=True)
        seqs = seqs.to(device, non_blocking=True).long()
        labels = labels.to(device, non_blocking=True).long()

        valid_mask, ab_mask, ag_mask = get_ab_ag_masks(seqs, labels, eoc_id=config["eoc_id"], pad_id=config["pad_id"])
        seqs_in = token_dropout(seqs, labels, token_p, pad_id=config["pad_id"], eoc_id=config["eoc_id"], replace_id=config["token_replace_id"])
        edges_in = dropedge_padded(edges, dropedge_p)

        with amp_autocast(config["amp"]):
            logits = squeeze_logits(model(seqs_in, edges_in))
            ce_flat = F.cross_entropy(
                logits.reshape(-1, config["num_classes"]),
                labels.reshape(-1),
                weight=class_weights,
                ignore_index=-1,
                reduction="none",
            ).reshape(labels.shape)

            residue_weight = torch.ones_like(ce_flat)
            residue_weight[ab_mask] = config["antibody_loss_weight"]
            residue_weight[ag_mask] = config["antigen_loss_weight"]
            residue_weight = residue_weight * valid_mask.float()

            ce_loss = (ce_flat * residue_weight).sum() / residue_weight.sum().clamp_min(1.0)
            smooth = graph_smooth_loss(logits, edges, labels) * config["smooth_lambda"]
            auc_loss = soft_pr_auc_loss(logits, labels, valid_mask, bins=config["aucpr_bins"], temp=config["aucpr_temp"]) * auc_alpha
            loss = ce_loss + smooth + auc_loss
            loss_bp = loss / config["accum_steps"]

        scaler.scale(loss_bp).backward()
        do_step = ((step + 1) % config["accum_steps"] == 0) or ((step + 1) == len(loader))
        if do_step:
            if config["max_grad_norm"] and config["max_grad_norm"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item())
        n_steps += 1
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)[..., 1]
            update_metric_lists(labels, probs, valid_mask, ab_mask, ag_mask, stores)

    metrics = summarize_stores(stores)
    metrics.update({
        "loss": total_loss / max(1, n_steps),
        "dropedge_p": dropedge_p,
        "token_p": token_p,
        "auc_alpha": auc_alpha,
    })
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, config, class_weights):
    model.eval()
    total_loss = 0.0
    n_steps = 0
    stores = empty_stores()

    for edges, seqs, labels in loader:
        edges = edges.to(device, non_blocking=True)
        seqs = seqs.to(device, non_blocking=True).long()
        labels = labels.to(device, non_blocking=True).long()
        valid_mask, ab_mask, ag_mask = get_ab_ag_masks(seqs, labels, eoc_id=config["eoc_id"], pad_id=config["pad_id"])

        with amp_autocast(config["amp"]):
            logits = squeeze_logits(model(seqs, edges))
            ce_flat = F.cross_entropy(
                logits.reshape(-1, config["num_classes"]),
                labels.reshape(-1),
                weight=class_weights,
                ignore_index=-1,
                reduction="none",
            ).reshape(labels.shape)
            residue_weight = torch.ones_like(ce_flat)
            residue_weight[ab_mask] = config["antibody_loss_weight"]
            residue_weight[ag_mask] = config["antigen_loss_weight"]
            residue_weight = residue_weight * valid_mask.float()
            ce_loss = (ce_flat * residue_weight).sum() / residue_weight.sum().clamp_min(1.0)
            smooth = graph_smooth_loss(logits, edges, labels) * config["smooth_lambda"]
            auc_loss = soft_pr_auc_loss(logits, labels, valid_mask, bins=config["aucpr_bins"], temp=config["aucpr_temp"]) * config["aucpr_alpha"]
            loss = ce_loss + smooth + auc_loss

        total_loss += float(loss.detach().item())
        n_steps += 1
        probs = torch.softmax(logits, dim=-1)[..., 1]
        update_metric_lists(labels, probs, valid_mask, ab_mask, ag_mask, stores)

    metrics = summarize_stores(stores)
    metrics["loss"] = total_loss / max(1, n_steps)
    return metrics


def label_stats(dataset, indices, name, pad_id=1, eoc_id=24):
    pos = neg = ignore = 0
    eoc_nonignore = 0
    pad_nonignore = 0
    for idx in indices:
        seq, lab, _ = dataset[idx]
        seq = torch.as_tensor(seq).long()
        q = seq[0]
        arr = np.asarray(lab)
        pos += int((arr == 1).sum())
        neg += int((arr == 0).sum())
        ignore += int((arr < 0).sum())
        if arr.ndim == 1 and arr.shape[0] == q.numel():
            q_np = q.cpu().numpy()
            eoc_nonignore += int((arr[q_np == eoc_id] >= 0).sum())
            pad_nonignore += int((arr[q_np == pad_id] >= 0).sum())
    valid = pos + neg
    frac = pos / valid if valid else 0
    print(f"{name}: valid={valid}, pos={pos}, neg={neg}, ignore={ignore}, pos_frac={frac:.5f}, eoc_nonignore={eoc_nonignore}, pad_nonignore={pad_nonignore}")


def make_warmup_cosine_scheduler(optimizer, config):
    warmup_epochs = int(config["warmup_epochs"])
    total_epochs = int(config["num_epochs"])
    min_lr = float(config["min_lr"])
    base_lr = float(config["learning_rate"])
    min_factor = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(epoch_idx):
        step_epoch = epoch_idx + 1
        if warmup_epochs > 0 and step_epoch <= warmup_epochs:
            return max(min_factor, step_epoch / float(warmup_epochs))
        denom = max(1, total_epochs - warmup_epochs)
        progress = (step_epoch - warmup_epochs) / float(denom)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_model(config):
    mod = importlib.import_module(config["model_module"])
    ClassificationModel = getattr(mod, "ClassificationModel")
    kwargs = dict(
        vocab_size=config["vocab_size"],
        seq_len=config["seq_len"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        num_layers=config["num_layers"],
        num_gnn_layers=config["num_gnn_layers"],
        num_int_layers=config["num_int_layers"],
        num_classes=config["num_classes"],
        drop_path_rate=config["drop_path_rate"],
    )
    if config["model_module"] != "Models_fullnew":
        kwargs.update(
            use_esm=config["use_esm"],
            esm_model_name=config["esm_model_name"],
            esm_repr_layer=config["esm_repr_layer"],
            esm_alpha_init=config["esm_alpha_init"],
            esm_adapter_dropout=config["esm_adapter_dropout"],
            esm_freeze=config["esm_freeze"],
            esm_max_len=config["esm_max_len"],
            pad_id=config["pad_id"],
            eoc_id=config["eoc_id"],
        )
    return ClassificationModel(**kwargs)


def print_esm_alpha_if_available(model):
    m = model.module if isinstance(model, nn.DataParallel) else model
    try:
        fusion = m.mc_model.cg_model.esm_query_fusion
    except Exception:
        return
    if fusion is None:
        return
    if hasattr(fusion, "get_alpha_values"):
        print("[esm] alpha values:", fusion.get_alpha_values())
    if hasattr(fusion, "get_adapter_trainable_param_counts"):
        print("[esm] adapter trainable param counts:", fusion.get_adapter_trainable_param_counts())


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_module", type=str, default="Models_fullnew")
    p.add_argument("--only_fold", type=int, default=None)
    p.add_argument("--out_dir", type=str, default="checkpoints_pretrained")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--num_gnn_layers", type=int, default=8)
    p.add_argument("--num_int_layers", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--drop_path_rate", type=float, default=0.10)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=5e-7)

    p.add_argument("--select_ab_weight", type=float, default=0.5)
    p.add_argument("--select_ag_weight", type=float, default=0.5)
    p.add_argument("--antibody_loss_weight", type=float, default=1.0)
    p.add_argument("--antigen_loss_weight", type=float, default=1.0)
    p.add_argument("--token_drop_p", type=float, default=0.0)
    p.add_argument("--token_replace_id", type=int, default=None)

    p.add_argument("--pretrained_ckpt", type=str, default=None)
    p.add_argument("--pretrained_ckpt_pattern", type=str, default=None)

    p.add_argument("--use_esm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--esm_model_name", type=str, default="esm2_t6_8M_UR50D")
    p.add_argument("--esm_repr_layer", type=int, default=None)
    p.add_argument("--esm_alpha_init", type=float, default=0.0)
    p.add_argument("--esm_adapter_dropout", type=float, default=0.10)
    p.add_argument("--esm_freeze", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--esm_max_len", type=int, default=1022)
    return p


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cudnn.benchmark = True

    config = {
        "model_module": args.model_module,
        "sequence_file": "para_tv_esmsequences_1600.npz",
        "data_file": "para_tv_esminterfaces_1600.npz",
        "edge_file": "para_tv_esmedges_1600.npz",
        "seq_len": 1600,
        "vocab_size": 31,
        "num_classes": 2,
        "pad_id": 1,
        "eoc_id": 24,
        "embed_dim": 256,
        "num_heads": 16,
        "num_layers": args.num_layers,
        "num_gnn_layers": args.num_gnn_layers,
        "num_int_layers": args.num_int_layers,
        "dropout": args.dropout,
        "drop_path_rate": args.drop_path_rate,
        "batch_size": 4,
        "num_epochs": 60,
        "warmup_epochs": 10,
        "min_stop_epoch": 50,
        "early_stop": 20,
        "learning_rate": args.learning_rate,
        "min_lr": args.min_lr,
        "weight_decay": 1e-4,
        "max_grad_norm": 0.5,
        "accum_steps": 1,
        "amp": True,
        "weight_start": 1.0,
        "weight_end": 1.0,
        "weight_anneal_tau": 8.0,
        "dropedge_min": 0.10,
        "dropedge_max": 0.20,
        "token_drop_p": args.token_drop_p,
        "token_replace_id": 3,
        "smooth_lambda": 0.01,
        "aucpr_alpha": 0.00,
        "aucpr_ramp_start": 10,
        "aucpr_ramp_len": 10,
        "aucpr_bins": 64,
        "aucpr_temp": 0.05,
        "antibody_loss_weight": args.antibody_loss_weight,
        "antigen_loss_weight": args.antigen_loss_weight,
        "n_splits": 10,
        "num_workers": args.num_workers,
        "select_ab_weight": args.select_ab_weight,
        "select_ag_weight": args.select_ag_weight,
        "use_esm": args.use_esm,
        "esm_model_name": args.esm_model_name,
        "esm_repr_layer": args.esm_repr_layer,
        "esm_alpha_init": args.esm_alpha_init,
        "esm_adapter_dropout": args.esm_adapter_dropout,
        "esm_freeze": args.esm_freeze,
        "esm_max_len": args.esm_max_len,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("Visible GPUs:", torch.cuda.device_count())
        print("GPU 0:", torch.cuda.get_device_name(0))

    print("Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    dataset = SequenceParatopeDataset(
        data_file=config["data_file"],
        sequence_file=config["sequence_file"],
        edge_file=config["edge_file"],
        max_len=config["seq_len"],
    )
    print("Dataset size:", len(dataset))

    kf = KFold(n_splits=config["n_splits"], shuffle=True, random_state=args.seed)
    all_indices = np.arange(len(dataset))

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(all_indices), 1):
        if args.only_fold is not None and fold != args.only_fold:
            continue

        print("\n" + "=" * 90)
        print(f"Fold {fold}/{config['n_splits']}")
        print("=" * 90)
        label_stats(dataset, tr_idx, "Train labels", pad_id=config["pad_id"], eoc_id=config["eoc_id"])
        label_stats(dataset, vl_idx, "Val labels", pad_id=config["pad_id"], eoc_id=config["eoc_id"])

        train_loader = DataLoader(
            Subset(dataset, tr_idx),
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(config["num_workers"] > 0),
            collate_fn=custom_collate_fn,
            drop_last=False,
        )
        val_loader = DataLoader(
            Subset(dataset, vl_idx),
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(config["num_workers"] > 0),
            collate_fn=custom_collate_fn,
            drop_last=False,
        )

        model = build_model(config).to(device)

        pretrained_ckpt = args.pretrained_ckpt
        if args.pretrained_ckpt_pattern is not None:
            pretrained_ckpt = args.pretrained_ckpt_pattern.format(fold=fold)
        load_pretrained_partial(model, pretrained_ckpt, device)

        if torch.cuda.device_count() > 1:
            print("Multiple GPUs visible; this script does not wrap DataParallel. Prefer CUDA_VISIBLE_DEVICES=0.")

        print_esm_alpha_if_available(model)
#        set_dym_trainable(model, False)
        set_dym_trainable(model, True)

        # Include all params from the start. Frozen params have no grad.
        optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        scheduler = make_warmup_cosine_scheduler(optimizer, config)
        scaler = make_grad_scaler(enabled=config["amp"])

        ckpt_name = (
            f"isiciPara_balanced_{config['model_module']}"
            f"_l{config['num_layers']}_g{config['num_gnn_layers']}_i{config['num_int_layers']}"
            f"_do{config['dropout']:.2f}_dpr{config['drop_path_rate']:.2f}"
            f"_lr{config['learning_rate']:.4g}_heads{config['num_heads']}_fold{fold}.pth"
        )
        ckpt_path = out_dir / ckpt_name

        best_metric = -1.0
        best_epoch = -1
        best_state = None
        patience = 0
        dym_unfrozen = False

        for epoch in range(1, config["num_epochs"] + 1):
            if epoch == 10 and not dym_unfrozen:
                print("Unfreezing DyM parameters.")
                set_dym_trainable(model, True)
                dym_unfrozen = True

            pw = config["weight_end"] + (config["weight_start"] - config["weight_end"]) * math.exp(-(epoch - 1) / config["weight_anneal_tau"])
            pw = max(config["weight_end"], float(pw))
            class_weights = torch.tensor([1.0, pw], dtype=torch.float32, device=device)

            train_m = train_one_epoch(model, train_loader, optimizer, scaler, device, config, epoch, class_weights)
            val_m = evaluate(model, val_loader, device, config, class_weights)
            select_metric = balanced_selection_metric(val_m, ab_weight=config["select_ab_weight"], ag_weight=config["select_ag_weight"])
            lr0 = optimizer.param_groups[0]["lr"]

            print(
                f"Ep{epoch:03d}  LR0={lr0:.2e}  pw={pw:.2f}  "
                f"aucα={train_m['auc_alpha']:.3f}  token_p={train_m['token_p']:.3f}  "
                f"dropE=({config['dropedge_min']:.3f},{config['dropedge_max']:.3f})  "
                f"smooth={config['smooth_lambda']:.3f}  "
                f"train_loss={train_m['loss']:.4f}  "
                f"train_AUPR_all={train_m['aupr_all']:.4f}  "
                f"train_AUPR_ab={train_m['aupr_ab']:.4f}  "
                f"train_AUPR_ag={train_m['aupr_ag']:.4f}  "
                f"train_P@5_ab={train_m['p5_ab']:.4f}  "
                f"train_P@5_ag={train_m['p5_ag']:.4f}  "
                f"val_loss={val_m['loss']:.4f}  "
                f"val_AUPR_all={val_m['aupr_all']:.4f}  "
                f"val_AUPR_ab={val_m['aupr_ab']:.4f}  "
                f"val_AUPR_ag={val_m['aupr_ag']:.4f}  "
                f"val_P@5_ab={val_m['p5_ab']:.4f}  "
                f"val_P@5_ag={val_m['p5_ag']:.4f}  "
                f"select={select_metric:.4f}"
            )

            improved = (not np.isnan(select_metric)) and (select_metric > best_metric + 1e-4)
            if improved:
                best_metric = float(select_metric)
                best_epoch = epoch
                best_state = get_model_state(model)
                patience = 0
                torch.save(best_state, ckpt_path)
                print(
                    f"  Saved checkpoint: {ckpt_path}  "
                    f"select={best_metric:.4f}  "
                    f"val_ab={val_m['aupr_ab']:.4f}  "
                    f"val_ag={val_m['aupr_ag']:.4f}  "
                    f"val_all={val_m['aupr_all']:.4f}"
                )
                print_esm_alpha_if_available(model)
            else:
                patience += 1

            scheduler.step()

            if epoch >= config["min_stop_epoch"] and patience >= config["early_stop"]:
                print(f"Early stopping at epoch {epoch}. Best select={best_metric:.4f} at epoch {best_epoch}.")
                break

        if best_state is not None:
            torch.save(best_state, ckpt_path)
            print(f"Final saved checkpoint for fold {fold}: {ckpt_path}")
            print(f"Best select={best_metric:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()

