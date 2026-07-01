import math
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from Dataloader_itf import SequenceParatopeDataset
from Models_fullnew import ClassificationModel


# ───────────────────────────────────────── CONFIG
config = {
    # data / model
    "sequence_file": "para_tv_esmsequences_1600.npz",
    "data_file": "para_tv_esmss_1600.npz",
    "edge_file": "para_tv_esmedges_1600.npz",
    "vocab_size": 31,
    "seq_len": 1600,
    "embed_dim": 256,
    "num_heads": 16,
    "dropout": 0.10,
    "num_layers": 1,
    "num_gnn_layers": 10,
    "num_int_layers": 5,
    "drop_path_rate": 0.10,
    "num_classes": 8,

    # optimization
    "batch_size": 4,
    "num_epochs": 100,
    "warmup_epochs": 10,
    "learning_rate": 1.0e-4,     # CTSR fine-tuning default
    "weight_decay": 1.0e-2,
    "max_grad_norm": 0.1,
    "accum_steps": 1,

    # CV / early stop
    "n_splits": 5,
    "early_stop": 20,
    "min_stop_epoch": 20,

    # optional pretrained core/model checkpoint. Use None to train from scratch.
    "pretrained_ckpt": "iPara_balanced_l1_g10_i5_do0.10_dpr0.10_lr0.0002_heads16_fold4_core.pth",
    # DataParallel support for CTSR. Manually control GPUs with CUDA_VISIBLE_DEVICES.
    "use_data_parallel": True,

    # output
    "output_dir": "ctsr_ss_checkpoints",
    "prefix": "ctsr_ss",
}

print(config)
os.makedirs(config["output_dir"], exist_ok=True)

torch.backends.cudnn.benchmark = True
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Device: {device}; number of GPUs visible: {num_gpus}")


# ───────────────────────────────────────── Utils
def get_model_state_from_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)

    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model", "model_state"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {path}")

    if len(ckpt) > 0 and next(iter(ckpt.keys())).startswith("module."):
        ckpt = {k[len("module."):]: v for k, v in ckpt.items()}

    return ckpt


def safe_partial_load(model, ckpt_path, device):
    if ckpt_path is None:
        print("No pretrained checkpoint specified.")
        return

    if not os.path.exists(ckpt_path):
        print(f"WARNING: pretrained checkpoint not found: {ckpt_path}")
        return

    print(f"Loading pretrained checkpoint: {ckpt_path}")
    src = get_model_state_from_checkpoint(ckpt_path, map_location=device)
    dst = model.state_dict()

    loaded = {}
    skipped_shape = []
    skipped_missing = []

    for k, v in src.items():
        key = k[len("module."):] if k.startswith("module.") else k
        if key not in dst:
            skipped_missing.append(key)
            continue
        if tuple(dst[key].shape) != tuple(v.shape):
            skipped_shape.append((key, tuple(v.shape), tuple(dst[key].shape)))
            continue
        loaded[key] = v

    dst.update(loaded)
    model.load_state_dict(dst, strict=True)

    print(f"Loaded keys: {len(loaded)}")
    print(f"Skipped missing keys: {len(skipped_missing)}")
    print(f"Skipped shape-mismatch keys: {len(skipped_shape)}")
    if skipped_missing[:10]:
        print("First skipped missing:", skipped_missing[:10])
    if skipped_shape[:10]:
        print("First skipped shape mismatches:", skipped_shape[:10])


def custom_collate_fn(batch):
    seqs, labels, edges = zip(*batch)

    seqs = torch.stack(seqs)
    labels = torch.as_tensor(np.array(labels), dtype=torch.long)

    max_edges = max(edge_index.shape[0] for edge_index in edges)
    padded_edges = []

    for edge_index in edges:
        if torch.is_tensor(edge_index):
            ei = edge_index.T.clone().detach().long()
        else:
            ei = torch.tensor(edge_index.T, dtype=torch.long)

        edge_pad = -torch.ones((2, max_edges), dtype=torch.long)
        edge_pad[:, :ei.shape[1]] = ei
        padded_edges.append(edge_pad)

    padded_edges = torch.stack(padded_edges)
    return padded_edges, seqs, labels


def make_scheduler(optimizer):
    def lr_lambda(epoch):
        if epoch < config["warmup_epochs"]:
            return float(epoch + 1) / float(max(1, config["warmup_epochs"]))

        denom = max(1, config["num_epochs"] - config["warmup_epochs"])
        progress = float(epoch - config["warmup_epochs"]) / float(denom)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def forward_ss_loss(model, edges, sequences, labels, criterion):
    # ClassificationModel returns logits [B, 1, L, C].
    logits = model(
        sequences=sequences,
        padded_edges=edges,
        return_attention=False,
    )

    logits = logits.reshape(-1, config["num_classes"])
    target = labels.reshape(-1)
    return criterion(logits, target), logits.detach(), target.detach()


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    seen = 0

    optimizer.zero_grad(set_to_none=True)

    for step, (edges, sequences, labels) in enumerate(loader):
        edges = edges.to(device, non_blocking=True)
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if torch.isnan(labels.float()).any() or torch.isinf(labels.float()).any():
            print("NaN/Inf found in target; skipping batch.")
            continue

        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            loss, _, _ = forward_ss_loss(model, edges, sequences, labels, criterion)
            loss = loss / config["accum_steps"]

        scaler.scale(loss).backward()

        if (step + 1) % config["accum_steps"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = sequences.shape[0]
        total_loss += loss.item() * config["accum_steps"] * bs
        seen += bs

    return total_loss / max(1, seen)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    seen = 0

    for edges, sequences, labels in loader:
        edges = edges.to(device, non_blocking=True)
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if torch.isnan(labels.float()).any() or torch.isinf(labels.float()).any():
            print("NaN/Inf found in target; skipping batch.")
            continue

        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            loss, _, _ = forward_ss_loss(model, edges, sequences, labels, criterion)

        bs = sequences.shape[0]
        total_loss += loss.item() * bs
        seen += bs

    return total_loss / max(1, seen)


def save_state(path, state):
    torch.save(state, path)
    print(f"Saved {path}")


# ───────────────────────────────────────── Dataset & CV
dataset = SequenceParatopeDataset(
    data_file=config["data_file"],
    sequence_file=config["sequence_file"],
    edge_file=config["edge_file"],
    max_len=config["seq_len"],
)

kf = KFold(n_splits=config["n_splits"], shuffle=True, random_state=42)
dataset_indices = np.arange(len(dataset))
fold_results = []


# ───────────────────────────────────────── Main loop
for fold, (train_indices, val_indices) in enumerate(kf.split(dataset_indices), 1):
    print(f"\nStarting Fold {fold}/{config['n_splits']}...")

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config["batch_size"],
        collate_fn=custom_collate_fn,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=config["batch_size"],
        collate_fn=custom_collate_fn,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = ClassificationModel(
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

    if config["use_data_parallel"] and num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    model = model.to(device)
    safe_partial_load(model, config["pretrained_ckpt"], device=device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = make_scheduler(optimizer)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(config["num_epochs"]):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss = validate(model, val_loader, criterion)

        scheduler.step()

        print(
            f"Fold {fold} Ep {epoch + 1:03d}/{config['num_epochs']} "
            f"LR={optimizer.param_groups[0]['lr']:.2e} "
            f"train={tr_loss:.4f} val={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.module.state_dict() if hasattr(model, "module") else model.state_dict())
            patience = 0
            print(f"New best model val loss: {best_val:.4f}")
        else:
            patience += 1
            print(f"No improvement. Patience: {patience}/{config['early_stop']}")

        if epoch + 1 >= config["min_stop_epoch"] and patience >= config["early_stop"]:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    base = (
        f"{config['prefix']}_l{config['num_layers']}_g{config['num_gnn_layers']}"
        f"_i{config['num_int_layers']}_do{config['dropout']:.2f}"
        f"_dpr{config['drop_path_rate']:.2f}_lr{config['learning_rate']}"
        f"_fold{fold}"
    )

    if best_state is not None:
        save_state(os.path.join(config["output_dir"], base + "_best_val.pth"), best_state)

    final_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    save_state(os.path.join(config["output_dir"], base + "_final.pth"), final_state)

    fold_results.append(best_val)

print("\nCross-validation results:")
for fold, val in enumerate(fold_results, 1):
    print(f"Fold {fold}: Best Validation Loss = {val:.4f}")
print(f"Average Validation Loss: {np.mean(fold_results):.4f}")

