import argparse
import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from Dataloader_ctm import SequenceParatopeDataset
from Models_fullnew import CTMModel


# ───────────────────────────────────────── Arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one selected cross-validation fold."
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Fold number to train, using 1-based numbering.",
    )
    return parser.parse_args()


args = parse_args()


# ───────────────────────────────────────── Configuration
config = {
    # Data / model
    "sequence_file": "para_tv_esmsequences_1600.npz",
    "data_file": "global_maps_para_esmtv.npz",
    "edge_file": "para_tv_esmedges_1600.npz",
    "max_len": 1600,
    "vocab_size": 31,
    "embed_dim": 256,
    "num_heads": 16,
    "dropout": 0.10,
    "num_layers": 1,
    "num_gnn_layers": 10,
    "num_int_layers": 5,
    "drop_path_rate": 0.10,
    "num_classes": 2,

    # Optimization
    "batch_size": 4,
    "num_epochs": 1000,
    "warmup_epochs": 10,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-2,
    "max_grad_norm": 0.1,
    "accum_steps": 1,

    # Cross-validation / early stopping
    "n_splits": 10,
    "early_stop": 20,
    "min_stop_epoch": 20,
    "random_seed": 42,

    # Optional pretrained core/model checkpoint.
    # Set to None to train from scratch.
    "pretrained_ckpt": (
        "isiPara_balanced_Models_fullnew_l1_g10_i5_"
        "do0.15_dpr0.15_lr0.0001_heads16_fold8_core.pth"
    ),

    # DataParallel support.
    # Control visible GPUs using CUDA_VISIBLE_DEVICES.
    "use_data_parallel": True,

    # DataLoader
    "num_workers": 0,

    # Output
    "output_dir": "checkpoints_para_isic",
    "prefix": "isicPara",
}


# ───────────────────────────────────────── Validation
if config["n_splits"] < 2:
    raise ValueError("n_splits must be at least 2.")

if not 1 <= args.fold <= config["n_splits"]:
    raise ValueError(
        f"--fold must be between 1 and {config['n_splits']}, "
        f"but received {args.fold}."
    )

if config["accum_steps"] < 1:
    raise ValueError("accum_steps must be at least 1.")

selected_fold = args.fold


# ───────────────────────────────────────── Environment setup
print("Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")

print(
    f"\nSelected fold: {selected_fold}/{config['n_splits']}"
)

os.makedirs(config["output_dir"], exist_ok=True)

torch.backends.cudnn.benchmark = True

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

num_gpus = torch.cuda.device_count()

print(
    f"Device: {device}; "
    f"number of GPUs visible: {num_gpus}"
)


# ───────────────────────────────────────── Reproducibility
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_random_seed(config["random_seed"])


# ───────────────────────────────────────── Checkpoint utilities
def get_model_state_from_checkpoint(
    path,
    map_location="cpu",
):
    checkpoint = torch.load(
        path,
        map_location=map_location,
    )

    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
            "model_state",
        ):
            if (
                key in checkpoint
                and isinstance(checkpoint[key], dict)
            ):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint does not contain a state dictionary: "
            f"{path}"
        )

    if checkpoint:
        first_key = next(iter(checkpoint.keys()))

        if first_key.startswith("module."):
            checkpoint = {
                key[len("module."):]: value
                for key, value in checkpoint.items()
            }

    return checkpoint


def safe_partial_load(
    model,
    checkpoint_path,
    device,
):
    if checkpoint_path is None:
        print("No pretrained checkpoint specified.")
        return

    if not os.path.exists(checkpoint_path):
        print(
            "WARNING: pretrained checkpoint not found: "
            f"{checkpoint_path}"
        )
        return

    raw_model = (
        model.module
        if isinstance(model, nn.DataParallel)
        else model
    )

    print(
        f"Loading pretrained checkpoint: "
        f"{checkpoint_path}"
    )

    source_state = get_model_state_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )

    destination_state = raw_model.state_dict()

    loaded = {}
    skipped_shape = []
    skipped_missing = []

    for key, value in source_state.items():
        normalized_key = (
            key[len("module."):]
            if key.startswith("module.")
            else key
        )

        if normalized_key not in destination_state:
            skipped_missing.append(normalized_key)
            continue

        source_shape = tuple(value.shape)
        destination_shape = tuple(
            destination_state[normalized_key].shape
        )

        if source_shape != destination_shape:
            skipped_shape.append(
                (
                    normalized_key,
                    source_shape,
                    destination_shape,
                )
            )
            continue

        loaded[normalized_key] = value

    destination_state.update(loaded)

    raw_model.load_state_dict(
        destination_state,
        strict=True,
    )

    print(f"Loaded keys: {len(loaded)}")
    print(
        f"Skipped missing keys: "
        f"{len(skipped_missing)}"
    )
    print(
        f"Skipped shape-mismatch keys: "
        f"{len(skipped_shape)}"
    )

    if skipped_missing[:10]:
        print(
            "First skipped missing keys:",
            skipped_missing[:10],
        )

    if skipped_shape[:10]:
        print(
            "First skipped shape mismatches:",
            skipped_shape[:10],
        )


def save_state(path, state):
    torch.save(state, path)
    print(f"Saved {path}")


def get_raw_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module

    return model


# ───────────────────────────────────────── Data utilities
def custom_collate_fn(batch):
    sequences, contact_maps, edges = zip(*batch)

    sequence_tensor = torch.stack(sequences)

    contact_map_tensor = torch.as_tensor(
        np.array(contact_maps),
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
            edge_tensor = torch.tensor(
                edge_index.T,
                dtype=torch.long,
            )

        edge_pad = -torch.ones(
            (2, max_edges),
            dtype=torch.long,
        )

        edge_pad[:, :edge_tensor.shape[1]] = (
            edge_tensor
        )

        padded_edges.append(edge_pad)

    padded_edges = torch.stack(padded_edges)

    return (
        padded_edges,
        sequence_tensor,
        contact_map_tensor,
    )


# ───────────────────────────────────────── Scheduler
def make_scheduler(optimizer):
    def lr_lambda(epoch):
        if epoch < config["warmup_epochs"]:
            return float(epoch + 1) / float(
                max(1, config["warmup_epochs"])
            )

        denominator = max(
            1,
            config["num_epochs"]
            - config["warmup_epochs"],
        )

        progress = float(
            epoch - config["warmup_epochs"]
        ) / float(denominator)

        progress = min(max(progress, 0.0), 1.0)

        return 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )


# ───────────────────────────────────────── Loss
def forward_ctm_loss(
    model,
    edges,
    sequences,
    labels,
    criterion,
):
    # CTMModel returns:
    # logits: [batch, classes, length, length]
    # attention: model-specific attention output
    logits, _ = model(
        sequences=sequences,
        padded_edges=edges,
        return_attention=True,
    )

    logits = (
        logits
        .permute(0, 2, 3, 1)
        .reshape(-1, config["num_classes"])
    )

    target = labels.reshape(-1)

    loss = criterion(logits, target)

    return (
        loss,
        logits.detach(),
        target.detach(),
    )


# ───────────────────────────────────────── Training
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
):
    model.train()

    total_loss = 0.0
    seen = 0
    valid_steps = 0

    optimizer.zero_grad(set_to_none=True)

    for step, (
        edges,
        sequences,
        labels,
    ) in enumerate(loader):
        edges = edges.to(
            device,
            non_blocking=True,
        )

        sequences = sequences.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        if (
            torch.isnan(labels.float()).any()
            or torch.isinf(labels.float()).any()
        ):
            print(
                "NaN/Inf found in target; "
                "skipping batch."
            )
            continue

        with autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available(),
        ):
            raw_loss, _, _ = forward_ctm_loss(
                model,
                edges,
                sequences,
                labels,
                criterion,
            )

            scaled_loss = (
                raw_loss / config["accum_steps"]
            )

        scaler.scale(scaled_loss).backward()

        valid_steps += 1

        should_step = (
            valid_steps % config["accum_steps"] == 0
        )

        if should_step:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["max_grad_norm"],
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        batch_size = sequences.shape[0]

        total_loss += (
            raw_loss.item() * batch_size
        )

        seen += batch_size

    # Perform the final optimizer step when the number
    # of valid batches is not divisible by accum_steps.
    if (
        valid_steps > 0
        and valid_steps % config["accum_steps"] != 0
    ):
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config["max_grad_norm"],
        )

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, seen)


# ───────────────────────────────────────── Validation
@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
):
    model.eval()

    total_loss = 0.0
    seen = 0

    for (
        edges,
        sequences,
        labels,
    ) in loader:
        edges = edges.to(
            device,
            non_blocking=True,
        )

        sequences = sequences.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        if (
            torch.isnan(labels.float()).any()
            or torch.isinf(labels.float()).any()
        ):
            print(
                "NaN/Inf found in target; "
                "skipping validation batch."
            )
            continue

        with autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available(),
        ):
            loss, _, _ = forward_ctm_loss(
                model,
                edges,
                sequences,
                labels,
                criterion,
            )

        batch_size = sequences.shape[0]

        total_loss += loss.item() * batch_size
        seen += batch_size

    return total_loss / max(1, seen)


# ───────────────────────────────────────── Dataset
dataset = SequenceParatopeDataset(
    data_file=config["data_file"],
    sequence_file=config["sequence_file"],
    edge_file=config["edge_file"],
    max_len=config["max_len"],
)

print(f"Dataset size: {len(dataset)}")

if len(dataset) < config["n_splits"]:
    raise ValueError(
        f"Dataset contains {len(dataset)} samples, "
        f"which is fewer than n_splits="
        f"{config['n_splits']}."
    )


# ───────────────────────────────────────── Fold generation
kfold = KFold(
    n_splits=config["n_splits"],
    shuffle=True,
    random_state=config["random_seed"],
)

dataset_indices = np.arange(len(dataset))

selected_train_indices = None
selected_val_indices = None

for fold, (
    train_indices,
    val_indices,
) in enumerate(
    kfold.split(dataset_indices),
    start=1,
):
    if fold == selected_fold:
        selected_train_indices = train_indices
        selected_val_indices = val_indices
        break

if (
    selected_train_indices is None
    or selected_val_indices is None
):
    raise RuntimeError(
        f"Unable to construct fold {selected_fold}."
    )

print(
    f"Fold {selected_fold}: "
    f"{len(selected_train_indices)} training samples, "
    f"{len(selected_val_indices)} validation samples."
)


# ───────────────────────────────────────── DataLoaders
train_loader = DataLoader(
    Subset(dataset, selected_train_indices),
    batch_size=config["batch_size"],
    collate_fn=custom_collate_fn,
    shuffle=True,
    num_workers=config["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    Subset(dataset, selected_val_indices),
    batch_size=config["batch_size"],
    collate_fn=custom_collate_fn,
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=torch.cuda.is_available(),
)


# ───────────────────────────────────────── Model
model = CTMModel(
    vocab_size=config["vocab_size"],
    seq_len=config["max_len"],
    embed_dim=config["embed_dim"],
    num_heads=config["num_heads"],
    dropout=config["dropout"],
    num_layers=config["num_layers"],
    num_gnn_layers=config["num_gnn_layers"],
    num_int_layers=config["num_int_layers"],
    drop_path_rate=config["drop_path_rate"],
    num_classes=config["num_classes"],
)

model = model.to(device)

safe_partial_load(
    model,
    config["pretrained_ckpt"],
    device=device,
)

if (
    config["use_data_parallel"]
    and num_gpus > 1
):
    print(
        f"Using {num_gpus} GPUs "
        "with DataParallel."
    )

    model = nn.DataParallel(model)


# ───────────────────────────────────────── Optimization
criterion = nn.CrossEntropyLoss(
    ignore_index=-1
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)

scheduler = make_scheduler(optimizer)

scaler = GradScaler(
    enabled=torch.cuda.is_available()
)


# ───────────────────────────────────────── Output names
base_name = (
    f"{config['prefix']}"
    f"_l{config['num_layers']}"
    f"_g{config['num_gnn_layers']}"
    f"_i{config['num_int_layers']}"
    f"_do{config['dropout']:.2f}"
    f"_dpr{config['drop_path_rate']:.2f}"
    f"_lr{config['learning_rate']}"
    f"_fold{selected_fold}"
)

best_model_path = os.path.join(
    config["output_dir"],
    base_name + "_best_val.pth",
)

final_model_path = os.path.join(
    config["output_dir"],
    base_name + "_final.pth",
)


# ───────────────────────────────────────── Main training loop
best_val = float("inf")
best_state = None
best_epoch = None
patience = 0

print(
    f"\nStarting Fold "
    f"{selected_fold}/{config['n_splits']}..."
)

for epoch in range(config["num_epochs"]):
    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        scaler,
    )

    val_loss = validate(
        model,
        val_loader,
        criterion,
    )

    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Fold {selected_fold} "
        f"Ep {epoch + 1:03d}/"
        f"{config['num_epochs']} "
        f"LR={current_lr:.2e} "
        f"train={train_loss:.4f} "
        f"val={val_loss:.4f}"
    )

    if val_loss < best_val:
        best_val = val_loss
        best_epoch = epoch + 1

        best_state = copy.deepcopy(
            get_raw_model(model).state_dict()
        )

        patience = 0

        print(
            f"New best validation loss: "
            f"{best_val:.4f}"
        )
    else:
        patience += 1

        print(
            f"No improvement. Patience: "
            f"{patience}/"
            f"{config['early_stop']}"
        )

    if (
        epoch + 1 >= config["min_stop_epoch"]
        and patience >= config["early_stop"]
    ):
        print(
            f"Early stopping at epoch "
            f"{epoch + 1}."
        )
        break


# ───────────────────────────────────────── Save models
if best_state is not None:
    save_state(
        best_model_path,
        best_state,
    )
else:
    print(
        "WARNING: no best model state was recorded."
    )

final_state = get_raw_model(model).state_dict()

save_state(
    final_model_path,
    final_state,
)


# ───────────────────────────────────────── Summary
print("\nTraining result:")
print(
    f"Fold: {selected_fold}/"
    f"{config['n_splits']}"
)
print(
    f"Best validation loss: "
    f"{best_val:.4f}"
)

if best_epoch is not None:
    print(f"Best epoch: {best_epoch}")

print(f"Best model: {best_model_path}")
print(f"Final model: {final_model_path}")
