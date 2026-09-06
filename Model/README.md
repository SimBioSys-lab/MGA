# MGA Model

This directory contains the MGA model implementation, Dynamic Mask module, dataloaders, training code, and dataset-level inference utilities.

Pretrained model weights and processed datasets are available from:

**https://zenodo.org/records/22525642**

---

## Files

### `Models_fullnew.py`

Main MGA model implementation. The model combines sequence/MSA representations with residue-level graph processing for binary residue-level antibody–antigen interface prediction.

### `Dynamic_Mask.py`

Implementation of Dynamic Mask (DyM). DyM learns input-dependent modulation of residue/token representations inside the MGA architecture.

### `Dataloader_itf.py`

Dataset loader for interface-prediction training. Inputs include sequence NPZ, interface-label NPZ, and edge NPZ files.

### `Dataloader_ctm.py`

Dataset loader used for related MGA structural/modeling tasks.

### `Train_model.py`

Main training script. The current implementation supports:

- 10-fold cross-validation
- mixed precision
- AdamW
- warmup + cosine LR scheduling
- graph edge dropout
- weighted interface loss
- AUPR evaluation
- antibody- and antigen-specific metrics
- optional pretrained checkpoint loading
- Dynamic Mask parameters

### `run_model.py`

Runs prediction for multiple structures from combined sequence and graph NPZ files.

---

## Data

Download processed datasets and pretrained weights from:

https://zenodo.org/records/22525642

See [`../data/README.md`](../data/README.md) for details.

---

## Training inputs

The current `Train_model.py` configuration expects:

```text
para_tv_esmsequences_1600.npz
para_tv_esminterfaces_1600.npz
para_tv_esmedges_1600.npz
```

The default model configuration includes:

```text
sequence length = 1600
vocabulary size = 31
embedding size = 256
attention heads = 16
classes = 2
```

---

## Basic training

From this directory:

```bash
python Train_model.py \
    --model_module Models_fullnew \
    --out_dir checkpoints
```

---

## Train a single fold

```bash
python Train_model.py \
    --model_module Models_fullnew \
    --only_fold 1 \
    --out_dir checkpoints
```

---

## Pretrained initialization

```bash
python Train_model.py \
    --model_module Models_fullnew \
    --pretrained_ckpt /path/to/checkpoint.pth \
    --out_dir checkpoints
```

A fold-dependent checkpoint pattern is also supported:

```bash
python Train_model.py \
    --pretrained_ckpt_pattern "/path/to/model_fold{fold}.pth"
```

---

## Selected command-line options

Architecture:

```text
--num_layers
--num_gnn_layers
--num_int_layers
--dropout
--drop_path_rate
```

Optimization:

```text
--learning_rate
--min_lr
--seed
```

Cross-validation:

```text
--only_fold
```

Pretraining:

```text
--pretrained_ckpt
--pretrained_ckpt_pattern
```

Output:

```text
--out_dir
```

Use:

```bash
python Train_model.py --help
```

for the complete current argument list.

---

## Interface masks

The query sequence is divided into antibody and antigen regions using EOC positions:

```text
Light chain → EOC → Heavy chain → EOC → Antigen chain(s)
```

The current special-token IDs are:

```text
PAD = 1
EOC = 24
```

Special tokens and ignored labels do not contribute to residue-level interface metrics.

---

## Evaluation

The training script calculates residue-level metrics separately for:

```text
ALL
AB
AG
```

corresponding to all valid residues, antibody residues, and antigen residues.

AUPR is used as a primary model-selection metric, and the script also computes precision-at-k metrics.

---

## Inference

Dataset-level inference can be run with:

```bash
python run_model.py \
    --csv /path/to/metadata.csv \
    --combined-seq-npz /path/to/sequences.npz \
    --combined-edges-npz /path/to/edges.npz \
    --preprocess-out-dir /path/to/output \
    --scripts-dir /path/to/MGA/data_preprocess
```

The inference pipeline extracts each requested complex from the combined NPZ files and passes it to the prediction utilities in `data_preprocess/`.

The pretrained checkpoint is available from:

https://zenodo.org/records/22525642

