Model code and training

Contents:
- `Models_fullnew.py`: model architectures used for interface prediction.
- `Dataloader_itf.py`, `Dataloader_pred.py`: dataset and dataloader implementations.
- `run_model.py`: script for training and/or inference.
- `isicisiParamodelnewesm_l1_g10_i5_do0.30_dpr0.25_lr0.0001_fold10.pth`: example pretrained weights.
- `PT_MG_TV_reg.py`, `Dynamic_Mask.py`: training utilities and regularization helpers.

Training
--------

`PT_MG_TV_reg.py` implements the training loop with k-fold cross-validation, regularization, and AUCPR loss.

To train:
1. Edit the `config` dictionary in `PT_MG_TV_reg.py` to set data file paths and hyperparameters:
   - `sequence_file`, `data_file`, `edge_file`: paths to preprocessed NPZ files (typically in `data/`).
   - `num_epochs`, `learning_rate`, `batch_size`, etc.: model and optimization hyperparameters.
   - `n_splits`: number of k-fold cross-validation splits (default: 10).

2. Run the training script:

```
python PT_MG_TV_reg.py
```

The script trains a model per fold, applies regularization (drop-edge, token dropout, region smoothing), uses an AUCPR surrogate loss, and logs metrics to stdout.

Example config snippet:
```python
config = {
    'sequence_file': 'para_tv_esmsequences_1600.npz',
    'data_file': 'para_tv_esminterfaces_1600.npz',
    'edge_file': 'para_tv_esmedges_1600.npz',
    'batch_size': 4,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'n_splits': 10,
    # ... more hyperparameters
}
```

After training, model checkpoints are saved (filename pattern includes hyperparameters, e.g., `isicisiParamodelnewesm_l1_g10_i5_do0.30_dpr0.25_lr0.0001_fold10.pth`).

Prediction
----------

`run_model.py` runs prediction on PDBs listed in a CSV, using combined ESM sequences and edge NPZs.

Basic usage:

```
python run_model.py \
  --csv /path/to/metadata.csv \
  --combined-seq-npz /path/to/combined_esm_sequences.npz \
  --combined-edges-npz /path/to/combined_edges_combined.npz \
  --preprocess-out-dir /path/to/output \
  --scripts-dir /path/to/data_preprocess
```

Arguments:
- `--csv`: Metadata CSV with first column as `PDB_ID` and additional columns (`Light_chain`, `Heavy_chain`, `ag`).
- `--combined-seq-npz`: Combined ESM sequence embeddings for all PDBs (keys are `PDB_ID`).
- `--combined-edges-npz`: Combined edge lists for all PDBs (keys are `PDB_ID`).
- `--preprocess-out-dir`: Output directory where per-PDB predictions and plots will be written.
- `--scripts-dir`: Path to `data_preprocess/` folder (contains `make_prediction.py`, `plot_pred_from_npz.py`).
- `--esm-model-name`: ESM model name (default: `esm2_t6_8M_UR50D`).
- `--eoc-id`: EOC token ID (default: 24).
- `--sample-idx`: Sample index for visualization (default: 0).

Output:
- Per-PDB predictions are written to `<preprocess-out-dir>/<PDB_ID>/`.
- CSV with predicted paratope residues and a prediction grid plot are generated for each structure.

Notes:
- Ensure trained model checkpoint is available (place in Model folder or specify path if needed).
- Dependencies: PyTorch, numpy, scikit-learn. See `data_preprocess/requirements.txt` for the full list.
