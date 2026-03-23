Data directory for MGA

Contents:
- combined_edges_combined.npz: combined graph edge data used by the model.
- combined_esm_sequences.npz: ESM-derived sequence embeddings for combined dataset.
- interfaces_combined.npz: combined interface labels.
- train_set.csv: CSV listing training examples and metadata.
- Per-PDB subfolders (e.g. `2b4c/`, `4jan/`): contain per-structure `edges_combined.npz` and `esm_sequences.npz`.
- interfaces_raw/: original interface files (format: `<pdb>_chain_X_Y_interface4.5`).

Notes:
- NPZ files store numpy arrays expected by `Model` dataloaders.
- Use scripts in `data_preprocess/` to regenerate or extend these files.
