# MGA

MGA is a codebase for predicting protein–protein interface residues using sequence and graph representations.

Repository overview:
- data/: processed and raw input datasets (npz, csv, per-PDB folders).
- data_preprocess/: scripts to build inputs (edges, sequences, interfaces) and run preprocessing pipelines.
- Model/: model definitions, dataloaders, training and inference scripts, and example weights.
- temp/: temporary artifacts created while preparing or testing data and models.

Quick start:
1. Prepare data using the scripts in `data_preprocess/` (see that folder for usage examples).
2. Train or run inference with scripts in `Model/` such as `run_model.py`.

See the README files in each subfolder for details on the files and scripts they contain.
