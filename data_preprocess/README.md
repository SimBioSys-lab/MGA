Preprocessing scripts for MGA

Key scripts:
- `data_preprocess.py`: main preprocessing utilities (generate features and labels).
- `edge_gen.py`, `adj_gen.py`, `itf_gen.py`: helpers to build graph edges and interfaces.
- `combine_npzs.py`: combine per-PDB NPZs into dataset-wide NPZs.
- `run_pipeline.py` / `itf_pipeline.py`: example pipelines to run full preprocessing.
- `requirements.txt` / `environment.yml`: dependency lists for the preprocessing environment.

Usage:
- Inspect individual scripts for argument details. Typical flow:
  1. Extract per-PDB features (ESM embeddings, edges).
  2. Convert raw interface files into label arrays.
  3. Combine per-PDB NPZs into dataset NPZs consumed by the model.
