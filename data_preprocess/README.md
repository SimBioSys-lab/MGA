Preprocessing scripts for MGA

Key scripts:
- `data_preprocess.py`: main preprocessing utilities (generate features and labels).
- `edge_gen.py`, `adj_gen.py`, `itf_gen.py`: helpers to build graph edges and interfaces.
- `combine_npzs.py`: combine per-PDB NPZs into dataset-wide NPZs.
- `run_pipeline.py` / `itf_pipeline.py`: example pipelines to run full preprocessing.
- `requirements.txt` / `environment.yml`: dependency lists for the preprocessing environment.


Run the full pipeline (`run_pipeline.py`)
--------------------------------------

`run_pipeline.py` is an end-to-end example that processes PDBs listed in a CSV and runs the full preprocessing -> MSA -> ESM seq+edges -> prediction + plot flow.

Basic example:

python run_pipeline.py \
  --pdb-dir /path/to/pdb_folder \
  --csv /path/to/metadata.csv \
  --hh-db-path /path/to/hhdb/Uniref30_2023_02 \
  --scripts-dir /path/to/data_preprocess \
  --work-dir /path/to/work_base \
  --keep-work-dir

CSV format
- The CSV must include:
  - **First column** (PDB_ID column, any name): the 4-letter PDB code
  - **`Light_chain`**: chain letter for light chain (e.g., `L`, `B`)
  - **`Heavy_chain`**: chain letter for heavy chain (e.g., `H`, `A`)
  - **`ag`**: antigen chain letter(s), separated by `;` if multiple (e.g., `G`, `I`, or `G;C`)
- Example minimal CSV (`metadata.csv`):

pdb_code,Light_chain,Heavy_chain,ag
4jan,B,A,I
2b4c,L,H,G;C

Where to put the PDB files and CSV
- `--pdb-dir`: point this to a directory that contains PDB files named like `<PDB_ID>.pdb` (case-insensitive: the script will try lower/upper variations). The script matches files to the first column (PDB_ID column) of your CSV.
- If a PDB file for a `PDB_ID` is not found in `--pdb-dir`, that entry will be skipped and reported as missing.
- `--csv`: can be anywhere on the filesystem; provide the path to your CSV file (e.g., `data/metadata.csv`).

Working directories and outputs
- By default the pipeline creates a base working directory at the parent of `--pdb-dir` and uses `tmp/<PDB_ID>` subfolders for each structure. Use `--work-dir` to override the base working directory.
- During a run the pipeline writes per-PDB artifacts (`esm_sequences.npz`, `edges_combined.npz`, prediction CSVs, plots) into the per-PDB work folder. Use `--keep-work-dir` to keep these folders for inspection; otherwise they are deleted when the run completes.

Other notes
- `--hh-db-path` must point to a local HHblits database (e.g., `Uniref30_2023_02`).
- `--scripts-dir` should point to this `data_preprocess/` folder (or another folder that contains the required helper scripts such as `process_pdb.py`, `adj_gen.py`, `run_hh.py`, `seq_gen.py`, `edge_gen.py`, `make_prediction.py`, `plot_pred_from_npz.py`).
- Inspect `run_pipeline.py` for more flags (e.g., `--esm-model-name`, `--eoc-id`).
