# MGA Data Preprocessing

This directory contains scripts for converting antibody–antigen structures into the processed sequence, graph, and interface representations required by MGA.

Pre-generated data are available from:

**https://zenodo.org/records/22525642**

Users who only want to run MGA with the released processed files do not need to rerun the complete preprocessing pipeline.

---

## Overview

```text
PDB structure
     │
     ▼
structure processing
     ├──────────────► interface labels
     ▼
sequence extraction
     ▼
MSA generation
     ▼
MSA/token representation ──────────────► sequence NPZ
     ▼
residue graph construction ────────────► edge NPZ
```

The resulting representations can be used for training or inference.

---

## Important scripts

- `data_preprocess.py`: general preprocessing utilities.
- `process_pdb.py`: processes antibody–antigen PDB structures.
- `run_hh.py`: runs HHblits to generate MSAs.
- `seq_gen.py`: converts sequence/MSA information into MGA token representations.
- `edge_gen.py`: constructs graph edges.
- `adj_gen.py`: generates adjacency-related representations.
- `itf_gen.py`: generates residue-level interface information.
- `Extract_interface.py`: extracts antibody–antigen interface residues.
- `combine_npzs.py`: combines per-complex NPZ files into dataset-level NPZ files.
- `make_prediction.py`: runs MGA prediction from processed sequence and edge representations.
- `plot_pred_from_npz.py`: generates plots from prediction output.

The pretrained `.pth` model can be downloaded from:

https://zenodo.org/records/22525642

Before running `make_prediction.py`, set:

```python
MODEL_FILE = "/path/to/downloaded/model.pth"
```

to the local checkpoint path.

---

## Environment

An environment specification is provided as:

```text
environment.yml
```

Create the environment with:

```bash
conda env create -f environment.yml
```

Then activate the environment specified in the YAML file.

---

## Input structures

Input structures should be supplied as PDB files. A metadata CSV identifies the antibody and antigen chains.

```csv
pdb_code,Light_chain,Heavy_chain,ag
4jan,B,A,I
2b4c,L,H,G;C
```

Required columns are:

- `Light_chain`
- `Heavy_chain`
- `ag`

Multiple antigen chains can be specified using `;`, for example `G;C`.

---

## End-to-end preprocessing

A representative workflow uses:

```bash
python run_pipeline.py \
    --pdb-dir /path/to/pdb_folder \
    --csv /path/to/metadata.csv \
    --hh-db-path /path/to/Uniref30_2023_02 \
    --scripts-dir /path/to/MGA/data_preprocess \
    --work-dir /path/to/work_directory \
    --keep-work-dir
```

Use:

```bash
python run_pipeline.py --help
```

to inspect the currently available arguments.

---

## PDB directory

`--pdb-dir` should contain files corresponding to the IDs in the first column of the metadata CSV.

```text
structures/
├── 4jan.pdb
├── 2b4c.pdb
└── ...
```

Missing structures are reported and skipped.

---

## HHblits database

MSA generation requires a local HHblits database, for example:

```text
Uniref30_2023_02
```

Supply its location with:

```bash
--hh-db-path /path/to/Uniref30_2023_02
```

The HHblits database is not distributed with this repository.

---

## Intermediate files

Depending on the selected pipeline, preprocessing can generate per-complex files such as:

```text
esm_sequences.npz
edges_combined.npz
```

and additional sequence, interface, MSA, or graph intermediates.

When `--keep-work-dir` is enabled, per-complex working directories are preserved for inspection.

---

## Combined NPZ files

For dataset-level inference, per-complex representations can be combined using `combine_npzs.py` and supplied to `Model/run_model.py`.

```bash
python ../Model/run_model.py \
    --csv metadata.csv \
    --combined-seq-npz esm_sequences_all.npz \
    --combined-edges-npz edges_combined_all.npz \
    --preprocess-out-dir predictions \
    --scripts-dir .
```

---

## Token conventions

The current MGA preprocessing and model code use:

```text
PAD token ID = 1
EOC token ID = 24
```

The first MSA row contains the query sequence, organized approximately as:

```text
Light chain ... EOC
Heavy chain ... EOC
Antigen chain(s) ...
PAD
```

EOC and PAD positions are ignored during residue-level interface evaluation.

---

## Pre-generated data

Users who do not need to reproduce preprocessing can download the processed files directly from:

https://zenodo.org/records/22525642

See [`../data/README.md`](../data/README.md) for additional information.

---

## Prediction output

`make_prediction.py` produces residue-level probabilities and labels. Typical output fields include:

```text
SampleIdx
Key
ChainType
Position
PredLabel
Probability
```

Chain types include `Lchain`, `Hchain`, `AGchain_0`, `AGchain_1`, and additional antigen chains when present.

