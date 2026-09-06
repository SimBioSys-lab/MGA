# MGA Data and Pretrained Model

Large MGA data files and pretrained model checkpoints are hosted on Zenodo rather than stored directly in this GitHub repository.

## Download

All released data and model files are available at:

**https://zenodo.org/records/22525642**

Please use the Zenodo record as the authoritative source for the downloadable files and their current versions.

---

## Why are the files not stored in GitHub?

MGA uses large binary files including:

```text
*.npz
*.pth
```

These files contain processed sequence/graph data and pretrained PyTorch model parameters and are therefore stored externally on Zenodo.

---

## Recommended local organization

After downloading the files from Zenodo, a convenient local organization is:

```text
MGA/
├── data/
│   ├── README.md
│   ├── <downloaded sequence NPZ files>
│   ├── <downloaded edge NPZ files>
│   ├── <downloaded interface NPZ files>
│   └── <other processed data>
├── checkpoints/
│   └── <downloaded pretrained model>.pth
├── Model/
└── data_preprocess/
```

The files do not need to be stored inside the Git repository itself; absolute paths can also be supplied to the relevant scripts.

---

## Training data

Supervised MGA training uses three types of processed files.

### Sequence/MSA representation

Example expected filename:

```text
para_tv_esmsequences_1600.npz
```

This stores MSA-derived tokenized sequence representations. The maximum sequence length is 1600, and the first MSA row corresponds to the query sequence.

### Interface labels

Example expected filename:

```text
para_tv_esminterfaces_1600.npz
```

Residue labels use:

```text
0   non-interface residue
1   interface residue
-1  ignored/padded position
```

EOC and PAD positions should not contribute to the residue-level interface loss.

### Graph edges

Example expected filename:

```text
para_tv_esmedges_1600.npz
```

This stores graph connectivity used by the graph component of MGA.

---

## Token conventions

The processed MSA representation currently uses:

```text
PAD = 1
EOC = 24
```

The query sequence is organized approximately as:

```text
Light chain → EOC → Heavy chain → EOC → Antigen chain(s) → PAD
```

---

## Pretrained model

The pretrained MGA PyTorch checkpoint is also distributed through:

https://zenodo.org/records/22525642

After downloading the checkpoint, it can be stored locally, for example:

```text
MGA/checkpoints/model.pth
```

The current inference script `data_preprocess/make_prediction.py` contains:

```python
MODEL_FILE = "..."
```

Set `MODEL_FILE` to the path of the downloaded checkpoint.

---

## Inference data

`Model/run_model.py` can operate on combined sequence and edge NPZ files:

```bash
python Model/run_model.py \
    --csv metadata.csv \
    --combined-seq-npz /path/to/sequence_data.npz \
    --combined-edges-npz /path/to/edge_data.npz \
    --preprocess-out-dir predictions \
    --scripts-dir data_preprocess
```

The combined NPZ files are interpreted as mappings from a PDB/sample identifier to the corresponding sequence or graph representation.

---

## Inspecting an NPZ file

```python
import numpy as np

data = np.load("file.npz", allow_pickle=True)
print(data.files)

for key in data.files[:5]:
    value = data[key]
    print(key, type(value), getattr(value, "shape", None))
```

---

## Data provenance

The files on Zenodo are processed representations used by MGA.

Users interested in reproducing the representations from structures should use the scripts in `data_preprocess/`.

See [`../data_preprocess/README.md`](../data_preprocess/README.md).

---

## Citation

When using these files, please cite the associated Zenodo record:

https://zenodo.org/records/22525642

Citation information for the corresponding MGA publication will be added to the main repository README when finalized.

