# MGA

MGA is a deep-learning framework for residue-level antibody–antigen interface prediction using multiple-sequence-alignment (MSA) representations and residue-level graph information.

The repository contains the MGA model architecture, Dynamic Mask (DyM) module, training and inference code, and preprocessing utilities for converting antibody–antigen structures into model-ready sequence and graph representations.

Large processed datasets and pretrained model weights are hosted separately on Zenodo:

**Data and pretrained weights:**  
https://zenodo.org/records/22525642

---

## Repository structure

```text
MGA/
├── Model/
│   ├── Models_fullnew.py
│   ├── Dynamic_Mask.py
│   ├── Dataloader_itf.py
│   ├── Dataloader_ctm.py
│   ├── Train_model.py
│   ├── run_model.py
│   └── README.md
├── data/
│   └── README.md
├── data_preprocess/
│   ├── data_preprocess.py
│   ├── process_pdb.py
│   ├── run_hh.py
│   ├── seq_gen.py
│   ├── edge_gen.py
│   ├── adj_gen.py
│   ├── itf_gen.py
│   ├── Extract_interface.py
│   ├── combine_npzs.py
│   ├── make_prediction.py
│   ├── plot_pred_from_npz.py
│   ├── environment.yml
│   └── README.md
└── README.md
```

---

## Data and pretrained model

The processed `.npz` datasets and pretrained `.pth` model weights are available from Zenodo:

**https://zenodo.org/records/22525642**

These large binary files are not stored directly in this GitHub repository.

See [`data/README.md`](data/README.md) for download and file-placement instructions.

---

## Installation

An environment specification for preprocessing is provided in:

```text
data_preprocess/environment.yml
```

Create the environment with:

```bash
conda env create -f data_preprocess/environment.yml
```

Then activate the environment using the environment name specified in the YAML file.

The model is implemented in PyTorch. A CUDA-capable GPU is recommended for training and large-scale inference.

---

## Model inputs

MGA uses two primary input representations:

1. **MSA-derived sequence representation**
2. **Residue-level graph edges**

For supervised training, residue-level interface labels are also required.

The current training script expects:

```text
para_tv_esmsequences_1600.npz
para_tv_esminterfaces_1600.npz
para_tv_esmedges_1600.npz
```

The maximum modeled sequence length is 1600, and the query sequence is stored in the first row of the MSA representation.

---

## Sequence organization

Antibody and antigen chains are concatenated using an end-of-chain token:

```text
Light chain → EOC → Heavy chain → EOC → Antigen chain(s) → PAD
```

The current token convention is:

```text
PAD token ID = 1
EOC token ID = 24
```

For multiple antigen chains, additional EOC-separated antigen segments may be present. EOC and PAD positions are excluded from residue-level training losses and evaluation metrics.

---

## Metadata CSV

The preprocessing pipeline requires a CSV defining the antibody and antigen chains for each complex.

```csv
pdb_code,Light_chain,Heavy_chain,ag
4jan,B,A,I
2b4c,L,H,G;C
```

Required fields:

- first column: PDB identifier
- `Light_chain`: antibody light-chain ID
- `Heavy_chain`: antibody heavy-chain ID
- `ag`: antigen chain ID or IDs

Multiple antigen chains should be separated by `;`, for example `G;C`.

---

## Preprocessing

Preprocessing scripts are located in `data_preprocess/`.

| Script | Purpose |
|---|---|
| `process_pdb.py` | Process input PDB structures |
| `run_hh.py` | Generate MSAs using HHblits |
| `seq_gen.py` | Generate sequence/MSA representations |
| `edge_gen.py` | Construct residue graph edges |
| `adj_gen.py` | Generate adjacency-related representations |
| `itf_gen.py` | Generate interface information |
| `Extract_interface.py` | Extract residue-level interfaces |
| `combine_npzs.py` | Combine per-complex NPZ files |
| `make_prediction.py` | Run MGA inference |
| `plot_pred_from_npz.py` | Visualize predictions |

See [`data_preprocess/README.md`](data_preprocess/README.md) for detailed instructions.

---

## Model architecture

The main MGA architecture is implemented in `Model/Models_fullnew.py`.

The framework combines sequence representations with residue-level graph message passing for antibody–antigen interface prediction.

The Dynamic Mask implementation is located in `Model/Dynamic_Mask.py`. DyM provides learned, input-dependent modulation of residue/token representations.

See [`Model/README.md`](Model/README.md) for additional details.

---

## Training

Training is implemented in `Model/Train_model.py`.

The current training pipeline supports:

- 10-fold cross-validation
- mixed-precision training
- AdamW optimization
- warmup and cosine learning-rate scheduling
- graph edge dropout
- class weighting for interface imbalance
- antibody- and antigen-specific evaluation
- AUPR-based model selection
- optional pretrained checkpoint initialization
- Dynamic Mask training

A basic training run from the `Model/` directory is:

```bash
python Train_model.py \
    --model_module Models_fullnew \
    --out_dir checkpoints
```

To train a single fold:

```bash
python Train_model.py \
    --model_module Models_fullnew \
    --only_fold 1 \
    --out_dir checkpoints
```

To initialize from an existing checkpoint:

```bash
python Train_model.py \
    --model_module Models_fullnew \
    --pretrained_ckpt /path/to/model.pth \
    --out_dir checkpoints
```

---

## Inference

`Model/run_model.py` supports running prediction for multiple complexes from combined NPZ files.

```bash
python Model/run_model.py \
    --csv /path/to/metadata.csv \
    --combined-seq-npz /path/to/esm_sequences_all.npz \
    --combined-edges-npz /path/to/edges_combined_all.npz \
    --preprocess-out-dir /path/to/prediction_output \
    --scripts-dir /path/to/MGA/data_preprocess \
    --sample-idx 0
```

For each PDB ID, the script extracts the corresponding sequence and graph representations, creates per-complex NPZ files, runs interface prediction, writes residue-level probabilities, and generates prediction plots.

The pretrained checkpoint can be downloaded from:

https://zenodo.org/records/22525642

The current `data_preprocess/make_prediction.py` defines the checkpoint through `MODEL_FILE`. Change this value to the path of the downloaded `.pth` checkpoint before running inference.

---

## Prediction output

Residue-level prediction CSV files contain fields such as:

```text
SampleIdx
Key
ChainType
Position
PredLabel
Probability
```

`ChainType` distinguishes `Lchain`, `Hchain`, `AGchain_0`, `AGchain_1`, and additional antigen chains when present.

---

## Reproducibility

For reproducible experiments, record the MGA commit/version, pretrained checkpoint, dataset version, train/validation split, random seed, model hyperparameters, and preprocessing parameters.

The large processed data and model checkpoints associated with this repository are archived at:

https://zenodo.org/records/22525642

---

## Citation

Citation information for the MGA manuscript will be added when the corresponding publication information is finalized.

When using the archived data or pretrained weights, please also cite the associated Zenodo record.

---

## License

Please follow the licensing terms associated with MGA and with any third-party structural, sequence, or pretrained-model resources used by the preprocessing pipeline.

