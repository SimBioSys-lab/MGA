'''
python run_pred.py \
  --csv /projects/SimBioSys/Xing/MGA/data/train_set.csv \
  --combined-seq-npz /projects/SimBioSys/Xing/MGA/data/esm_sequences_all.npz \
  --combined-edges-npz /projects/SimBioSys/Xing/MGA/data/edges_combined_all.npz \
  --preprocess-out-dir /projects/SimBioSys/Xing/MGA/pred_runs \
  --scripts-dir /projects/SimBioSys/Xing/MGA/data_preprocess \
  --sample-idx 0
'''
#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import traceback
import csv

import numpy as np


def run(cmd, cwd=None):
    print(f">>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def read_pdb_ids_from_csv_first_col(csv_path):
    pdb_ids = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header / no columns: {csv_path}")
        pdb_col = reader.fieldnames[0]
        for row in reader:
            val = row.get(pdb_col)
            if val is None:
                continue
            pdb_id = str(val).strip()
            if pdb_id:
                pdb_ids.append(pdb_id)
    return sorted(set(pdb_ids)), reader.fieldnames[0]


def load_combined_npz_as_mapping(npz_path: str) -> dict:
    """
    Tries to interpret a combined .npz as:
      A) direct mapping: keys are pdb_ids -> value (often a dict-like object or an array)
      B) a single pickled dict under arr_0 / data / samples, etc.
    Returns: mapping[pdb_id] -> payload
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Combined NPZ not found: {npz_path}")

    z = np.load(npz_path, allow_pickle=True)

    # Case A: multiple keys in NPZ are pdb_ids directly
    # Heuristic: if there are many keys and at least one looks like a pdb id-ish string.
    keys = list(z.files)

    # Case B: single pickled object containing dict
    if len(keys) == 1 and keys[0] in ("arr_0", "data", "samples", "payload"):
        obj = z[keys[0]]
        # obj could be a 0-d object array storing a dict
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            obj = obj.item()
        if isinstance(obj, dict):
            return obj
        raise ValueError(
            f"Combined NPZ '{npz_path}' has a single key '{keys[0]}' but is not a dict. "
            f"Type={type(obj)}"
        )

    # Try: if there's a key that itself is a dict
    for k in ("data", "samples", "payload"):
        if k in z.files:
            obj = z[k]
            if isinstance(obj, np.ndarray) and obj.dtype == object:
                obj = obj.item()
            if isinstance(obj, dict):
                return obj

    # Default: treat it as key -> value mapping
    mapping = {}
    for k in keys:
        v = z[k]
        # If v is object-array holding a python object, unwrap
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            v = v.item()
        mapping[k] = v
    return mapping


def write_payload_as_npz(out_path: str, payload):
    """
    Writes a per-PDB payload back to .npz.

    Supports:
      - dict of arrays (recommended): np.savez_compressed(**payload)
      - numpy arrays: saved as arr_0
      - other python objects: saved pickled as arr_0 (object array)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if isinstance(payload, dict):
        np.savez_compressed(out_path, **payload)
        return

    if isinstance(payload, np.ndarray):
        np.savez_compressed(out_path, arr_0=payload)
        return

    # Fallback: store python object
    np.savez_compressed(out_path, arr_0=np.array(payload, dtype=object))


def main():
    parser = argparse.ArgumentParser(
        description="Run prediction + plot using combined NPZs (many PDBs) listed in CSV (first column)."
    )
    parser.add_argument("--csv", required=True, help="Headered CSV; FIRST column is PDB_ID list.")

    # NEW: combined NPZ inputs
    parser.add_argument("--combined-seq-npz", required=True, help="Combined esm_sequences.npz for ALL pdbs.")
    parser.add_argument("--combined-edges-npz", required=True, help="Combined edges_combined.npz for ALL pdbs.")

    # Output root (we'll create <out>/<pdb_id>/ and drop per-PDB npzs + outputs there)
    parser.add_argument(
        "--preprocess-out-dir",
        required=True,
        help="Output root dir; will write <PDB_ID>/esm_sequences.npz and edges_combined.npz + predictions/plots.",
    )

    parser.add_argument(
        "--scripts-dir",
        default=".",
        help="Directory containing make_prediction.py and plot_pred_from_npz.py.",
    )
    parser.add_argument("--esm-model-name", default="esm2_t6_8M_UR50D", help="ESM model name for plotting.")
    parser.add_argument("--eoc-id", type=int, default=24, help="EOC token id for plotting.")
    parser.add_argument("--sample-idx", type=str, default="0", help="Sample index for plot script.")
    args = parser.parse_args()

    py = sys.executable
    scripts_dir = os.path.abspath(args.scripts_dir)
    args.csv = os.path.abspath(args.csv)

    out_root = os.path.abspath(args.preprocess_out_dir)
    os.makedirs(out_root, exist_ok=True)

    combined_seq_npz = os.path.abspath(args.combined_seq_npz)
    combined_edges_npz = os.path.abspath(args.combined_edges_npz)

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"--csv does not exist: {args.csv}")

    predict_py = os.path.join(scripts_dir, "make_prediction.py")
    plot_py = os.path.join(scripts_dir, "plot_pred_from_npz.py")

    if not os.path.isfile(predict_py):
        raise FileNotFoundError(f"make_prediction.py not found under --scripts-dir: {predict_py}")
    if not os.path.isfile(plot_py):
        raise FileNotFoundError(f"plot_pred_from_npz.py not found under --scripts-dir: {plot_py}")

    pdb_ids, pdb_col = read_pdb_ids_from_csv_first_col(args.csv)
    if not pdb_ids:
        print(f"No PDB IDs found in first column '{pdb_col}' of CSV: {args.csv}")
        sys.exit(1)

    print(f"[info] Using first CSV column as PDB ID: '{pdb_col}'")
    print(f"Found {len(pdb_ids)} unique PDB IDs in CSV.")
    print("First few PDB IDs:", pdb_ids[:10])

    print(f"[info] Loading combined seq NPZ  : {combined_seq_npz}")
    print(f"[info] Loading combined edges NPZ: {combined_edges_npz}")
    seq_map = load_combined_npz_as_mapping(combined_seq_npz)
    edges_map = load_combined_npz_as_mapping(combined_edges_npz)

    print(f"[info] Combined seq keys  : {len(seq_map)}")
    print(f"[info] Combined edges keys: {len(edges_map)}")

    num_total = len(pdb_ids)
    num_success = 0
    num_fail = 0
    num_missing = 0

    for idx, pdb_id in enumerate(pdb_ids, start=1):
        print(f"\n#### [{idx}/{num_total}] Running model for PDB_ID={pdb_id} ####\n")

        # Create output dir for this PDB
        pre_dir = os.path.join(out_root, pdb_id)
        os.makedirs(pre_dir, exist_ok=True)

        # Extract payloads for this pdb_id
        if pdb_id not in seq_map or pdb_id not in edges_map:
            num_missing += 1
            print(f"⚠️  Missing combined payload for {pdb_id}:")
            if pdb_id not in seq_map:
                print("    - not found in combined seq npz")
            if pdb_id not in edges_map:
                print("    - not found in combined edges npz")
            print("    Skipping.")
            continue

        # Write per-PDB NPZs in the format downstream scripts already expect
        seq_npz = os.path.join(pre_dir, "esm_sequences.npz")
        edges_npz = os.path.join(pre_dir, "edges_combined.npz")

        try:
            write_payload_as_npz(seq_npz, seq_map[pdb_id])
            write_payload_as_npz(edges_npz, edges_map[pdb_id])
        except Exception as e:
            num_fail += 1
            print(f"\n❌ Failed to write per-PDB NPZs for {pdb_id}: {e}")
            traceback.print_exc()
            print("Skipping to next PDB...\n")
            continue

        try:
            # --- prediction ---
            print("\n=== [1/2] Run paratope prediction ===")
            run([py, predict_py, seq_npz, edges_npz], cwd=pre_dir)

            pred_csv = os.path.join(pre_dir, os.path.basename(seq_npz).replace(".npz", "_msa_pred.csv"))
            if not os.path.isfile(pred_csv):
                raise FileNotFoundError(f"Expected prediction CSV not found: {pred_csv}")

            # --- plot ---
            print("\n=== [2/2] Make prediction grid plot ===")
            out_png = os.path.join(pre_dir, f"{pdb_id}_pred_grid.png")
            run(
                [
                    py,
                    plot_py,
                    "--seq-npz",
                    seq_npz,
                    "--pred-csv",
                    pred_csv,
                    "--model-name",
                    args.esm_model_name,
                    "--eoc-id",
                    str(args.eoc_id),
                    "--key",
                    pdb_id,
                    "--sample-idx",
                    str(args.sample_idx),
                    "--out",
                    out_png,
                ],
                cwd=pre_dir,
            )

            print("\n✅ Model run finished for", pdb_id)
            print(f"   Prediction: {pred_csv}")
            print(f"   Plot      : {out_png}")
            num_success += 1

        except subprocess.CalledProcessError as e:
            num_fail += 1
            print(f"\n❌ Subprocess failed for {pdb_id} with return code {e.returncode}")
            print("Command:", e.cmd)
            print("Skipping to next PDB...\n")
        except Exception as e:
            num_fail += 1
            print(f"\n❌ Exception while running model for {pdb_id}: {e}")
            traceback.print_exc()
            print("Skipping to next PDB...\n")

    print("\n========================================")
    print("🎯 Model runs finished.")
    print(f"   Successful: {num_success}")
    print(f"   Failed    : {num_fail}")
    print(f"   Missing   : {num_missing} (not found in combined NPZs)")
    print("========================================\n")


if __name__ == "__main__":
    main()

