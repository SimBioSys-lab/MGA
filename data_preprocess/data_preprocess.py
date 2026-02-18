#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
import subprocess
import traceback
import csv
import numpy as np


def run(cmd, cwd=None):
    print(f">>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


# -------------------------
# CSV helpers (first column is PDB ID)
# -------------------------

def read_pdb_ids_from_csv_first_col(csv_path):
    """
    Read PDB IDs from the FIRST column of a headered CSV.
    Returns (pdb_ids_sorted_unique, first_col_name).
    """
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
    pdb_ids = sorted(set(pdb_ids))
    return pdb_ids, pdb_col


def write_one_row_csv(master_csv_path, pdb_id, out_csv_path):
    """
    Write a single-row CSV (header + one row) for a given pdb_id,
    matching on FIRST column value.
    Returns the pdb_col name used.
    """
    with open(master_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header / no columns: {master_csv_path}")
        pdb_col = reader.fieldnames[0]

        matches = []
        for row in reader:
            val = row.get(pdb_col)
            if val is None:
                continue
            if str(val).strip() == pdb_id:
                matches.append(row)

    if len(matches) == 0:
        raise ValueError(
            f"PDB_ID '{pdb_id}' not found in CSV {master_csv_path} "
            f"(matching first column '{pdb_col}')."
        )
    if len(matches) > 1:
        print(f"⚠️  Multiple rows found for PDB_ID={pdb_id}; using the first row.")

    row = matches[0]

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerow(row)

    return pdb_col


# -------------------------
# PDB locating
# -------------------------

def find_pdb_file(pdb_dir, pdb_id):
    """
    Find a PDB file for pdb_id inside pdb_dir.
    Tries: <pdb_id>.pdb, lower, upper.
    """
    candidates = [
        f"{pdb_id}.pdb",
        f"{pdb_id.lower()}.pdb",
        f"{pdb_id.upper()}.pdb",
    ]
    for name in candidates:
        path = os.path.join(pdb_dir, name)
        if os.path.isfile(path):
            return path
    return None


# -------------------------
# NPZ merge helpers
# -------------------------

def list_pdb_dirs(preprocess_out_dir):
    return [
        name for name in sorted(os.listdir(preprocess_out_dir))
        if os.path.isdir(os.path.join(preprocess_out_dir, name))
    ]


def merge_npzs(preprocess_out_dir, filename, out_path, key_prefix_mode="pdb"):
    """
    Merge per-PDB NPZ files into one big NPZ.

    key_prefix_mode:
      - "pdb":     combined key is just pdb_id (requires exactly 1 key per per-PDB NPZ)  [DEFAULT]
      - "pdb/key": combined key is f"{pdb_id}/{orig_key}"  (safest if multi-key per file)
      - "orig":    combined key is orig_key (requires keys globally unique across PDBs)
    """
    pdb_dirs = list_pdb_dirs(preprocess_out_dir)
    if not pdb_dirs:
        raise FileNotFoundError(f"No subdirectories found under {preprocess_out_dir}")

    merged = {}
    skipped = []

    for pdb_id in pdb_dirs:
        npz_path = os.path.join(preprocess_out_dir, pdb_id, filename)
        if not os.path.isfile(npz_path):
            skipped.append((pdb_id, "missing"))
            continue

        data = np.load(npz_path)
        keys = list(data.keys())
        if len(keys) == 0:
            skipped.append((pdb_id, "empty"))
            continue

        if key_prefix_mode == "pdb":
            if len(keys) != 1:
                raise ValueError(
                    f"{npz_path} has {len(keys)} keys ({keys}), but key_prefix_mode='pdb' "
                    f"requires exactly 1 key per file."
                )
            new_key = pdb_id
            if new_key in merged:
                raise ValueError(f"Key collision for '{new_key}' from {npz_path}")
            merged[new_key] = data[keys[0]]

        elif key_prefix_mode == "orig":
            for k in keys:
                if k in merged:
                    raise ValueError(
                        f"Key collision for '{k}' when merging {npz_path}. "
                        f"Use key_prefix_mode='pdb/key' to avoid collisions."
                    )
                merged[k] = data[k]

        elif key_prefix_mode == "pdb/key":
            for k in keys:
                new_key = f"{pdb_id}/{k}"
                if new_key in merged:
                    raise ValueError(f"Key collision for '{new_key}' from {npz_path}")
                merged[new_key] = data[k]
        else:
            raise ValueError(f"Unknown key_prefix_mode: {key_prefix_mode}")

    if not merged:
        raise RuntimeError(f"No data merged for filename={filename}. Check paths and filenames.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **merged)

    print(f"✅ Merged {len(merged)} arrays into: {out_path}")
    if skipped:
        print(f"⚠️  Skipped {len(skipped)} PDB folders for {filename}:")
        for pdb_id, reason in skipped[:20]:
            print(f"   - {pdb_id}: {reason}")
        if len(skipped) > 20:
            print(f"   ... and {len(skipped) - 20} more.")


# -------------------------
# Per-PDB preprocessing (steps 1–6)
# -------------------------

def run_preprocess_for_pdb(input_pdb, pdb_id, args, scripts_dir):
    """
    Run steps 1–6 for a single PDB, using a one-row CSV so helper scripts
    won't try to process other PDBs in the big CSV.

    IMPORTANT FIX:
    - Some process_pdb.py versions write FASTAs into pdb_chains/.
      run_hh.py expects FASTAs in work_dir root.
      We copy *.fasta from pdb_chains_dir -> work_dir right after step 1.
    """
    py = sys.executable
    process_pdb_py = os.path.join(scripts_dir, "process_pdb.py")
    adj_gen_py = os.path.join(scripts_dir, "adj_gen.py")
    run_hh_py = os.path.join(scripts_dir, "run_hh.py")
    kmeans_py = os.path.join(scripts_dir, "kmean_msa_downsampling.new.py")
    seq_gen_py = os.path.join(scripts_dir, "seq_gen.py")
    edge_gen_py = os.path.join(scripts_dir, "edge_gen.py")

    input_pdb = os.path.abspath(input_pdb)
    pdb_name = os.path.splitext(os.path.basename(input_pdb))[0]

    # temp work dir per PDB: <work_dir>/tmp/<pdb_name>
    tmp_root = os.path.join(args.work_dir, "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    work_dir = os.path.join(tmp_root, pdb_name)
    os.makedirs(work_dir, exist_ok=True)

    print(f"\n==============================")
    print(f"🧪 Preprocessing PDB: {pdb_id}  (file base: {pdb_name})")
    print(f"🧪 Working directory: {work_dir}")
    print(f"==============================\n")

    # copy pdb into work_dir
    dst_pdb = os.path.join(work_dir, f"{pdb_name}.pdb")
    try:
        if os.path.abspath(input_pdb) != os.path.abspath(dst_pdb):
            shutil.copy2(input_pdb, dst_pdb)
    except shutil.SameFileError:
        pass

    # one-row CSV in work_dir
    row_csv = os.path.join(work_dir, f"{pdb_id}__row.csv")
    pdb_col = write_one_row_csv(args.csv, pdb_id, row_csv)
    print(f"[info] wrote one-row CSV for {pdb_id} to {row_csv} (first column '{pdb_col}')")

    pdb_chains_dir = os.path.join(work_dir, "pdb_chains")
    edge_lists_npz = os.path.join(work_dir, "edge_lists.npz")
    seq_npz = os.path.join(work_dir, "esm_sequences.npz")
    combined_edges_npz = os.path.join(work_dir, "edges_combined.npz")

    # [1/6] process_pdb.py
    print("\n=== [1/6] Extract chains and FASTA ===")
    run(
        [
            py, process_pdb_py,
            row_csv,
            "--pdb-dir", work_dir,
            "--ext", ".pdb",
            "--pdb-outdir", pdb_chains_dir,
            "--fasta-outdir", work_dir,  # even if ignored, we'll copy below
        ],
        cwd=work_dir,
    )

    # ---- CRITICAL FIX: ensure FASTAs are in work_dir root for run_hh.py ----
    copied = 0
    if os.path.isdir(pdb_chains_dir):
        for fn in os.listdir(pdb_chains_dir):
            if fn.endswith(".fasta"):
                src = os.path.join(pdb_chains_dir, fn)
                dst = os.path.join(work_dir, fn)
                shutil.copy2(src, dst)
                copied += 1
    print(f"[info] Copied {copied} FASTA files from {pdb_chains_dir} -> {work_dir} for HHblits")

    # fasta_files for hhblits script (from work_dir root)
    fasta_list_path = os.path.join(work_dir, "fasta_files")
    with open(fasta_list_path, "w") as fh:
        for fn in sorted(os.listdir(work_dir)):
            if fn.endswith(".fasta"):
                fh.write(fn + "\n")
    print(f"[info] wrote FASTA list to {fasta_list_path}")

    # [2/6] adj_gen.py
    print("\n=== [2/6] Build residue adjacency edge lists ===")
    run(
        [
            py, adj_gen_py,
            "--csv", row_csv,
            "--pdb-chains-dir", pdb_chains_dir,
            "--output", edge_lists_npz,
        ],
        cwd=work_dir,
    )

    # [3/6] run_hh.py
    print("\n=== [3/6] Run HHblits (MSA generation) ===")
    run(
        [py, run_hh_py, "--work-dir", work_dir, "--db-path", args.hh_db_path],
        cwd=work_dir,
    )

    # a3m_files for kmeans script
    a3m_list_path = os.path.join(work_dir, "a3m_files")
    with open(a3m_list_path, "w") as fh:
        for fn in sorted(os.listdir(work_dir)):
            if fn.endswith(".a3m"):
                fh.write(fn + "\n")
    print(f"[info] wrote A3M list to {a3m_list_path}")

    # [4/6] kmeans downsample
    print("\n=== [4/6] Downsample MSAs (k-means) ===")
    run([py, kmeans_py, "--work-dir", work_dir], cwd=work_dir)

    # [5/6] seq_gen.py
    print("\n=== [5/6] Build ESM-tokenized combined sequences ===")
    run(
        [
            py, seq_gen_py,
            "--csv", row_csv,
            "--msa-root", work_dir,
            "--output", seq_npz,
            "--depth", "64",
            "--model-name", args.esm_model_name,
            "--max-len", "1600",
            "--pdb-id", pdb_name,
        ],
        cwd=work_dir,
    )

    # [6/6] edge_gen.py
    print("\n=== [6/6] Combine edges with EOC offsets ===")
    run(
        [
            py, edge_gen_py,
            "--csv", row_csv,
            "--edges-npz", edge_lists_npz,
            "--seqs-npz", seq_npz,
            "--output", combined_edges_npz,
            "--eoc-id", str(args.eoc_id),
            "--pdb-id", pdb_name,
        ],
        cwd=work_dir,
    )

    # copy outputs to preprocess_out_dir/<pdb_id>/
    pre_dir = os.path.join(args.preprocess_out_dir, pdb_id)
    os.makedirs(pre_dir, exist_ok=True)

    dst_seq_npz = os.path.join(pre_dir, "esm_sequences.npz")
    dst_edges_npz = os.path.join(pre_dir, "edges_combined.npz")
    shutil.copy2(seq_npz, dst_seq_npz)
    shutil.copy2(combined_edges_npz, dst_edges_npz)

    print("\n✅ Preprocessing finished successfully for", pdb_id)
    print(f"   Stored sequence NPZ : {dst_seq_npz}")
    print(f"   Stored edges NPZ    : {dst_edges_npz}")

    if not args.keep_work_dir:
        print(f"\n🧹 Removing temp work directory for {pdb_id}: {work_dir}")
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"\n📦 Keeping temp work directory for {pdb_id}: {work_dir}")


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Data preprocessing (steps 1–6) + optional merge to two training NPZs."
    )
    parser.add_argument("--pdb-dir", required=True, help="Directory containing PDB files.")
    parser.add_argument("--csv", required=True, help="Headered CSV; FIRST column is pdb_code list.")
    parser.add_argument("--hh-db-path", required=True, help="Path to HHblits database (directory or prefix).")
    parser.add_argument(
        "--scripts-dir",
        default=".",
        help="Directory containing process_pdb.py, adj_gen.py, run_hh.py, "
             "kmean_msa_downsampling.new.py, seq_gen.py, edge_gen.py.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Base working directory. Uses '<work>/tmp/<pdb_code>/' per PDB. "
             "If omitted, uses parent directory of --pdb-dir.",
    )
    parser.add_argument(
        "--preprocess-out-dir",
        default=None,
        help="Directory where per-PDB outputs are stored: <preprocess-out-dir>/<pdb_code>/. "
             "If omitted, defaults to '<work-dir>/npz_by_pdb'.",
    )
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep per-PDB tmp dirs.")
    parser.add_argument("--esm-model-name", default="esm2_t6_8M_UR50D", help="ESM model name for seq_gen.")
    parser.add_argument("--eoc-id", type=int, default=24, help="EOC token id for edge_gen.")

    # Merge options
    parser.add_argument(
        "--merge",
        action="store_true",
        help="If set, merge all per-PDB NPZs into two combined NPZs for training after preprocessing."
    )
    parser.add_argument(
        "--merged-seq-npz",
        default=None,
        help="Output path for merged sequence NPZ (default: <preprocess-out-dir>/combined_esm_sequences.npz)."
    )
    parser.add_argument(
        "--merged-edges-npz",
        default=None,
        help="Output path for merged edges NPZ (default: <preprocess-out-dir>/combined_edges_combined.npz)."
    )
    parser.add_argument(
        "--key-prefix-mode",
        default="pdb",
        choices=["pdb", "pdb/key", "orig"],
        help="Key naming in merged NPZ (default: pdb → keys are just pdb_code)."
    )

    args = parser.parse_args()

    scripts_dir = os.path.abspath(args.scripts_dir)
    pdb_dir = os.path.abspath(args.pdb_dir)
    args.csv = os.path.abspath(args.csv)

    if not os.path.isdir(pdb_dir):
        raise FileNotFoundError(f"--pdb-dir does not exist or is not a directory: {pdb_dir}")
    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"--csv does not exist: {args.csv}")

    # defaults for work_dir
    if args.work_dir is None:
        args.work_dir = os.path.dirname(pdb_dir)
    args.work_dir = os.path.abspath(args.work_dir)

    # default output dir for per-PDB NPZs (NOT "preprocessed")
    if args.preprocess_out_dir is None:
        args.preprocess_out_dir = os.path.join(args.work_dir, "npz_by_pdb")
    args.preprocess_out_dir = os.path.abspath(args.preprocess_out_dir)
    os.makedirs(args.preprocess_out_dir, exist_ok=True)

    # default merged output paths
    if args.merged_seq_npz is None:
        args.merged_seq_npz = os.path.join(args.preprocess_out_dir, "combined_esm_sequences.npz")
    if args.merged_edges_npz is None:
        args.merged_edges_npz = os.path.join(args.preprocess_out_dir, "combined_edges_combined.npz")
    args.merged_seq_npz = os.path.abspath(args.merged_seq_npz)
    args.merged_edges_npz = os.path.abspath(args.merged_edges_npz)

    # read pdb codes from first column
    pdb_ids, pdb_col = read_pdb_ids_from_csv_first_col(args.csv)
    if not pdb_ids:
        print(f"No PDB IDs found in first column '{pdb_col}' of CSV: {args.csv}")
        sys.exit(1)

    print(f"[info] Using first CSV column as pdb_code: '{pdb_col}'")
    print(f"Found {len(pdb_ids)} unique pdb_code entries in CSV.")
    print("First few pdb_code:", pdb_ids[:10])
    print(f"[info] work_dir           = {args.work_dir}")
    print(f"[info] preprocess_out_dir = {args.preprocess_out_dir}")

    num_total = len(pdb_ids)
    num_success = 0
    num_fail = 0
    num_missing = 0

    for idx, pdb_id in enumerate(pdb_ids, start=1):
        print(f"\n#### [{idx}/{num_total}] Preprocessing pdb_code={pdb_id} ####\n")

        pdb_path = find_pdb_file(pdb_dir, pdb_id)
        if pdb_path is None:
            num_missing += 1
            print(f"⚠️  PDB not found in --pdb-dir: {pdb_dir} (skipping {pdb_id})")
            continue

        try:
            run_preprocess_for_pdb(pdb_path, pdb_id, args, scripts_dir)
            num_success += 1
        except subprocess.CalledProcessError as e:
            num_fail += 1
            print(f"\n❌ Subprocess failed for {pdb_id} with return code {e.returncode}")
            print("Command:", e.cmd)
            print("Skipping to next PDB...\n")
        except Exception as e:
            num_fail += 1
            print(f"\n❌ Exception while preprocessing {pdb_id}: {e}")
            traceback.print_exc()
            print("Skipping to next PDB...\n")

    print("\n========================================")
    print("🎯 Data preprocessing finished.")
    print(f"   Successful: {num_success}")
    print(f"   Failed    : {num_fail}")
    print(f"   Missing   : {num_missing} (PDB file not found in --pdb-dir)")
    print("========================================\n")

    # merge stage
    if args.merge:
        if not list_pdb_dirs(args.preprocess_out_dir):
            raise FileNotFoundError(
                f"No per-PDB subfolders found under preprocess_out_dir={args.preprocess_out_dir}.\n"
                f"Expected structure: {args.preprocess_out_dir}/<pdb_code>/esm_sequences.npz and edges_combined.npz\n"
                f"Fix by setting --preprocess-out-dir to the folder that contains the per-PDB folders."
            )

        print("\n========================================")
        print("🔧 Merging per-PDB NPZs into two combined NPZs for training...")
        print(f"[info] merge root       = {args.preprocess_out_dir}")
        print(f"[info] key_prefix_mode  = {args.key_prefix_mode}")
        print("========================================\n")

        merge_npzs(
            preprocess_out_dir=args.preprocess_out_dir,
            filename="esm_sequences.npz",
            out_path=args.merged_seq_npz,
            key_prefix_mode=args.key_prefix_mode,
        )
        merge_npzs(
            preprocess_out_dir=args.preprocess_out_dir,
            filename="edges_combined.npz",
            out_path=args.merged_edges_npz,
            key_prefix_mode=args.key_prefix_mode,
        )

        print("\n✅ Merge completed.")
        print(f"   Merged sequences: {args.merged_seq_npz}")
        print(f"   Merged edges    : {args.merged_edges_npz}")


if __name__ == "__main__":
    main()

