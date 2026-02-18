'''
python run_pipeline.py \
  --pdb-dir /path/to/pdb_folder \
  --csv /path/to/metadata.csv \
  --hh-db-path /path/to/hhdb/Uniref30_2023_02 \
  --scripts-dir /path/to/scripts \
  --work-dir /path/to/work_base \
  --keep-work-dir
'''
#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
import subprocess
import traceback
import csv


def run(cmd, cwd=None):
    print(f">>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def find_pdb_file(pdb_dir, pdb_id):
    """
    Try to find a PDB file for a given PDB_ID in pdb_dir.
    Tries: <pdb_id>.pdb, lower, upper.

    Returns:
        path to pdb file if found, else None.
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


def run_pipeline_for_pdb(input_pdb, args, scripts_dir):
    """
    Run the full pipeline for a single PDB file.

    input_pdb: absolute path to the PDB file.
    args: parsed command-line arguments.
    scripts_dir: absolute path to the scripts directory.
    """
    py = sys.executable
    process_pdb_py = os.path.join(scripts_dir, "process_pdb.py")
    adj_gen_py = os.path.join(scripts_dir, "adj_gen.py")
    run_hh_py = os.path.join(scripts_dir, "run_hh.py")
    kmeans_py = os.path.join(scripts_dir, "kmean_msa_downsampling.new.py")
    seq_gen_py = os.path.join(scripts_dir, "seq_gen.py")
    edge_gen_py = os.path.join(scripts_dir, "edge_gen.py")
    predict_py = os.path.join(scripts_dir, "make_prediction.py")
    plot_py = os.path.join(scripts_dir, "plot_pred_from_npz.py")

    input_pdb = os.path.abspath(input_pdb)
    pdb_name = os.path.splitext(os.path.basename(input_pdb))[0]  # e.g. 2b4c

    # ---------------- working directory ----------------
    if args.work_dir:
        base_work_dir = os.path.abspath(args.work_dir)
    else:
        # default: use the parent directory of the PDB folder
        base_work_dir = os.path.dirname(os.path.abspath(args.pdb_dir))

    # each PDB gets its own subdir: <base_work_dir>/tmp/<pdb_name>
    tmp_root = os.path.join(base_work_dir, "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    work_dir = os.path.join(tmp_root, pdb_name)
    os.makedirs(work_dir, exist_ok=True)

    print(f"\n==============================")
    print(f"🧪 Processing PDB: {pdb_name}")
    print(f"🧪 Working directory: {work_dir}")
    print(f"==============================\n")

    # copy pdb into work_dir as <pdb_name>.pdb if needed
    dst_pdb = os.path.join(work_dir, f"{pdb_name}.pdb")
    try:
        if os.path.abspath(input_pdb) != os.path.abspath(dst_pdb):
            shutil.copy2(input_pdb, dst_pdb)
    except shutil.SameFileError:
        pass

    csv_path = os.path.abspath(args.csv)
    pdb_chains_dir = os.path.join(work_dir, "pdb_chains")
    edge_lists_npz = os.path.join(work_dir, "edge_lists.npz")
    seq_npz = os.path.join(work_dir, "esm_sequences.npz")
    combined_edges_npz = os.path.join(work_dir, "edges_combined.npz")

    # ---------------- 1) PDB -> chains + FASTA ----------------
    print("\n=== [1/8] Extract chains and FASTA ===")
    run(
        [
            py,
            process_pdb_py,
            csv_path,
            "--pdb-dir",
            work_dir,
            "--ext",
            ".pdb",
            "--pdb-outdir",
            pdb_chains_dir,
            "--fasta-outdir",
            work_dir,
        ],
        cwd=work_dir,
    )

    # Create fasta_files list for run_hh.py
    fasta_list_path = os.path.join(work_dir, "fasta_files")
    with open(fasta_list_path, "w") as fh:
        for fn in sorted(os.listdir(work_dir)):
            if fn.endswith(".fasta"):
                fh.write(fn + "\n")
    print(f"[info] wrote FASTA list to {fasta_list_path}")

    # ---------------- 2) adjacency -> edge_lists.npz ----------------
    print("\n=== [2/8] Build residue adjacency edge lists ===")
    run(
        [
            py,
            adj_gen_py,
            "--csv",
            csv_path,
            "--pdb-chains-dir",
            pdb_chains_dir,
            "--output",
            edge_lists_npz,
        ],
        cwd=work_dir,
    )

    # ---------------- 3) HHblits ----------------
    print("\n=== [3/8] Run HHblits (MSA generation) ===")
    run(
        [
            py,
            run_hh_py,
            "--work-dir",
            work_dir,
            "--db-path",
            args.hh_db_path,
        ],
        cwd=work_dir,
    )

    # After HHblits, create a3m_files list for k-means script (if it uses it)
    a3m_list_path = os.path.join(work_dir, "a3m_files")
    with open(a3m_list_path, "w") as fh:
        for fn in sorted(os.listdir(work_dir)):
            if fn.endswith(".a3m"):
                fh.write(fn + "\n")
    print(f"[info] wrote A3M list to {a3m_list_path}")

    # ---------------- 4) k-means downsampling ----------------
    print("\n=== [4/8] Downsample MSAs (k-means) ===")
    run(
        [
            py,
            kmeans_py,
            "--work-dir",
            work_dir,
        ],
        cwd=work_dir,
    )

    # ---------------- 5) seq_gen -> esm_sequences.npz ----------------
    print("\n=== [5/8] Build ESM-tokenized combined sequences ===")
    run(
        [
            py,
            seq_gen_py,
            "--csv",
            csv_path,
            "--msa-root",
            work_dir,
            "--output",
            seq_npz,
            "--depth",
            "64",
            "--model-name",
            args.esm_model_name,
            "--max-len",
            "1600",
            "--pdb-id",
            pdb_name,
        ],
        cwd=work_dir,
    )

    # ---------------- 6) edge_gen -> combined_edges.npz ----------------
    print("\n=== [6/8] Combine edges with EOC offsets ===")
    run(
        [
            py,
            edge_gen_py,
            "--csv",
            csv_path,
            "--edges-npz",
            edge_lists_npz,
            "--seqs-npz",
            seq_npz,
            "--output",
            combined_edges_npz,
            "--eoc-id",
            str(args.eoc_id),
            "--pdb-id",
            pdb_name,
        ],
        cwd=work_dir,
    )

    # ---------------- 7) prediction ----------------
    print("\n=== [7/8] Run paratope prediction ===")
    run(
        [
            py,
            predict_py,
            seq_npz,
            combined_edges_npz,
        ],
        cwd=work_dir,
    )

    pred_csv = os.path.join(
        work_dir, os.path.basename(seq_npz).replace(".npz", "_msa_pred.csv")
    )

    # ---------------- 8) plot ----------------
    print("\n=== [8/8] Make prediction grid plot ===")
    out_png = os.path.join(work_dir, f"{pdb_name}_pred_grid.png")
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
            pdb_name,
            "--sample-idx",
            "0",
            "--out",
            out_png,
        ],
        cwd=work_dir,
    )

    print("\n✅ Pipeline finished successfully for", pdb_name)
    print(f"   Sequence NPZ : {seq_npz}")
    print(f"   Edges NPZ    : {combined_edges_npz}")
    print(f"   Prediction   : {pred_csv}")
    print(f"   Plot         : {out_png}")

    if not args.keep_work_dir:
        print(f"\n🧹 Removing work directory for {pdb_name}: {work_dir}")
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"\n📦 Keeping work directory for {pdb_name}: {work_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: PDBs listed in a CSV -> MSA -> ESM seq+edges -> prediction + plot"
    )
    parser.add_argument(
        "--pdb-dir",
        required=True,
        help="Directory containing PDB files.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV with at least a PDB_ID column (rows to process).",
    )
    parser.add_argument(
        "--hh-db-path",
        required=True,
        help="Path to HHblits database (e.g. Uniref30_2023_02).",
    )
    parser.add_argument(
        "--scripts-dir",
        default=".",
        help="Directory containing process_pdb.py, adj_gen.py, run_hh.py, "
             "kmean_msa_downsampling.new.py, seq_gen.py, edge_gen.py, "
             "make_prediction.py, plot_pred_from_npz.py.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Base working directory. A 'tmp/<pdb_name>' subdir will be used inside this. "
             "If omitted, uses the parent directory of --pdb-dir.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="If set, do not delete the per-PDB temporary work directory after run.",
    )
    parser.add_argument(
        "--model-file",
        default="isicisiParamodelnewesm_l1_g10_i5_do0.30_dpr0.25_lr0.0001_fold10.pth",
        help="Model checkpoint filename (should be in scripts-dir or CWD). "
             "This is not used directly here but kept for compatibility.",
    )
    parser.add_argument(
        "--esm-model-name",
        default="esm2_t6_8M_UR50D",
        help="ESM model name used in seq_gen and plotting [default: esm2_t6_8M_UR50D].",
    )
    parser.add_argument(
        "--eoc-id",
        type=int,
        default=24,
        help="EOC token id in ESM tokenizer and prediction script [default: 24].",
    )
    args = parser.parse_args()

    scripts_dir = os.path.abspath(args.scripts_dir)
    pdb_dir = os.path.abspath(args.pdb_dir)

    if not os.path.isdir(pdb_dir):
        raise FileNotFoundError(f"--pdb-dir does not exist or is not a directory: {pdb_dir}")
    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"--csv does not exist: {args.csv}")

    # ---------------- read CSV and get PDB_ID list ----------------
    pdb_ids = []
    with open(args.csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "PDB_ID" not in reader.fieldnames:
            raise ValueError(
                f"CSV file {args.csv} must have a 'PDB_ID' column, "
                f"but columns are: {reader.fieldnames}"
            )
        for row in reader:
            pdb_id = row["PDB_ID"].strip()
            if pdb_id:
                pdb_ids.append(pdb_id)

    # Unique and sorted for reproducibility
    pdb_ids = sorted(set(pdb_ids))
    if not pdb_ids:
        print(f"No PDB_ID entries found in CSV: {args.csv}")
        sys.exit(1)

    print(f"Found {len(pdb_ids)} PDB_ID entries in CSV.")
    print("First few PDB_IDs:", pdb_ids[:10])

    num_total = len(pdb_ids)
    num_success = 0
    num_fail = 0
    num_missing = 0

    for idx, pdb_id in enumerate(pdb_ids, start=1):
        print(f"\n#### [{idx}/{num_total}] Starting pipeline for PDB_ID={pdb_id} ####\n")

        pdb_path = find_pdb_file(pdb_dir, pdb_id)
        if pdb_path is None:
            num_missing += 1
            print(f"❌ No PDB file found in {pdb_dir} for PDB_ID={pdb_id} "
                  f"(tried {pdb_id}.pdb / lower / upper). Skipping.")
            continue

        try:
            run_pipeline_for_pdb(pdb_path, args, scripts_dir)
            num_success += 1
        except subprocess.CalledProcessError as e:
            num_fail += 1
            print(f"\n❌ Subprocess failed for {pdb_id} with return code {e.returncode}")
            print("Command:", e.cmd)
            print("Skipping to next PDB...\n")
        except Exception as e:
            num_fail += 1
            print(f"\n❌ Exception while processing {pdb_id}: {e}")
            traceback.print_exc()
            print("Skipping to next PDB...\n")

    print("\n========================================")
    print("🎯 Pipeline finished for PDBs listed in CSV.")
    print(f"   Successful: {num_success}")
    print(f"   Failed    : {num_fail}")
    print(f"   Missing   : {num_missing} (PDB file not found in --pdb-dir)")
    print("========================================\n")


if __name__ == "__main__":
    main()

