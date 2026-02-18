#!/usr/bin/env python3
import os
import subprocess
import time
import argparse


def run_hhblits(query_fasta, output_hhr, output_a3m,
                db_path, n_iterations=3, e_value=0.001, num_threads=25):
    """
    Runs hhblits on a query sequence against a database.
    """
    cmd = [
        "hhblits",
        "-i", query_fasta,
        "-o", output_hhr,
        "-oa3m", output_a3m,
        "-d", db_path,
        "-n", str(n_iterations),
        "-e", str(e_value),
        "-cpu", str(num_threads),
    ]

    print("Running HHblits with command:", " ".join(cmd))

    start_time = time.time()
    subprocess.run(cmd, check=True)
    elapsed_time = time.time() - start_time

    print(f"HHblits completed for {os.path.basename(query_fasta)}")
    print(f"  HHR : {output_hhr}")
    print(f"  A3M : {output_a3m}")
    print(f"  Time: {elapsed_time:.2f} seconds\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run HHblits for all FASTA files listed in fasta_files."
    )
    parser.add_argument(
        "--work-dir",
        default=".",
        help="Working directory containing fasta_files and FASTA files [default: current dir]."
    )
    parser.add_argument(
        "--fasta-list",
        default="fasta_files",
        help="File listing FASTA filenames, one per line [default: fasta_files]."
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="HHblits database prefix (e.g., /path/UniRef30_2023_02)."
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=3,
        help="Number of HHblits iterations [default: 3]."
    )
    parser.add_argument(
        "--e-value",
        type=float,
        default=0.001,
        help="E-value threshold [default: 0.001]."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=10,
        help="Number of CPU threads [default: 10]."
    )
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    fasta_list_path = os.path.join(work_dir, args.fasta_list)

    if not os.path.exists(work_dir):
        raise FileNotFoundError(f"Work directory not found: {work_dir}")
    if not os.path.exists(fasta_list_path):
        raise FileNotFoundError(f"FASTA list file not found: {fasta_list_path}")

    # 🔧 Only check the directory that contains the DB prefix
    db_prefix = args.db_path
    db_dir = os.path.dirname(db_prefix) or "."
    if not os.path.isdir(db_dir):
        raise FileNotFoundError(f"HHblits DB directory not found: {db_dir}")

    print(f"Working directory: {work_dir}")
    print(f"FASTA list file : {fasta_list_path}")
    print(f"HHblits DB prefix: {db_prefix}")
    print(f"HHblits DB dir   : {db_dir}")

    with open(fasta_list_path, "r") as f:
        fasta_names = [line.strip() for line in f if line.strip()]

    if not fasta_names:
        print("No FASTA entries found in fasta_files. Nothing to do.")
        return

    print(f"Found {len(fasta_names)} FASTA entries to process.\n")

    for fasta_name in fasta_names:
        query_fasta = os.path.join(work_dir, fasta_name)
        if not os.path.exists(query_fasta):
            print(f"⚠️  FASTA not found, skipping: {query_fasta}")
            continue

        chain = os.path.splitext(fasta_name)[0]
        output_hhr = os.path.join(work_dir, chain + ".hhr")
        output_a3m = os.path.join(work_dir, chain + ".a3m")

        try:
            run_hhblits(
                query_fasta=query_fasta,
                output_hhr=output_hhr,
                output_a3m=output_a3m,
                db_path=db_prefix,
                n_iterations=args.n_iter,
                e_value=args.e_value,
                num_threads=args.threads,
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ HHblits failed for {fasta_name}: {e}")


if __name__ == "__main__":
    main()

