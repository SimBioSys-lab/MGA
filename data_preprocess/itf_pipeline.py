#!/usr/bin/env python3
"""
itf_pipeline.py

Run AFTER data_preprocess.py (assumes the combined sequences NPZ already exists).
Pipeline:
  1) Extract_interface.py -> raw interface text files (per PDB, per chain pair)
  2) itf_gen.py           -> combined interfaces NPZ for model training

This pipeline:
- reads work dir and sequence npz name (supports auto-discovery if not provided)
- takes CSV as input (CSV header ignored by downstream scripts)
"""

import os
import sys
import glob
import argparse
import subprocess


def run(cmd, cwd=None):
    print(f">>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def abspath(p):
    return os.path.abspath(os.path.expanduser(p))


def find_seq_npz(work_dir: str) -> str:
    """
    Auto-discover a sequences NPZ under work_dir.
    Prefers names containing "seq" or "esm" and ending with .npz.
    """
    patterns = [
        os.path.join(work_dir, "*seq*.npz"),
        os.path.join(work_dir, "*Seq*.npz"),
        os.path.join(work_dir, "*esm*seq*.npz"),
        os.path.join(work_dir, "*esm*sequence*.npz"),
        os.path.join(work_dir, "*sequences*.npz"),
        os.path.join(work_dir, "*.npz"),
    ]
    cands = []
    for pat in patterns:
        cands.extend(glob.glob(pat))

    # remove obvious non-seq artifacts if possible
    cands = [p for p in cands if os.path.isfile(p)]

    # rank candidates
    def score(p):
        name = os.path.basename(p).lower()
        s = 0
        if "esm" in name:
            s += 3
        if "seq" in name or "sequence" in name:
            s += 3
        if "interface" in name or "itf" in name:
            s -= 3
        if "edge" in name:
            s -= 2
        if "label" in name:
            s -= 2
        return s

    cands.sort(key=lambda p: (-score(p), p))
    if not cands:
        raise FileNotFoundError(
            f"Could not auto-discover a sequences NPZ under work_dir={work_dir}. "
            f"Please pass --seq-npz explicitly."
        )
    return cands[0]


def main():
    ap = argparse.ArgumentParser(description="Interface generation pipeline (post data_preprocess).")
    ap.add_argument("--csv", required=True, help="Train/val/test CSV. First column is PDB ID or PDB filename.")
    ap.add_argument("--work-dir", required=True, help="Work dir (where data_preprocess outputs live).")
    ap.add_argument("--seq-npz", default=None, help="Combined sequences NPZ. If omitted, auto-discover in work dir.")
    ap.add_argument("--pdb-dir", default=None, help="Directory holding PDB files (optional).")
    ap.add_argument("--ext", default=".pdb", help="PDB extension if missing (default: .pdb).")
    ap.add_argument("--cutoff", type=float, default=10.0, help="Distance cutoff in Angstrom (default: 10).")
    ap.add_argument("--eoc-token-id", type=int, default=24, help="EOC token id in sequences (default: 24).")
    ap.add_argument("--scripts-dir", default=".", help="Where Extract_interface.py and itf_gen.py live.")
    ap.add_argument("--interface-dir", default=None, help="Raw interface output dir (default: <work-dir>/interfaces_raw).")
    ap.add_argument("--out-npz", default=None, help="Combined interface NPZ (default: <work-dir>/interfaces_combined.npz).")
    args = ap.parse_args()

    work_dir = abspath(args.work_dir)
    csv_path = abspath(args.csv)
    scripts_dir = abspath(args.scripts_dir)

    if not os.path.isdir(work_dir):
        raise FileNotFoundError(f"--work-dir not found: {work_dir}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"--csv not found: {csv_path}")
    if not os.path.isdir(scripts_dir):
        raise FileNotFoundError(f"--scripts-dir not found: {scripts_dir}")

    extract_py = os.path.join(scripts_dir, "Extract_interface.py")
    itf_gen_py = os.path.join(scripts_dir, "itf_gen.py")
    if not os.path.isfile(extract_py):
        raise FileNotFoundError(f"Extract_interface.py not found in scripts_dir: {extract_py}")
    if not os.path.isfile(itf_gen_py):
        raise FileNotFoundError(f"itf_gen.py not found in scripts_dir: {itf_gen_py}")

    if args.seq_npz:
        seq_npz = abspath(args.seq_npz)
    else:
        seq_npz = find_seq_npz(work_dir)

    if not os.path.isfile(seq_npz):
        raise FileNotFoundError(f"Sequences NPZ not found: {seq_npz}")

    interface_dir = abspath(args.interface_dir) if args.interface_dir else os.path.join(work_dir, "interfaces_raw")
    out_npz = abspath(args.out_npz) if args.out_npz else os.path.join(work_dir, "interfaces_combined.npz")
    os.makedirs(interface_dir, exist_ok=True)

    pdb_dir = abspath(args.pdb_dir) if args.pdb_dir else None
    if pdb_dir and not os.path.isdir(pdb_dir):
        raise FileNotFoundError(f"--pdb-dir is not a directory: {pdb_dir}")

    py = sys.executable

    print("========================================")
    print("🚀 Interface pipeline (post data_preprocess)")
    print(f"  work_dir      : {work_dir}")
    print(f"  csv           : {csv_path}")
    print(f"  scripts_dir   : {scripts_dir}")
    print(f"  seq_npz       : {seq_npz}")
    print(f"  pdb_dir       : {pdb_dir or '<none>'}")
    print(f"  ext           : {args.ext}")
    print(f"  cutoff        : {args.cutoff}")
    print(f"  interface_dir : {interface_dir}")
    print(f"  out_npz       : {out_npz}")
    print("========================================\n")

    # Step 1: Extract raw interface files
    cmd1 = [
        py,
        extract_py,
        "--csv",
        csv_path,
        "--work-dir",
        work_dir,
        "--ext",
        args.ext,
        "--cutoff",
        str(args.cutoff),
        "--interface-dir",
        interface_dir,
    ]
    if pdb_dir:
        cmd1 += ["--pdb-dir", pdb_dir]

    print("[pipeline] Step 1/2: Extract_interface.py")
    run(cmd1)

    # Step 2: Generate combined interface NPZ
    cmd2 = [
        py,
        itf_gen_py,
        "--csv",
        csv_path,
        "--work-dir",
        work_dir,
        "--seq-npz",
        seq_npz,
        "--interface-dir",
        interface_dir,
        "--cutoff",
        str(args.cutoff),
        "--eoc-token-id",
        str(args.eoc_token_id),
        "--out-npz",
        out_npz,
    ]

    print("\n[pipeline] Step 2/2: itf_gen.py")
    run(cmd2)

    print("\n✅ Pipeline finished.")
    print(f"  Raw interface files: {interface_dir}")
    print(f"  Combined NPZ       : {out_npz}")


if __name__ == "__main__":
    main()

