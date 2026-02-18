#!/usr/bin/env python3
import os
import argparse
import numpy as np


def list_pdb_dirs(preprocess_out_dir):
    dirs = []
    for name in sorted(os.listdir(preprocess_out_dir)):
        p = os.path.join(preprocess_out_dir, name)
        if os.path.isdir(p):
            dirs.append(name)
    return dirs


def load_npz(npz_path):
    # allow_pickle=False by default; if your arrays contain objects, set allow_pickle=True
    return np.load(npz_path)


def merge_npzs(preprocess_out_dir, filename, out_path, key_prefix_mode="pdb/key"):
    """
    key_prefix_mode:
      - "pdb/key": combined key is f"{pdb_id}/{orig_key}"
      - "pdb":     combined key is just pdb_id (only safe if each per-pdb npz has exactly one key)
      - "orig":    combined key is orig_key (only safe if orig_keys are globally unique across pdbs)
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

        data = load_npz(npz_path)
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
        print(f"⚠️  Skipped {len(skipped)} PDB folders:")
        for pdb_id, reason in skipped[:20]:
            print(f"   - {pdb_id}: {reason}")
        if len(skipped) > 20:
            print(f"   ... and {len(skipped) - 20} more.")


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-PDB npz files into two global npz files (for training)."
    )
    parser.add_argument("--preprocess-out-dir", required=True,
                        help="Root dir containing per-PDB subfolders.")
    parser.add_argument("--out-seq-npz", required=True,
                        help="Output combined NPZ for sequences (esm_sequences).")
    parser.add_argument("--out-edges-npz", required=True,
                        help="Output combined NPZ for edges (edges_combined).")
    parser.add_argument("--seq-filename", default="esm_sequences.npz",
                        help="Per-PDB seq NPZ filename inside each PDB folder.")
    parser.add_argument("--edges-filename", default="edges_combined.npz",
                        help="Per-PDB edges NPZ filename inside each PDB folder.")
    parser.add_argument("--key-prefix-mode", default="pdb/key",
                        choices=["pdb/key", "pdb", "orig"],
                        help="How to name keys in the combined NPZ to avoid collisions.")
    args = parser.parse_args()

    pre = os.path.abspath(args.preprocess_out_dir)
    if not os.path.isdir(pre):
        raise FileNotFoundError(f"Not a directory: {pre}")

    print(f"[info] preprocess_out_dir = {pre}")
    print(f"[info] key_prefix_mode    = {args.key_prefix_mode}")

    # merge sequences
    merge_npzs(
        preprocess_out_dir=pre,
        filename=args.seq_filename,
        out_path=os.path.abspath(args.out_seq_npz),
        key_prefix_mode=args.key_prefix_mode,
    )

    # merge edges
    merge_npzs(
        preprocess_out_dir=pre,
        filename=args.edges_filename,
        out_path=os.path.abspath(args.out_edges_npz),
        key_prefix_mode=args.key_prefix_mode,
    )


if __name__ == "__main__":
    main()

