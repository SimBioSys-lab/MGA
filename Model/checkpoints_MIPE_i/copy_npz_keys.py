#!/usr/bin/env python3
"""
Copy specific keys from one .npz file to another, saving the result under a new name.

Usage:
    python copy_npz_keys.py <keys.csv> <src.npz> <dst.npz> <out.npz>

- keys.csv: one key name per line
- src.npz: source npz file to copy keys from
- dst.npz: base npz file whose existing contents are preserved (not modified on disk)
- out.npz: output file — contains dst.npz's keys plus the copied keys from src.npz
           (if dst.npz doesn't exist, out.npz will just contain the copied keys)
"""

import sys
import csv
import numpy as np


def load_keys(keys_csv):
    with open(keys_csv, "r", newline="") as f:
        reader = csv.reader(f)
        keys = [row[0].strip() for row in reader if row and row[0].strip()]
    return keys


def load_npz_dict(path):
    try:
        with np.load(path, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}
    except FileNotFoundError:
        return {}


def main():
    if len(sys.argv) != 5:
        print("Usage: python copy_npz_keys.py <keys.csv> <src.npz> <dst.npz> <out.npz>")
        sys.exit(1)

    keys_csv, src_npz, dst_npz, out_npz = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    keys = load_keys(keys_csv)
    print(f"Found {len(keys)} keys to copy: {keys}")

    src_dict = load_npz_dict(src_npz)
    dst_dict = load_npz_dict(dst_npz)

    missing_keys = []
    for key in keys:
        if key in src_dict:
            dst_dict[key] = src_dict[key]
        else:
            missing_keys.append(key)

    if missing_keys:
        print(f"Warning: these keys were not found in {src_npz}: {missing_keys}")

    np.savez(out_npz, **dst_dict)
    print(f"Done. {out_npz} now contains {len(dst_dict)} keys.")


if __name__ == "__main__":
    main()
