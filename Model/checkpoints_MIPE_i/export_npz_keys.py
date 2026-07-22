#!/usr/bin/env python3
"""
Export the keys of an .npz file to a text file, one key per line.

Usage:
    python export_npz_keys.py <input.npz> <output.txt>
"""

import sys
import numpy as np


def main():
    if len(sys.argv) != 3:
        print("Usage: python export_npz_keys.py <input.npz> <output.txt>")
        sys.exit(1)

    npz_path, out_path = sys.argv[1], sys.argv[2]

    with np.load(npz_path, allow_pickle=True) as data:
        keys = data.files

    with open(out_path, "w") as f:
        for key in keys:
            f.write(key + "\n")

    print(f"Wrote {len(keys)} keys from {npz_path} to {out_path}")


if __name__ == "__main__":
    main()
