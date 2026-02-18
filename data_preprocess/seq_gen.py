#!/usr/bin/env python
from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import esm  # ⚠️ pip install fair-esm


def build_tokenizer(model_name: str, pad_char: str, end_char: str):
    """
    Load ESM alphabet and build helpers for tokenization.
    Returns:
      _tokenize(seq: str, tgt_len: int) -> np.ndarray[int16]
      _load_msa(pdb_chain: str, msa_root: str, depth: int) -> np.ndarray[int16]
      pad_idx: int  (token id of PAD)
      end_idx: int  (token id of END_CHAR)
    """
    _, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
    char2idx: dict[str, int] = alphabet.tok_to_idx
    pad_idx: int = alphabet.padding_idx
    end_idx: int = char2idx[end_char]

    def _tok_idx(c: str) -> int:
        if c == pad_char:
            return pad_idx
        # unknown chars → PAD
        return char2idx.get(c, pad_idx)

    def _tokenize(seq: str, tgt_len: int) -> np.ndarray:
        """
        AA string → token row (len = tgt_len + 1)
        * clip / pad with pad_char to tgt_len
        * append END_CHAR
        """
        seq_trim = (seq[:tgt_len]).ljust(tgt_len, pad_char)
        seq_trim += end_char
        return np.fromiter((_tok_idx(c) for c in seq_trim),
                           dtype=np.int16,
                           count=len(seq_trim))

    def _load_msa(pdb_chain: str, msa_root: str, depth: int) -> np.ndarray:
        """
        Returns int16 tensor of shape (depth, query_len+1) for given chain.

        Handles both:
        - plain DS (one sequence per line), and
        - FASTA-style DS (">header" + sequence lines).
        """
        pdb, ch = pdb_chain.split('_')
        path = os.path.join(msa_root, f"{pdb}_chain_{ch}.a3m.DS")
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        sequences = []
        with open(path) as fh:
            current = []
            fasta_mode = None
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    # FASTA header line
                    fasta_mode = True
                    if current:
                        sequences.append("".join(current))
                        current = []
                else:
                    if fasta_mode:
                        current.append(line)
                    else:
                        # plain one-seq-per-line mode
                        sequences.append(line)

            if fasta_mode and current:
                sequences.append("".join(current))

        if not sequences:
            raise RuntimeError(f"{path} has no sequences")

        # query length from FIRST sequence (assumed to be query row)
        q_len = len(sequences[0])
        toks = np.stack([_tokenize(seq, q_len)
                         for seq in sequences[:depth]], axis=0)

        if toks.shape[0] < depth:
            pad_row = np.full((1, q_len + 1), pad_idx, np.int16)
            toks = np.vstack([toks,
                              pad_row.repeat(depth - toks.shape[0], axis=0)])
        return toks

    return _tokenize, _load_msa, pad_idx, end_idx


def main():
    parser = argparse.ArgumentParser(
        description="Generate ESM-tokenized MSA tensors (L+H+AG concatenated, padded to 1600) into an NPZ."
    )
    parser.add_argument(
        "--csv",
        default="tv_set.csv",
        help="CSV with columns: PDB_ID, LChain, HChain, AGChain "
             "[default: tv_set.csv].",
    )
    parser.add_argument(
        "--msa-root",
        required=True,
        help="Directory containing {pdb}_chain_{ch}.a3m.DS files.",
    )
    parser.add_argument(
        "--output",
        default="esm_sequences.npz",
        help="Output NPZ file [default: esm_sequences.npz].",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=64,
        help="MSA depth (rows kept per chain) [default: 64].",
    )
    parser.add_argument(
        "--pad-char",
        default="0",
        help="Padding character used in DS sequences [default: '0'].",
    )
    parser.add_argument(
        "--end-char",
        default="X",
        help="End-of-chain character to append [default: 'X'].",
    )
    parser.add_argument(
        "--model-name",
        default="esm2_t6_8M_UR50D",
        help="ESM model name to load from hub [default: esm2_t6_8M_UR50D].",
    )
    parser.add_argument(
        "--pdb-id",
        help="Optional: only process this PDB_ID (case-insensitive).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1600,
        help="Final sequence length after concatenation and padding [default: 1600].",
    )
    args = parser.parse_args()

    csv_path = args.csv
    msa_root = os.path.abspath(args.msa_root)
    out_npz = args.output
    depth = args.depth
    pad_char = args.pad_char
    end_char = args.end_char
    max_len = args.max_len

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isdir(msa_root):
        raise NotADirectoryError(f"MSA root directory not found: {msa_root}")

    print(f"[info] Loading ESM model/alphabet: {args.model_name}")
    _tokenize, _load_msa, pad_idx, end_idx = build_tokenizer(
        model_name=args.model_name,
        pad_char=pad_char,
        end_char=end_char,
    )
    print(f"[info] END_CHAR='{end_char}' → token id {end_idx}")
    print(f"[info] PAD token id          → {pad_idx}")
    print(f"[info] Final padded length   → {max_len}")

    # Load CSV
    df = pd.read_csv(
        csv_path,
        header=0,
        names=["PDB_ID", "LChain", "HChain", "AGChain"],
    )

    combined: dict[str, np.ndarray] = {}

    for _, row in df.iterrows():
        pdb = str(row.PDB_ID).strip()
        if not pdb:
            continue


        if pd.isna(row.LChain) or pd.isna(row.HChain) or pd.isna(row.AGChain):
            print(f"[skip] {pdb} – missing chain info")
            continue

        try:
            # Load L and H
            L = _load_msa(f"{pdb}_{row.LChain}", msa_root, depth)
            H = _load_msa(f"{pdb}_{row.HChain}", msa_root, depth)

            # Antigen chain(s)
            ag_ids = [c.strip() for c in str(row.AGChain).split(";") if c.strip()]
            if not ag_ids:
                print(f"[skip] {pdb} – no antigen chains")
                continue

            AG_list = [_load_msa(f"{pdb}_{cid}", msa_root, depth) for cid in ag_ids]
            AG = np.concatenate(AG_list, axis=1)   # concat columns

            # Concatenate L, H, AG along columns
            combo = np.concatenate([L, H, AG], axis=1)  # shape (depth, total_len)
            total_len = combo.shape[1]

            if total_len > max_len:
                print(f"[ERROR] {pdb}: combined length {total_len} > {max_len}. Skipping this complex.")
                continue

            if total_len < max_len:
                pad_width = max_len - total_len
                combo = np.pad(
                    combo,
                    ((0, 0), (0, pad_width)),
                    mode="constant",
                    constant_values=pad_idx,
                )

            # Now combo is (depth, max_len)
            combined[pdb] = combo
            print(f"[ok] {pdb}: combo shape {combo.shape}, original length {total_len}")

        except Exception as err:
            print(f"[error] {pdb}: {err}")

    if combined:
        np.savez_compressed(out_npz, **combined)
        print(f"Saved {len(combined)} complexes → {out_npz}")
        print(f"[note] Use END token id = {end_idx} as --eoc-id in edge_gen.py")
    else:
        print("Nothing collected; NPZ not written.")


if __name__ == "__main__":
    main()

