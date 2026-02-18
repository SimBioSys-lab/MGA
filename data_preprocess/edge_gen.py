#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd


def build_chain_positions(tokens: np.ndarray, eoc_id: int):
    """
    Build per-chain mapping from local residue index -> global token index.

    tokens : 1D array of token ids (length L_total).
    eoc_id : token id used as end-of-chain marker.

    Returns:
      chain_positions: list[list[int]]
        chain_positions[k][i] = global token index for
        residue i (0-based) in chain k.

    EOC tokens themselves are NOT included as residues; they only
    separate chains.
    """
    chain_positions = [[]]
    chain_idx = 0

    for pos, t in enumerate(tokens):
        if t == eoc_id:
            # End-of-chain: start a new chain
            chain_positions.append([])
            chain_idx += 1
        else:
            # This token is a residue; record its global token index
            chain_positions[chain_idx].append(pos)

    # Drop trailing empty chain if the last token was an EOC
    if chain_positions and len(chain_positions[-1]) == 0:
        chain_positions.pop()

    return chain_positions


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-chain edge lists into a single edge index per PDB, "
            "using EOC positions from ESM-tokenized sequences. "
            "Edges are mapped into TOKEN index space, leaving gaps at EOC positions."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV with columns: PDB_ID, LChain, HChain, AGChain",
    )
    parser.add_argument(
        "--edges-npz",
        required=True,
        help="NPZ with per-chain edges (keys like 'PDBID_CHAINID').",
    )
    parser.add_argument(
        "--seqs-npz",
        required=True,
        help="NPZ with ESM-tokenized sequences (keys=PDB IDs).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output NPZ filename for combined edges.",
    )
    parser.add_argument(
        "--eoc-id",
        type=int,
        default=24,
        help="Token id for the EOC (end-of-chain) token [default: 24].",
    )
    parser.add_argument(
        "--pdb-id",
        help="Optional: only process this exact PDB_ID.",
    )
    args = parser.parse_args()

    csv_path = args.csv
    edges_path = args.edges_npz
    seqs_path = args.seqs_npz
    out_path = args.output
    eoc_id = args.eoc_id
    pdb_filter = args.pdb_id

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    if not os.path.exists(edges_path):
        raise FileNotFoundError(edges_path)
    if not os.path.exists(seqs_path):
        raise FileNotFoundError(seqs_path)

    pdb_data = pd.read_csv(
        csv_path,
        header=0,
        names=["PDB_ID", "LChain", "HChain", "AGChain"],
    )

    edges_npz = np.load(edges_path, allow_pickle=True)
    seqs_npz = np.load(seqs_path, allow_pickle=True)

    combined_edges_dict = {}

    for key in seqs_npz.keys():
        pdb_id = str(key)

        if pdb_filter is not None and pdb_id != pdb_filter:
            continue

        # exact match, no case mangling
        if pdb_id not in pdb_data["PDB_ID"].values:
            print(f"[warn] PDB '{pdb_id}' not found in CSV. Skipping.")
            continue

        row = pdb_data.loc[pdb_data["PDB_ID"] == pdb_id].iloc[0]

        Lchain_id = str(row["LChain"]) if pd.notna(row["LChain"]) else None
        Hchain_id = str(row["HChain"]) if pd.notna(row["HChain"]) else None
        ag_ids = (
            [c.strip() for c in str(row["AGChain"]).split(";") if c.strip()]
            if pd.notna(row["AGChain"])
            else []
        )

        chain_keys = []
        chain_labels = []

        if Lchain_id is not None:
            chain_keys.append(f"{pdb_id}_{Lchain_id}")
            chain_labels.append("Lchain")
        if Hchain_id is not None:
            chain_keys.append(f"{pdb_id}_{Hchain_id}")
            chain_labels.append("Hchain")
        for i, ag in enumerate(ag_ids):
            chain_keys.append(f"{pdb_id}_{ag}")
            chain_labels.append(f"AGchain_{i}")

        if not chain_keys:
            print(f"[warn] {pdb_id}: no chain IDs, skipping.")
            continue

        # --- Load tokenized sequence and build mapping ---
        seq_arr = seqs_npz[key]  # shape (depth, L_tokens)
        if seq_arr.ndim != 2:
            print(f"[warn] {pdb_id}: unexpected seq shape {seq_arr.shape}, skipping.")
            continue

        tokens = seq_arr[0].astype(int)  # query row
        L_tokens = tokens.shape[0]

        chain_positions = build_chain_positions(tokens, eoc_id=eoc_id)

        if len(chain_positions) < len(chain_keys):
            print(
                f"[error] {pdb_id}: only {len(chain_positions)} chains inferred from EOC, "
                f"but CSV expects {len(chain_keys)} (L/H/AG...). Skipping."
            )
            continue

        # Only take as many chains as CSV says
        chain_positions = chain_positions[: len(chain_keys)]

        # Debug: residue counts per chain
        chain_lengths = [len(pos_list) for pos_list in chain_positions]
        print(
            f"[info] {pdb_id}: token length={L_tokens}, "
            f"chain_lengths (residues)={chain_lengths}"
        )

        combined_edges_list = []
        skip_pdb = False

        for cidx, (ckey, label, pos_list) in enumerate(
            zip(chain_keys, chain_labels, chain_positions)
        ):
            if ckey not in edges_npz:
                print(
                    f"[warn] {pdb_id}: missing edges for chain key '{ckey}', "
                    f"label={label}, skipping this chain."
                )
                continue

            chain_edges = edges_npz[ckey]  # (E, 2) local residue indices
            if chain_edges.size == 0:
                continue

            n_res = len(pos_list)
            pos_array = np.array(pos_list, dtype=int)  # (n_res,)

            # sanity: local indices must be < n_res
            if chain_edges.max() >= n_res:
                print(
                    f"[error] {pdb_id}, {label}: edge index {chain_edges.max()} "
                    f">= #residues inferred from tokens ({n_res}). Skipping PDB."
                )
                skip_pdb = True
                break

            # Map local residue indices -> global token indices
            global_edges = np.empty_like(chain_edges, dtype=int)
            global_edges[:, 0] = pos_array[chain_edges[:, 0]]
            global_edges[:, 1] = pos_array[chain_edges[:, 1]]

            combined_edges_list.append(global_edges)

        if skip_pdb or not combined_edges_list:
            print(f"[warn] {pdb_id}: no combined edges generated.")
            continue

        combined_edges = np.concatenate(combined_edges_list, axis=0)

        # final sanity: indices must be < L_tokens
        if combined_edges.max() >= L_tokens:
            print(
                f"[error] {pdb_id}: combined edge index {combined_edges.max()} "
                f">= token length {L_tokens}. Skipping."
            )
            continue

        combined_edges_dict[pdb_id] = combined_edges.astype(np.int32)
        print(
            f"[ok] {pdb_id}: combined edges shape {combined_edges.shape}, "
            f"max index={combined_edges.max()}, token length={L_tokens}"
        )

    if combined_edges_dict:
        np.savez_compressed(out_path, **combined_edges_dict)
        print(f"[done] saved {len(combined_edges_dict)} entries → {out_path}")
    else:
        print("[done] no edges saved (nothing combined).")


if __name__ == "__main__":
    main()

