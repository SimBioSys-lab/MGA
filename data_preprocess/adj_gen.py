#!/usr/bin/env python3
import os
import argparse
import numpy as np
from Bio.PDB import PDBParser, is_aa
from scipy.spatial import KDTree
import pandas as pd


def calculate_residue_adjacency_matrix(pdb_file, threshold=10.0):
    """
    Calculate residue-to-residue adjacency matrix based on atomic distances.
    A residue pair is considered connected if any atom in one residue
    is within the threshold distance of any atom in another residue.
    Only considers standard amino acids.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    # Collect atoms and residue indices
    atom_coords = []
    residue_indices = []
    residues = [res for res in structure.get_residues() if is_aa(res, standard=True)]

    for i, residue in enumerate(residues):
        for atom in residue:
            atom_coords.append(atom.coord)   # Atom coordinates
            residue_indices.append(i)        # Residue index for this atom

    if not atom_coords:
        raise ValueError(f"No standard residues/atoms found in {pdb_file}")

    atom_coords = np.array(atom_coords)
    residue_indices = np.array(residue_indices)

    # Build KDTree for atom coordinates
    tree = KDTree(atom_coords)

    # Find all pairs of atoms within the threshold
    pairs = tree.query_pairs(threshold)

    # Initialize adjacency matrix
    num_residues = len(residues)
    adjacency_matrix = np.zeros((num_residues, num_residues), dtype=int)

    # Map atom pairs to residue pairs
    for i, j in pairs:
        res_i = residue_indices[i]
        res_j = residue_indices[j]
        if res_i != res_j:  # Exclude self-loops
            adjacency_matrix[res_i, res_j] = 1
            adjacency_matrix[res_j, res_i] = 1  # Symmetric connection

    return adjacency_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Generate residue-level adjacency edge lists from per-chain PDBs."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Input CSV with columns: PDB_ID, LChain, HChain, AGChain (header is ignored, first 4 cols used)."
    )
    parser.add_argument(
        "--pdb-chains-dir",
        default="pdb_chains",
        help="Directory where per-chain PDB files live "
             "(e.g., output of process_pdb.py) [default: pdb_chains]"
    )
    parser.add_argument(
        "--output",
        default="edge_lists.npz",
        help="Output NPZ file to store all edge lists [default: edge_lists.npz]"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Distance threshold (Å) for residue-residue adjacency [default: 10.0]"
    )
    parser.add_argument(
        "--pdb-id",
        help="Optional: only process rows with this PDB_ID (case-insensitive). "
             "If not set, processes all rows in the CSV."
    )
    args = parser.parse_args()

    input_csv = args.csv
    pdb_chains_dir = args.pdb_chains_dir
    output_file = args.output
    threshold_distance = args.threshold
    pdb_id_filter = args.pdb_id.lower() if args.pdb_id else None

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"CSV not found: {input_csv}")
    if not os.path.isdir(pdb_chains_dir):
        raise NotADirectoryError(f"PDB chains directory not found: {pdb_chains_dir}")

    # Read CSV – ignore header, force 4 columns
    pdb_data = pd.read_csv(
        input_csv,
        header=0,
        usecols=[0, 1, 2, 3],
        names=["PDB_ID", "LChain", "HChain", "AGChain"]
    )

    # Build list of PDBID_chainID keys
    chains = []
    for _, row in pdb_data.iterrows():
        pdb_id = str(row["PDB_ID"]).strip()
        if not pdb_id:
            continue
        if pdb_id_filter and pdb_id.lower() != pdb_id_filter:
            continue

        if pd.notna(row["HChain"]):
            chains.append(f"{pdb_id}_{row['HChain']}")
        if pd.notna(row["LChain"]):
            chains.append(f"{pdb_id}_{row['LChain']}")

        if pd.notna(row["AGChain"]):
            AGchains = [
                f"{pdb_id}_{agchain.strip()}"
                for agchain in str(row["AGChain"]).split(';')
                if agchain.strip()
            ]
            chains.extend(AGchains)

    if not chains:
        print("No matching chains found from CSV (check --pdb-id and CSV contents).")
        return

    edge_lists = {}
    for chain in chains:
        pdbid, chainid = chain.split('_', 1)
        pdb_file_path = os.path.join(
            pdb_chains_dir,
            f"{pdbid.lower()}_chain_{chainid}.pdb"
        )

        try:
            if not os.path.exists(pdb_file_path):
                raise FileNotFoundError(f"PDB file not found: {pdb_file_path}")

            adj_matrix = calculate_residue_adjacency_matrix(
                pdb_file_path,
                threshold=threshold_distance
            )
            edge_list = np.array(np.where(adj_matrix == 1)).T  # edge pairs
            edge_lists[f"{pdbid}_{chainid}"] = edge_list
            print(f"Processed {pdbid} chain {chainid}")

        except Exception as e:
            print(f"Failed to process {pdbid} chain {chainid}: {e}")

    if edge_lists:
        np.savez_compressed(output_file, **edge_lists)
        print(f"Saved combined edge lists to {output_file}")
    else:
        print("No edge lists generated. Please check your input data or files.")


if __name__ == "__main__":
    main()

