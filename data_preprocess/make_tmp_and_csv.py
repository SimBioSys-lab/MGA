'''
python make_tmp_and_csv.py \
  --ab-pdb antibody.pdb \
  --ag-pdb antigen.pdb \
  --lchain L \
  --hchain H \
  --agchains A,B \
  --out-pdb tmp.pdb \
  --out-csv train_set.csv \
  --pdb-id tmp
'''


#!/usr/bin/env python3
import argparse
import os
import copy
import csv

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model


def combine_pdbs(ab_pdb, ag_pdb, out_pdb, pdb_id):
    """
    Combine antibody and antigen PDBs into a single PDB file.

    - ab_pdb: path to antibody PDB
    - ag_pdb: path to antigen PDB
    - out_pdb: output combined PDB path (e.g. tmp.pdb)
    - pdb_id: structure id (string, e.g. 'tmp')
    """
    parser = PDBParser(QUIET=True)

    ab_struct = parser.get_structure("AB", ab_pdb)
    ag_struct = parser.get_structure("AG", ag_pdb)

    # Create a new empty structure with one model
    new_struct = Structure(pdb_id)
    model = Model(0)
    new_struct.add(model)

    # Add chains from antibody PDB
    ab_model = next(ab_struct.get_models())
    for chain in ab_model:
        model.add(copy.deepcopy(chain))

    # Add chains from antigen PDB
    ag_model = next(ag_struct.get_models())
    for chain in ag_model:
        model.add(copy.deepcopy(chain))

    # Warn if duplicate chain IDs exist
    chain_ids = [ch.id for ch in model.get_chains()]
    if len(chain_ids) != len(set(chain_ids)):
        print("⚠️  Warning: duplicate chain IDs exist in combined structure:",
              chain_ids)

    io = PDBIO()
    io.set_structure(new_struct)
    io.save(out_pdb)
    print(f"✅ Combined PDB written to {out_pdb}")


def normalize_agchains(ag_str: str) -> str:
    """
    Normalize AG chain string to semicolon-separated (e.g. 'A;B;C').

    Accepts input like:
      'A;B', 'A,B', ' A ; B ', etc.
    """
    if not ag_str:
        return ""
    # unify separators to ';'
    ag_str = ag_str.replace(",", ";")
    parts = [p.strip() for p in ag_str.split(";") if p.strip()]
    return ";".join(parts)


def write_csv(csv_path, pdb_id, lchain, hchain, agchains):
    """
    Write train_set.csv with a single row:
    PDB_ID,LChain,HChain,AGChain
    """
    header = ["PDB_ID", "LChain", "HChain", "AGChain"]

    # If file exists and has content, append; otherwise write header
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(header)
        writer.writerow([pdb_id, lchain, hchain, agchains])

    print(f"✅ CSV row written to {csv_path}: {pdb_id}, {lchain}, {hchain}, {agchains}")


def main():
    ap = argparse.ArgumentParser(
        description="Combine antibody and antigen PDBs into 'tmp.pdb' "
                    "and generate train_set.csv based on chain IDs."
    )
    ap.add_argument("--ab-pdb", required=True,
                    help="Antibody PDB file path (contains L and H chains).")
    ap.add_argument("--ag-pdb", required=True,
                    help="Antigen PDB file path.")
    ap.add_argument("--lchain", required=True,
                    help="Light chain ID (e.g. L).")
    ap.add_argument("--hchain", required=True,
                    help="Heavy chain ID (e.g. H).")
    ap.add_argument("--agchains", required=True,
                    help="Antigen chain ID(s). Use ',' or ';' as separators, e.g. 'A' or 'A;B'.")
    ap.add_argument("--out-pdb", default="tmp.pdb",
                    help="Output combined PDB filename [default: tmp.pdb].")
    ap.add_argument("--out-csv", default="train_set.csv",
                    help="Output CSV filename [default: train_set.csv].")
    ap.add_argument("--pdb-id", default="tmp",
                    help="PDB_ID to use in CSV [default: tmp].")

    args = ap.parse_args()

    # Normalize AG chains to semicolon-separated
    ag_field = normalize_agchains(args.agchains)

    # 1) Combine PDBs into tmp.pdb
    combine_pdbs(args.ab_pdb, args.ag_pdb, args.out_pdb, args.pdb_id)

    # 2) Write / append the CSV row
    write_csv(args.out_csv, args.pdb_id, args.lchain, args.hchain, ag_field)


if __name__ == "__main__":
    main()

