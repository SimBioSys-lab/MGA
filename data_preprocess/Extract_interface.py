#!/usr/bin/env python3
import os
import re
import csv
import argparse
import numpy as np
from Bio import PDB
from Bio.PDB.Polypeptide import is_aa


def read_pdb_ids_from_csv_first_col(csv_path):
    pdb_ids = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            val = str(row[0]).strip()
            if not val:
                continue
            pdb_ids.append(val)
    # de-dup but keep stable order
    seen = set()
    out = []
    for x in pdb_ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def resolve_pdb_path(pdb_id: str, pdb_dir: str | None, ext: str | None):
    # if already a valid file path
    if os.path.isfile(pdb_id):
        return pdb_id

    name = pdb_id.strip()
    if ext:
        if not ext.startswith("."):
            ext = "." + ext
        if not name.lower().endswith(ext.lower()):
            name = name + ext

    if pdb_dir:
        return os.path.join(pdb_dir, name)
    return name


def Form_interface(rlist, llist, right_list, left_list, right_count, left_count, cut_off=10.0):
    cut_off_sq = float(cut_off) ** 2

    r_coords = rlist[:, :3].astype(float)
    l_coords = llist[:, :3].astype(float)

    dist_matrix = np.sum((r_coords[:, np.newaxis, :] - l_coords[np.newaxis, :, :]) ** 2, axis=-1)
    close_contact_indices = np.where(dist_matrix <= cut_off_sq)
    r_index = set(close_contact_indices[0])
    l_index = set(close_contact_indices[1])

    right_atom_one_hot = [0] * right_count
    left_atom_one_hot = [0] * left_count

    right_residue_ids = set()
    left_residue_ids = set()

    for i in r_index:
        right_atom_one_hot[int(rlist[i, 4])] = 1
        right_residue_ids.add(str(rlist[i, 5]))

    for i in l_index:
        left_atom_one_hot[int(llist[i, 4])] = 1
        left_residue_ids.add(str(llist[i, 5]))

    # unique residues in order
    unique_right_residues = []
    seen_right = set()
    for res in right_list:
        rid = res[1]
        if rid not in seen_right:
            unique_right_residues.append(rid)
            seen_right.add(rid)

    unique_left_residues = []
    seen_left = set()
    for res in left_list:
        rid = res[1]
        if rid not in seen_left:
            unique_left_residues.append(rid)
            seen_left.add(rid)

    right_residue_one_hot = [1 if res in right_residue_ids else 0 for res in unique_right_residues]
    left_residue_one_hot = [1 if res in left_residue_ids else 0 for res in unique_left_residues]

    return (
        right_atom_one_hot,
        left_atom_one_hot,
        right_residue_one_hot,
        left_residue_one_hot,
        right_residue_ids,
        left_residue_ids,
    )


def Write_one_hot(interface_data, pdb_name, cut_off, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"interface{cut_off:g}"

    for (chain_1, chain_2), data in interface_data.items():
        out_path = os.path.join(out_dir, f"{pdb_name}_chain_{chain_1}_{chain_2}_{suffix}")
        with open(out_path, "w") as f:
            f.write(f"chain_{chain_1}_atom_one_hot\n")
            f.write(" ".join(map(str, data["right_atom_one_hot"])) + "\n")
            f.write(f"chain_{chain_2}_atom_one_hot\n")
            f.write(" ".join(map(str, data["left_atom_one_hot"])) + "\n")
            f.write(f"chain_{chain_1}_residue_one_hot\n")
            f.write(" ".join(map(str, data["right_residue_one_hot"])) + "\n")
            f.write(f"chain_{chain_2}_residue_one_hot\n")
            f.write(" ".join(map(str, data["left_residue_one_hot"])) + "\n")
            f.write(f"chain_{chain_1}_residue_ids\n")
            f.write(" ".join(map(str, list(data["right_residue_ids"]))) + "\n")
            f.write(f"chain_{chain_2}_residue_ids\n")
            f.write(" ".join(map(str, list(data["left_residue_ids"]))) + "\n")

        print(f"[wrote] {out_path}")


def Extract_Interface(pdb_file, cut_off, out_dir):
    pdb_base = os.path.basename(pdb_file)
    pdb_name = os.path.splitext(pdb_base)[0]

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, pdb_file)

    chain_data = {}
    interface_data = {}

    for model in structure:
        for chain in model:
            chain_id = chain.id
            atom_list = []
            residue_list = []
            atom_count = 0
            contains_standard_residue = False

            for residue in chain:
                if is_aa(residue, standard=True):
                    contains_standard_residue = True
                    _, res_seq_number, insertion_code = residue.id
                    residue_id = f"{res_seq_number}{insertion_code}".strip()
                    residue_list.append((chain_id, residue_id))
                    for atom in residue.get_atoms():
                        if atom.element == "H":
                            continue
                        x, y, z = atom.coord
                        atom_type = atom.element
                        atom_list.append([x, y, z, atom_type, atom_count, residue_id])
                        atom_count += 1

            if contains_standard_residue:
                chain_data[chain_id] = (atom_list, residue_list, atom_count)

    chain_ids = list(chain_data.keys())

    for i in range(len(chain_ids)):
        for j in range(i + 1, len(chain_ids)):
            c1 = chain_ids[i]
            c2 = chain_ids[j]
            rlist, right_list, count_r = chain_data[c1]
            llist, left_list, count_l = chain_data[c2]

            if len(rlist) == 0 or len(llist) == 0:
                continue

            (
                right_atom_one_hot,
                left_atom_one_hot,
                right_residue_one_hot,
                left_residue_one_hot,
                right_residue_ids,
                left_residue_ids,
            ) = Form_interface(
                np.array(rlist, dtype=object),
                np.array(llist, dtype=object),
                right_list,
                left_list,
                count_r,
                count_l,
                cut_off=cut_off,
            )

            interface_data[(c1, c2)] = {
                "right_atom_one_hot": right_atom_one_hot,
                "left_atom_one_hot": left_atom_one_hot,
                "right_residue_one_hot": right_residue_one_hot,
                "left_residue_one_hot": left_residue_one_hot,
                "right_residue_ids": right_residue_ids,
                "left_residue_ids": left_residue_ids,
            }

    Write_one_hot(interface_data, pdb_name, cut_off=cut_off, out_dir=out_dir)


def main():
    ap = argparse.ArgumentParser(description="Extract chain-chain interfaces for PDBs listed in CSV (first col).")
    ap.add_argument("--csv", required=True, help="CSV; first column contains PDB IDs or PDB filenames.")
    ap.add_argument("--work-dir", required=True, help="Work dir (used for defaults).")
    ap.add_argument("--pdb-dir", default=None, help="Directory containing PDB files (optional).")
    ap.add_argument("--ext", default=".pdb", help="Append extension if missing (default: .pdb).")
    ap.add_argument("--cutoff", type=float, default=10.0, help="Distance cutoff in Angstrom.")
    ap.add_argument("--interface-dir", default=None, help="Where to write interface text files.")
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv)
    work_dir = os.path.abspath(args.work_dir)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"--csv not found: {csv_path}")
    if not os.path.isdir(work_dir):
        raise FileNotFoundError(f"--work-dir not found: {work_dir}")

    pdb_dir = os.path.abspath(args.pdb_dir) if args.pdb_dir else None
    if pdb_dir and not os.path.isdir(pdb_dir):
        raise FileNotFoundError(f"--pdb-dir is not a directory: {pdb_dir}")

    interface_dir = os.path.abspath(args.interface_dir) if args.interface_dir else os.path.join(work_dir, "interfaces_raw")
    os.makedirs(interface_dir, exist_ok=True)

    pdb_ids = read_pdb_ids_from_csv_first_col(csv_path)
    if not pdb_ids:
        print("No PDB IDs found in CSV first column.")
        return

    n_ok = n_missing = n_fail = 0
    for i, pdb_id in enumerate(pdb_ids, start=1):
        pdb_path = resolve_pdb_path(pdb_id, pdb_dir=pdb_dir, ext=args.ext)

        print(f"\n#### [{i}/{len(pdb_ids)}] {pdb_id} -> {pdb_path} ####")
        if not os.path.isfile(pdb_path):
            n_missing += 1
            print(f"⚠️  Missing PDB file: {pdb_path}")
            continue

        try:
            Extract_Interface(pdb_path, cut_off=args.cutoff, out_dir=interface_dir)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"❌ Failed on {pdb_id}: {e}")

    print("\n========================================")
    print("🎯 Extract_interface finished.")
    print(f"   Successful: {n_ok}")
    print(f"   Missing   : {n_missing}")
    print(f"   Failed    : {n_fail}")
    print(f"   Output dir: {interface_dir}")
    print("========================================")


if __name__ == "__main__":
    main()

