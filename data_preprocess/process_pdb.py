#!/usr/bin/env python3
import argparse
import csv
import os
from typing import Set, Iterable

from Bio import PDB
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.SeqUtils import seq1

class ChainSelect(Select):
    """Select only one chain by ID when saving with PDBIO."""
    def __init__(self, chain_id: str):
        super().__init__()
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

def parse_chain_field(text: str) -> Iterable[str]:
    """Parse a chain field that may be empty or semicolon/comma-separated."""
    if not text:
        return []
    chains = []
    for part in text.replace(",", ";").split(";"):
        c = part.strip()
        if c:
            chains.append(c)
    return chains

def extract_chain_fasta(chain, pdb_name: str, chain_id: str, outdir: str):
    """Write FASTA for a single chain (standard amino acids only)."""
    seq_chars = []
    for residue in chain:
        if PDB.is_aa(residue, standard=True):
            seq_chars.append(seq1(residue.resname))
    seq = "".join(seq_chars)

    if not seq:
        print(f"ℹ️  {pdb_name}:{chain_id} has no standard residues. Skipping FASTA.")
        return

    os.makedirs(outdir, exist_ok=True)
    fasta_filename = os.path.join(outdir, f"{pdb_name}_chain_{chain_id}.fasta")
    with open(fasta_filename, "w") as f:
        f.write(f">{pdb_name}_Chain_{chain_id}\n{seq}\n")
    print(f"✅ FASTA written: {fasta_filename}")

def extract_chain_pdb(structure, pdb_name: str, chain_id: str, outdir: str):
    """Save a single chain as a PDB file using a Select filter (no copying)."""
    os.makedirs(outdir, exist_ok=True)
    output_file = os.path.join(outdir, f"{pdb_name}_chain_{chain_id}.pdb")
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file, select=ChainSelect(chain_id))
    print(f"✅ PDB written:   {output_file}")

def process_one_pdb(pdb_path: str, pdb_name: str, allowed_chains: Set[str],
                    pdb_outdir: str, fasta_outdir: str):
    """For a PDB file, write per-chain PDB and FASTA only for allowed chains."""
    if not os.path.exists(pdb_path):
        print(f"⚠️  PDB not found: {pdb_path} (skipping {pdb_name})")
        return

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_name, pdb_path)
    except Exception as e:
        print(f"⚠️  Failed to parse {pdb_path}: {e}")
        return

    # Build quick lookup of chain objects by ID (first model by default)
    # If multiple models exist, we iterate them and handle matches in any model.
    found_any = False
    for model in structure:
        # collect chain ids present in this model
        present_ids = {str(ch.id).strip() for ch in model.get_chains()}
        # intersect with requested
        target_ids = [cid for cid in allowed_chains if cid in present_ids]
        missing_ids = [cid for cid in allowed_chains if cid not in present_ids]
        for mid in missing_ids:
            # Only log once per structure (across models); do it here for clarity
            # (We log as info rather than error because some chains could be only in other models.)
            pass

        for chain in model:
            cid = str(chain.id).strip()
            if cid not in allowed_chains:
                continue
            found_any = True
            # Save PDB (from the full structure to keep header/atoms formatting consistent)
            extract_chain_pdb(structure, pdb_name, cid, pdb_outdir)
            # Save FASTA (from this specific chain object)
            extract_chain_fasta(chain, pdb_name, cid, fasta_outdir)

    if not found_any:
        print(f"ℹ️  No requested chains found in {pdb_name}: requested {sorted(allowed_chains)}")

def main():
    ap = argparse.ArgumentParser(
        description="From a CSV (pdb_code,Light_chain,Heavy_chain,ag), extract selected chains "
                    "from PDBs: write per-chain PDBs and FASTAs."
    )
    ap.add_argument("csv", help="CSV file with columns: pdb_code,Light_chain,Heavy_chain,ag")
    ap.add_argument("--pdb-dir", default=".", help="Directory containing PDB files (default: current dir)")
    ap.add_argument("--ext", default=".pdb", help="PDB file extension (default: .pdb)")
    ap.add_argument("--pdb-outdir", default="pdb_chains", help="Output directory for per-chain PDBs")
    ap.add_argument("--fasta-outdir", default=".", help="Output directory for per-chain FASTAs")
    args = ap.parse_args()

    with open(args.csv, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"pdb_code", "Light_chain", "Heavy_chain", "ag"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            pdb_code = (row.get("pdb_code") or "").strip().lower()  # filenames are lowercase
            if not pdb_code:
                print("⚠️  Row with empty pdb_code, skipping.")
                continue

            # L/H are single chain IDs; ag may be multiple like "A;B"
            light = [c for c in [row.get("Light_chain", "").strip()] if c]
            heavy = [c for c in [row.get("Heavy_chain", "").strip()] if c]
            ags   = list(parse_chain_field(row.get("ag", "").strip()))

            allowed = set(light + heavy + ags)
            if not allowed:
                print(f"ℹ️  {pdb_code}: no chains listed; skipping.")
                continue

            pdb_name = pdb_code
            pdb_path = os.path.join(args.pdb_dir, f"{pdb_name}{args.ext}")
            process_one_pdb(pdb_path, pdb_name, allowed, args.pdb_outdir, args.fasta_outdir)

if __name__ == "__main__":
    main()

