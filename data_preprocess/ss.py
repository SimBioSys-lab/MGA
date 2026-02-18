import os
import subprocess
import pandas as pd

# Configuration
input_csv = "train_set.csv"
pdb_data = pd.read_csv(input_csv, header=None, names=["PDB_ID", "LChain", "HChain", "AGChain"])

def run_stride(pdb_file):
    """
    Runs the STRIDE command on a given PDB file and returns its output.

    Args:
        pdb_file (str): Path to the PDB file.

    Returns:
        str: Output from STRIDE command.
    """
    if not os.path.exists(pdb_file):
        print(f"PDB file not found: {pdb_file}")
        return None

    try:
        result = subprocess.run(
            ["stride", pdb_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running STRIDE on {pdb_file}: {e.stderr}")
        return None

def parse_stride_output(stride_output, pdb_code, chain_id):
    """
    Parses the output from STRIDE and extracts secondary structure information.

    Args:
        stride_output (str): STRIDE command output.
        pdb_code (str): PDB code.
        chain_id (str): Chain identifier.

    Returns:
        list: List of secondary structure data.
              Format: [(PDB_Code, Chain_ID, Residue_ID, Residue_Name, Secondary_Structure)]
    """
    secondary_structure = []

    for line in stride_output.splitlines():
        if line.startswith("ASG"):  # Lines with secondary structure assignment
            parts = line.split()
            residue_name = parts[1]
            chain = parts[2]
            residue_id = parts[3]
            sec_struct = parts[5]  # Secondary structure (H, E, C, etc.)

            if chain == chain_id:
                secondary_structure.append((pdb_code, chain, residue_id, residue_name, sec_struct))

    return secondary_structure

def process_stride(pdb_code, chain_id, output_data):
    """
    Runs STRIDE and processes secondary structure for a given chain.

    Args:
        pdb_code (str): PDB code.
        chain_id (str): Chain identifier.
        output_data (list): List to store secondary structure data.
    """
    pdb_file = f"/projects/SimBioSys/Xing/PECAN_data/train_set/pdb_chains/{pdb_code}_chain_{chain_id}.pdb"
    stride_output = run_stride(pdb_file)
    if stride_output:
        chain_data = parse_stride_output(stride_output, pdb_code, chain_id)
        output_data.extend(chain_data)

## Load data from Excel file
#abag_data = pd.read_excel("final_ab-ag_summary_short.xlsx", sheet_name="Sheet1")
#
## Validate required columns
#expected_columns = [
#    "PDB_Code", "Hchain", "Lchain", "AGchain", "compound", "species",
#    "short_header", "antigen", "method", "affinity", "deltaG", "aff_method"
#]
#missing_columns = [col for col in expected_columns if col not in abag_data.columns]
#if missing_columns:
#    raise ValueError(f"Missing expected columns: {missing_columns}")

# Collect all secondary structure data
secondary_structure_data = []

# Process each row in the dataset
for _, row in pdb_data.iterrows():
    pdb_code = row['PDB_ID'].lower()

    # Check if required chains are present
    if pd.isna(row['HChain']) or pd.isna(row['LChain']) or pd.isna(row['AGChain']):
        print(f"Skipping {pdb_code}: Missing chain data.")
        continue

    # Process heavy and light chains
    Hchain = row['HChain']
    Lchain = row['LChain']

    process_stride(pdb_code, Hchain, secondary_structure_data)
    process_stride(pdb_code, Lchain, secondary_structure_data)

    # Process antigen chains
    AGchains = [ag.strip() for ag in str(row['AGChain']).split(';') if ag.strip()]
    for chain in AGchains:
        process_stride(pdb_code, chain, secondary_structure_data)

# Save all secondary structure data to a single CSV file
output_csv = "secondary_structure_data.csv"
secondary_structure_df = pd.DataFrame(
    secondary_structure_data,
    columns=["PDB_Code", "Chain_ID", "Residue_ID", "Residue_Name", "Secondary_Structure"]
)
secondary_structure_df.to_csv(output_csv, index=False)
print(f"Secondary structure data saved to {output_csv}")

