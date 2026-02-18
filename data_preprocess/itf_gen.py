#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd


def elementor(temp, cur):
    temp_array = np.array(temp, dtype=int)
    cur_array = np.array(cur, dtype=int)
    return np.bitwise_or(temp_array, cur_array).tolist()


def safe_chain_token(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    return s[-1]


def split_ag_chains(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(";") if p.strip()]
    return [p[-1] for p in parts]


def load_interface_file(interface_dir, key, chain1, chain2, cutoff):
    c1 = chain1[-1]
    c2 = chain2[-1]
    suffix = f"interface{cutoff:g}"
    file1 = os.path.join(interface_dir, f"{key}_chain_{c1}_{c2}_{suffix}")
    file2 = os.path.join(interface_dir, f"{key}_chain_{c2}_{c1}_{suffix}")
    if os.path.exists(file1):
        return file1
    if os.path.exists(file2):
        return file2
    return None


def pad_to_len(vec, length, pad_val=-1):
    vec = list(vec) if vec is not None else []
    if len(vec) >= length:
        return vec[:length]
    return vec + [pad_val] * (length - len(vec))


def pad_to_fixed(vec: np.ndarray, fixed_len: int = 1600, pad_val: int = -1) -> np.ndarray:
    vec = np.asarray(vec).reshape(-1)
    n = vec.shape[0]
    if n == fixed_len:
        return vec.astype(int, copy=False)
    if n > fixed_len:
        return vec[:fixed_len].astype(int, copy=False)
    out = np.full((fixed_len,), pad_val, dtype=int)
    out[:n] = vec.astype(int, copy=False)
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate combined interface NPZ for training from interface text files.")
    ap.add_argument("--csv", required=True, help="CSV with header. Uses col0=PDB_ID, col1=LChain, col2=HChain, col3=AGChain.")
    ap.add_argument("--work-dir", required=True, help="Work dir (used for defaults).")
    ap.add_argument("--seq-npz", required=True, help="Combined sequences NPZ (from data_preprocess).")
    ap.add_argument("--interface-dir", default=None, help="Dir containing raw interface text files.")
    ap.add_argument("--cutoff", type=float, default=10.0, help="Must match cutoff used in Extract_interface.")
    ap.add_argument("--eoc-token-id", type=int, default=24, help="EOC token id used in sequences.")
    ap.add_argument("--pad-to", type=int, default=1600, help="Pad/truncate final vector to this length with -1.")
    ap.add_argument("--out-npz", default=None, help="Output interfaces NPZ path.")
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv)
    work_dir = os.path.abspath(args.work_dir)
    seq_npz = os.path.abspath(args.seq_npz)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"--csv not found: {csv_path}")
    if not os.path.isdir(work_dir):
        raise FileNotFoundError(f"--work-dir not found: {work_dir}")
    if not os.path.isfile(seq_npz):
        raise FileNotFoundError(f"--seq-npz not found: {seq_npz}")

    interface_dir = os.path.abspath(args.interface_dir) if args.interface_dir else os.path.join(work_dir, "interfaces_raw")
    if not os.path.isdir(interface_dir):
        raise FileNotFoundError(f"interface dir not found: {interface_dir}")

    out_npz = os.path.abspath(args.out_npz) if args.out_npz else os.path.join(work_dir, "interfaces_combined.npz")

    # Read CSV with guaranteed header line; ignore column names by using iloc
    df = pd.read_csv(csv_path, header=0)
    if df.shape[1] < 4:
        raise ValueError(
            f"CSV must have at least 4 columns (col0=PDB_ID, col1=LChain, col2=HChain, col3=AGChain). Found {df.shape[1]}"
        )

    # Build lookup: exact matching of PDB key (strip only)
    row_by_pdb = {}
    for _, r in df.iterrows():
        raw_id = r.iloc[0]
        if pd.isna(raw_id):
            continue
        pdbid = str(raw_id).strip()
        if not pdbid:
            continue
        if pdbid not in row_by_pdb:
            row_by_pdb[pdbid] = r

    print(f"[info] CSV rows loaded (header skipped, no normalization): {len(row_by_pdb)}")

    seqs = np.load(seq_npz, allow_pickle=True)

    combined_itfs = {}
    n_ok = n_missing = n_fail = 0

    for key in seqs.keys():
        key_str = str(key).strip()
        if key_str not in row_by_pdb:
            continue

        itfs = {}  # reset per PDB

        try:
            row = row_by_pdb[key_str]

            # Column positions only
            Lc = safe_chain_token(row.iloc[1])
            Hc = safe_chain_token(row.iloc[2])
            AGs = split_ag_chains(row.iloc[3])

            Lchain = f"{key_str}_{Lc}" if Lc else None
            Hchain = f"{key_str}_{Hc}" if Hc else None
            AGchains = [f"{key_str}_{ag}" for ag in AGs if ag]

            arr = seqs[key]
            EOCs = np.where(arr[0] == args.eoc_token_id)[0]

            needed = 2 + len(AGchains)
            if len(EOCs) < needed:
                print(f"⚠️  {key_str}: not enough EOC (found {len(EOCs)}, need {needed}), skip")
                n_missing += 1
                continue

            pairs = []
            if Lchain:
                pairs += [(Lchain, ag) for ag in AGchains]
            if Hchain:
                pairs += [(Hchain, ag) for ag in AGchains]

            for chain1, chain2 in pairs:
                interface_file = load_interface_file(interface_dir, key_str, chain1, chain2, args.cutoff)
                if not interface_file:
                    continue

                with open(interface_file, "r") as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        s = line.strip()
                        if s == f"chain_{chain1[-1]}_residue_one_hot":
                            cur = list(map(int, f.readline().strip().split()))
                            itfs[chain1] = cur if chain1 not in itfs else elementor(itfs[chain1], cur)
                        elif s == f"chain_{chain2[-1]}_residue_one_hot":
                            cur = list(map(int, f.readline().strip().split()))
                            itfs[chain2] = cur if chain2 not in itfs else elementor(itfs[chain2], cur)

            # Segment lengths from EOCs
            L_len = EOCs[0] + 1
            H_len = EOCs[1] - EOCs[0]
            AG_lens = [EOCs[i + 2] - EOCs[i + 1] for i in range(len(AGchains))]

            L_vec = pad_to_len(itfs.get(Lchain, []), L_len, pad_val=-1) if Lchain else [-1] * L_len
            H_vec = pad_to_len(itfs.get(Hchain, []), H_len, pad_val=-1) if Hchain else [-1] * H_len
            AG_vecs = [pad_to_len(itfs.get(ag, []), ln, pad_val=-1) for ag, ln in zip(AGchains, AG_lens)]

            combined = np.concatenate(
                [np.array(L_vec, dtype=int), np.array(H_vec, dtype=int)]
                + [np.array(v, dtype=int) for v in AG_vecs]
            )

            combined_itfs[key_str] = pad_to_fixed(combined, fixed_len=args.pad_to, pad_val=-1)
            n_ok += 1

        except Exception as e:
            n_fail += 1
            print(f"❌ Error processing {key_str}: {e}")

    if combined_itfs:
        np.savez_compressed(out_npz, **combined_itfs)
        print("\n========================================")
        print("✅ itf_gen finished.")
        print(f"   Successful: {n_ok}")
        print(f"   Missing   : {n_missing}")
        print(f"   Failed    : {n_fail}")
        print(f"   Output    : {out_npz}")
        print(f"   Pad-to    : {args.pad_to} (pad value -1)")
        print("========================================")
    else:
        print("No combined interfaces found to save.")


if __name__ == "__main__":
    main()

