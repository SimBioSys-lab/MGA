#!/usr/bin/env python3
import os
import numpy as np
from sklearn.cluster import KMeans
import argparse


VOCAB = "ARNDCEQGHILKMFPSTWYV-"  # A reduced vocabulary for A3M


def one_hot_encode_sequence(sequence, max_len):
    """One-hot encodes a sequence of amino acids, padding or truncating to a fixed length."""
    one_hot = np.zeros((max_len, len(VOCAB)), dtype=np.float32)

    # Ensure the sequence is upper case and padded/truncated to max_len
    sequence = sequence.upper()[:max_len].ljust(max_len, "-")  # Padding with gaps (-)

    for i, res in enumerate(sequence):
        if res in VOCAB:
            one_hot[i, VOCAB.index(res)] = 1.0
    return one_hot.flatten()  # Flatten into a 1D array


def load_msa_sequences(msa_file):
    """Loads MSA sequences from a FASTA-like file."""
    sequences = []
    try:
        with open(msa_file, "r") as f:
            current_sequence = []
            for line in f:
                if line.startswith(">"):
                    if current_sequence:
                        sequences.append("".join(current_sequence).replace(".", "").upper())
                        current_sequence = []
                else:
                    current_sequence.append(line.strip())
            if current_sequence:
                sequences.append("".join(current_sequence).replace(".", "").upper())
    except FileNotFoundError:
        print(f"File {msa_file} not found.")
    except Exception as e:
        print(f"Error loading sequences from {msa_file}: {e}")
    return sequences


def kmeans_downsample_msa(msa_file, num_sequences=64, max_len=1600):
    """Downsamples an MSA using k-means clustering on one-hot encoded sequences."""
    sequences = load_msa_sequences(msa_file)
    if not sequences or len(sequences) < num_sequences:
        print(f"{msa_file}: Less than {num_sequences} sequences available, returning all.")
        return sequences

    # One-hot encode sequences for k-means clustering (padding/truncating to max_len)
    sequence_vectors = np.array([one_hot_encode_sequence(seq, max_len) for seq in sequences])

    # Perform k-means clustering
    try:
        kmeans = KMeans(n_clusters=num_sequences, random_state=42, n_init="auto")
        kmeans.fit(sequence_vectors)

        # Check if the number of unique clusters is less than the requested number
        unique_labels = np.unique(kmeans.labels_)
        if len(unique_labels) < num_sequences:
            print(
                f"Warning: {msa_file} - Only {len(unique_labels)} unique clusters found "
                f"out of {num_sequences} requested."
            )
    except ValueError as e:
        print(f"Error in k-means clustering for file {msa_file}: {e}")
        return sequences

    # Select one representative sequence per cluster
    downsampled_sequences = [sequences[0]]  # Include the first sequence by default (query)
    for cluster_label in range(num_sequences):
        indices = np.where(kmeans.labels_ == cluster_label)[0]
        if len(indices) > 0 and sequences[indices[0]] not in downsampled_sequences:
            downsampled_sequences.append(sequences[indices[0]])  # Take first sequence in the cluster

    return downsampled_sequences


def process_msa_list(work_dir, a3m_list, num_sequences=64, max_len=1600):
    """Process all A3M files listed in a3m_files and write downsampled sequences."""
    a3m_list_path = os.path.join(work_dir, a3m_list)
    if not os.path.exists(a3m_list_path):
        raise FileNotFoundError(f"a3m list file not found: {a3m_list_path}")

    with open(a3m_list_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("No entries in a3m_files. Nothing to do.")
        return

    for line in lines:
        fname = line.strip()
        chain = os.path.splitext(fname)[0]
        msa_file = os.path.join(work_dir, fname)
        output_file = msa_file + ".DS"

        if not os.path.exists(msa_file):
            print(f"MSA file not found, skipping: {msa_file}")
            continue

        print(f"Processing {msa_file}")

        downsampled_sequences = kmeans_downsample_msa(
            msa_file, num_sequences=num_sequences, max_len=max_len
        )

        if not downsampled_sequences:
            print(f"No sequences produced for {msa_file}, skipping write.")
            continue

        # Write as FASTA with simple headers
        with open(output_file, "w") as tg:
            for i, sequence in enumerate(downsampled_sequences):
                tg.write(f">DS_{chain}_{i}\n")
                tg.write(sequence + "\n")

        print(f"Wrote downsampled MSA to {output_file} ({len(downsampled_sequences)} sequences).")


def main():
    parser = argparse.ArgumentParser(
        description="K-means downsampling of A3M MSAs for all entries listed in a3m_files."
    )
    parser.add_argument(
        "--work-dir",
        default=".",
        help="Working directory containing a3m_files and *.a3m [default: current dir]."
    )
    parser.add_argument(
        "--a3m-list",
        default="a3m_files",
        help="File with list of .a3m filenames, one per line [default: a3m_files]."
    )
    parser.add_argument(
        "--num-seq",
        type=int,
        default=64,
        help="Target number of representative sequences per MSA [default: 64]."
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1600,
        help="Maximum sequence length for one-hot encoding [default: 1600]."
    )
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    if not os.path.isdir(work_dir):
        raise NotADirectoryError(f"Work directory not found: {work_dir}")

    process_msa_list(
        work_dir=work_dir,
        a3m_list=args.a3m_list,
        num_sequences=args.num_seq,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()

