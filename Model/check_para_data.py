#!/usr/bin/env python3
"""
check_para_data.py

Checks data issues that can cause CUDA index errors:
  - sequence token IDs < 0 or >= vocab_size
  - labels not in {-1, 0, 1}
  - graph edge indices < 0 or >= seq_len
  - shape mismatches among sequences, labels, edges

Run:
    python check_para_data.py

Optional:
    python check_para_data.py --check_batches
"""

import argparse
import traceback
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from Dataloader_itf import SequenceParatopeDataset


def custom_collate_fn(batch):
    seqs, labels, edges = zip(*batch)

    seqs = torch.stack(seqs)
    labels = torch.tensor(np.array(labels), dtype=torch.long)

    max_e = max(e.shape[0] for e in edges)
    pads = []
    for e in edges:
        pad = -torch.ones((2, max_e), dtype=torch.long)
        pad[:, : e.shape[0]] = e.T.clone().detach()
        pads.append(pad)

    edges = torch.stack(pads)
    return edges, seqs, labels


def as_tensor(x):
    return x if torch.is_tensor(x) else torch.as_tensor(x)


def minmax_int(t):
    t = as_tensor(t)
    if t.numel() == 0:
        return None, None
    t = t.long()
    return int(t.min().item()), int(t.max().item())


def inspect_npz_file(path, name, max_keys=30):
    print(f"\n[NPZ] {name}: {path}")
    try:
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())
        print(f"  keys ({len(keys)}): {keys[:max_keys]}")
        for k in keys[:max_keys]:
            arr = data[k]
            print(f"  {k}: shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}")
            if isinstance(arr, np.ndarray) and arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                try:
                    print(f"      min={np.nanmin(arr)}, max={np.nanmax(arr)}")
                except Exception:
                    pass
        data.close()
    except Exception as exc:
        print(f"  Could not inspect npz: {repr(exc)}")


def check_sample(idx, seq, label, edge, args):
    problems = []

    seq = as_tensor(seq)
    label = as_tensor(label)
    edge = as_tensor(edge)

    # Shape checks
    if seq.dim() not in (1, 2):
        problems.append(f"bad seq dim: expected [L] or [M,L], got {tuple(seq.shape)}")
        L = args.seq_len
    else:
        L = seq.shape[-1]

    if L != args.seq_len:
        problems.append(f"seq length mismatch: got L={L}, expected seq_len={args.seq_len}")

    if label.dim() != 1:
        problems.append(f"bad label dim: expected [L], got {tuple(label.shape)}")
    elif label.shape[0] != L:
        problems.append(f"label length mismatch: label L={label.shape[0]}, seq L={L}")

    if edge.dim() != 2:
        problems.append(f"bad edge dim: expected [E,2] or [2,E], got {tuple(edge.shape)}")
    elif not (edge.shape[-1] == 2 or edge.shape[0] == 2):
        problems.append(f"bad edge shape: expected [E,2] or [2,E], got {tuple(edge.shape)}")

    # NaN/Inf checks if somehow floating-point
    for name, t in [("seq", seq), ("label", label), ("edge", edge)]:
        if torch.is_floating_point(t):
            if torch.isnan(t).any() or torch.isinf(t).any():
                problems.append(f"{name} contains NaN/Inf")

    # Sequence token range
    seq_long = seq.long()
    bad_seq = (seq_long < 0) | (seq_long >= args.vocab_size)
    if bad_seq.any():
        vals = seq_long[bad_seq]
        problems.append(
            f"bad sequence tokens: count={int(bad_seq.sum().item())}, "
            f"min_bad={int(vals.min().item())}, max_bad={int(vals.max().item())}"
        )

    # Labels
    label_long = label.long()
    bad_label = ~((label_long == -1) | (label_long == 0) | (label_long == 1))
    if bad_label.any():
        vals = label_long[bad_label]
        problems.append(
            f"bad labels: count={int(bad_label.sum().item())}, "
            f"min_bad={int(vals.min().item())}, max_bad={int(vals.max().item())}"
        )

    # Edge checks, normalize to [E,2]
    edge_long = edge.long()
    e2 = None
    if edge_long.dim() == 2:
        if edge_long.shape[-1] == 2:
            e2 = edge_long
        elif edge_long.shape[0] == 2:
            e2 = edge_long.T

    if e2 is not None and e2.numel() > 0:
        src = e2[:, 0]
        dst = e2[:, 1]

        malformed_pad = ((src < 0) ^ (dst < 0))
        if malformed_pad.any():
            examples = e2[malformed_pad][:5].tolist()
            problems.append(
                f"malformed padded edges where only one endpoint is negative: "
                f"count={int(malformed_pad.sum().item())}, examples={examples}"
            )

        valid_edges = (src >= 0) & (dst >= 0)
        if valid_edges.any():
            src_v = src[valid_edges]
            dst_v = dst[valid_edges]
            bad_edge = (src_v >= L) | (dst_v >= L)
            if bad_edge.any():
                bad_pairs = torch.stack([src_v[bad_edge], dst_v[bad_edge]], dim=-1)
                problems.append(
                    f"bad edge indices: count={int(bad_edge.sum().item())}, "
                    f"examples={bad_pairs[:5].tolist()}, L={L}"
                )

    return problems


def print_bad_sample(idx, seq, label, edge, problems):
    seq = as_tensor(seq)
    label = as_tensor(label)
    edge = as_tensor(edge)
    print(f"\n[BAD SAMPLE] idx={idx}")
    print(f"  seq   shape={tuple(seq.shape)}, dtype={seq.dtype}, min/max={minmax_int(seq)}")
    print(f"  label shape={tuple(label.shape)}, dtype={label.dtype}, unique={torch.unique(label.long()).detach().cpu().tolist()[:30]}")
    print(f"  edge  shape={tuple(edge.shape)}, dtype={edge.dtype}, min/max={minmax_int(edge)}")
    for p in problems:
        print(f"  - {p}")


def batch_checks(dataset, args):
    print("\n=== DataLoader batch checks ===")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0,
    )

    for bi, (e, s, l) in enumerate(loader):
        print(f"\nBatch {bi}:")
        print(f"  seq   shape={tuple(s.shape)}, min/max={minmax_int(s)}")
        print(f"  edge  shape={tuple(e.shape)}, min/max={minmax_int(e)}")
        print(f"  label shape={tuple(l.shape)}, unique={torch.unique(l.long()).detach().cpu().tolist()[:30]}")

        bad_seq = (s < 0) | (s >= args.vocab_size)
        bad_label = ~((l == -1) | (l == 0) | (l == 1))
        valid_edge = e >= 0
        bad_edge = valid_edge & (e >= args.seq_len)

        if bad_seq.any():
            idx = bad_seq.nonzero(as_tuple=False)
            vals = s[bad_seq]
            print(f"  BAD seq tokens: count={int(bad_seq.sum().item())}")
            print(f"    first values: {vals[:20].detach().cpu().tolist()}")
            print(f"    first indices: {idx[:20].detach().cpu().tolist()}")

        if bad_label.any():
            idx = bad_label.nonzero(as_tuple=False)
            vals = l[bad_label]
            print(f"  BAD labels: count={int(bad_label.sum().item())}")
            print(f"    first values: {vals[:20].detach().cpu().tolist()}")
            print(f"    first indices: {idx[:20].detach().cpu().tolist()}")

        if bad_edge.any():
            idx = bad_edge.nonzero(as_tuple=False)
            vals = e[bad_edge]
            print(f"  BAD edges: count={int(bad_edge.sum().item())}")
            print(f"    first values: {vals[:20].detach().cpu().tolist()}")
            print(f"    first indices: {idx[:20].detach().cpu().tolist()}")

        if bi >= args.max_batches - 1:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_file", default="para_tv_esmsequences_1600.npz")
    parser.add_argument("--data_file", default="para_tv_esminterfaces_1600.npz")
    parser.add_argument("--edge_file", default="para_tv_esmedges_1600.npz")
    parser.add_argument("--seq_len", type=int, default=1600)
    parser.add_argument("--vocab_size", type=int, default=31)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_bad_print", type=int, default=30)
    parser.add_argument("--stop_after_bad", type=int, default=0,
                        help="0 means scan all; otherwise stop after this many bad samples")
    parser.add_argument("--check_batches", action="store_true")
    parser.add_argument("--max_batches", type=int, default=5)
    args = parser.parse_args()

    print("=== Direct NPZ overview ===")
    inspect_npz_file(args.sequence_file, "sequence_file")
    inspect_npz_file(args.data_file, "data_file")
    inspect_npz_file(args.edge_file, "edge_file")

    print("\n=== Loading SequenceParatopeDataset ===")
    dataset = SequenceParatopeDataset(
        data_file=args.data_file,
        sequence_file=args.sequence_file,
        edge_file=args.edge_file,
        max_len=args.seq_len,
    )
    print(f"Dataset length: {len(dataset)}")

    print("\n=== Per-sample checks ===")

    counters = Counter()
    token_hist = Counter()
    label_hist = Counter()
    bad_samples = []
    seq_global_min = None
    seq_global_max = None
    edge_global_min = None
    edge_global_max = None

    for idx in range(len(dataset)):
        try:
            seq, label, edge = dataset[idx]
        except Exception as exc:
            print(f"\n[ERROR] dataset[{idx}] failed to load: {repr(exc)}")
            traceback.print_exc()
            bad_samples.append((idx, [f"dataset load error: {repr(exc)}"]))
            counters["load_error"] += 1
            continue

        seq_t = as_tensor(seq)
        label_t = as_tensor(label)
        edge_t = as_tensor(edge)

        if seq_t.numel() > 0:
            mn, mx = minmax_int(seq_t)
            seq_global_min = mn if seq_global_min is None else min(seq_global_min, mn)
            seq_global_max = mx if seq_global_max is None else max(seq_global_max, mx)

        if edge_t.numel() > 0:
            mn, mx = minmax_int(edge_t)
            edge_global_min = mn if edge_global_min is None else min(edge_global_min, mn)
            edge_global_max = mx if edge_global_max is None else max(edge_global_max, mx)

        # Histograms
        try:
            seq_long = seq_t.long()
            for v in torch.unique(seq_long).detach().cpu().tolist():
                token_hist[int(v)] += int((seq_long == int(v)).sum().item())
            label_long = label_t.long()
            for v in torch.unique(label_long).detach().cpu().tolist():
                label_hist[int(v)] += int((label_long == int(v)).sum().item())
        except Exception:
            pass

        problems = check_sample(idx, seq_t, label_t, edge_t, args)
        if problems:
            bad_samples.append((idx, problems))
            for p in problems:
                if "bad sequence tokens" in p:
                    counters["bad_sequence_tokens"] += 1
                elif "bad labels" in p:
                    counters["bad_labels"] += 1
                elif "bad edge indices" in p:
                    counters["bad_edge_indices"] += 1
                elif "malformed padded edges" in p:
                    counters["malformed_edges"] += 1
                elif "mismatch" in p:
                    counters["shape_mismatch"] += 1
                else:
                    counters["other"] += 1

            if len(bad_samples) <= args.max_bad_print:
                print_bad_sample(idx, seq_t, label_t, edge_t, problems)

        if args.stop_after_bad and len(bad_samples) >= args.stop_after_bad:
            print(f"\nStopping early after {args.stop_after_bad} bad samples.")
            break

        if (idx + 1) % 500 == 0:
            print(f"  checked {idx + 1}/{len(dataset)} samples... bad so far={len(bad_samples)}")

    print("\n=== Summary ===")
    print(f"Bad samples: {len(bad_samples)}")
    print(f"Problem counters: {dict(counters)}")
    print(f"Sequence global min/max: {seq_global_min}, {seq_global_max}")
    print(f"Edge global min/max: {edge_global_min}, {edge_global_max}")

    print("\nToken histogram sorted by token ID:")
    for k in sorted(token_hist):
        flag = ""
        if k < 0 or k >= args.vocab_size:
            flag = "  <-- OUT OF RANGE"
        print(f"  token {k:4d}: {token_hist[k]}{flag}")

    print("\nLabel histogram:")
    for k in sorted(label_hist):
        flag = ""
        if k not in (-1, 0, 1):
            flag = "  <-- BAD LABEL"
        print(f"  label {k:4d}: {label_hist[k]}{flag}")

    if bad_samples:
        print("\nFirst bad sample indices:")
        print([x[0] for x in bad_samples[:100]])

    if args.check_batches:
        batch_checks(dataset, args)

    if len(bad_samples) == 0:
        print("\nNo obvious data index problems found.")
    else:
        print("\nData problems found. Fix or filter these before GPU training.")


if __name__ == "__main__":
    main()

