#!/usr/bin/env python3
"""
check_ctsr_data_model_alignment.py

Standalone diagnostic for CTSR/MGA dataloader + model compatibility.

Checks:
  - .npz files open
  - dataset samples load
  - sequence/label/edge shapes
  - PAD/EOC positions and labels
  - edge validity and edges touching PAD/EOC
  - DataLoader collation
  - one model forward pass

Examples:
  python check_ctsr_data_model_alignment.py \
    --task ss \
    --model-file Models_fullnew \
    --sequence-file para_tv_esmsequences_1600.npz \
    --data-file para_tv_esmss_1600.npz \
    --edge-file para_tv_esmedges_1600.npz \
    --num-classes 8 \
    --num-workers 0

  python check_ctsr_data_model_alignment.py \
    --task ctm \
    --model-file Models_fullnew \
    --sequence-file para_tv_esmsequences_1600.npz \
    --data-file global_maps_para_esmtv.npz \
    --edge-file para_tv_esmedges_1600.npz \
    --num-classes 2 \
    --num-workers 0
"""

import argparse
import importlib
import os
import traceback
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["ss", "ctm", "classification"], default="ss")
    p.add_argument("--dataloader-module", default=None)
    p.add_argument("--model-file", default="Models_fullnew")
    p.add_argument("--sequence-file", required=True)
    p.add_argument("--data-file", required=True)
    p.add_argument("--edge-file", required=True)
    p.add_argument("--seq-len", "--max-len", dest="seq_len", type=int, default=1600)
    p.add_argument("--vocab-size", type=int, default=31)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--num-gnn-layers", type=int, default=10)
    p.add_argument("--num-int-layers", type=int, default=5)
    p.add_argument("--drop-path-rate", type=float, default=0.10)
    p.add_argument("--num-classes", type=int, default=8)
    p.add_argument("--pad-id", type=int, default=1)
    p.add_argument("--eoc-id", type=int, default=24)
    p.add_argument("--ignore-index", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--batch-checks", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-forward", action="store_true")
    p.add_argument("--pretrained-ckpt", default=None)
    return p.parse_args()


def first_pad_valid_len(tokens, pad_id):
    tokens = torch.as_tensor(tokens).long()
    pad_pos = (tokens == pad_id).nonzero(as_tuple=True)[0]
    return int(pad_pos[0].item()) if pad_pos.numel() > 0 else int(tokens.numel())


def spans_fixed(tokens, eoc_id=24, pad_id=1):
    tokens = torch.as_tensor(tokens).long()
    valid_len = first_pad_valid_len(tokens, pad_id)
    toks = tokens[:valid_len]
    eocs = (toks == eoc_id).nonzero(as_tuple=True)[0].tolist()
    spans = []
    start = 0
    for eoc in eocs:
        eoc = int(eoc)
        if eoc > start:
            spans.append((start, eoc))
        start = eoc + 1
    if start < valid_len:
        spans.append((start, valid_len))
    return spans


def spans_right_eoc(tokens, eoc_id=24, pad_id=1):
    tokens = torch.as_tensor(tokens).long()
    valid_len = first_pad_valid_len(tokens, pad_id)
    toks = tokens[:valid_len]
    eocs = (toks == eoc_id).nonzero(as_tuple=True)[0].tolist()
    spans = []
    start = 0
    for eoc in eocs:
        eoc = int(eoc)
        if eoc >= start:
            spans.append((start, eoc + 1))
        start = eoc + 1
    if start < valid_len:
        spans.append((start, valid_len))
    return spans


def spans_left_eoc(tokens, eoc_id=24, pad_id=1):
    tokens = torch.as_tensor(tokens).long()
    spans = []
    for start, end in spans_fixed(tokens, eoc_id, pad_id):
        ctx = start - 1 if start > 0 and int(tokens[start - 1].item()) == eoc_id else start
        spans.append((ctx, end))
    return spans


def to_edge_tensor(edge):
    if torch.is_tensor(edge):
        e = edge.detach().cpu().long()
    else:
        e = torch.as_tensor(edge, dtype=torch.long)
    if e.ndim != 2:
        raise ValueError(f"edge must be 2D, got {tuple(e.shape)}")
    if e.shape[0] == 2:
        return e.contiguous()
    if e.shape[1] == 2:
        return e.T.contiguous()
    raise ValueError(f"edge shape must be [E,2] or [2,E], got {tuple(e.shape)}")


def collate_fn(batch):
    seqs, labels, edges = zip(*batch)
    seqs = torch.stack([torch.as_tensor(x).long() for x in seqs])
    labels = torch.as_tensor(np.array(labels), dtype=torch.long)
    max_e = max(to_edge_tensor(e).shape[1] for e in edges)
    pads = []
    for e in edges:
        ei = to_edge_tensor(e)
        pad = -torch.ones((2, max_e), dtype=torch.long)
        pad[:, :ei.shape[1]] = ei
        pads.append(pad)
    return torch.stack(pads), seqs, labels


def label_counter(x):
    return Counter(torch.as_tensor(x).long().view(-1).tolist())


def inspect_sample(idx, sample, args):
    seq, label, edge = sample
    seq = torch.as_tensor(seq).long()
    label = torch.as_tensor(label).long()
    edge = to_edge_tensor(edge)

    if seq.ndim != 2:
        raise ValueError(f"sample {idx}: seq expected [S,L], got {tuple(seq.shape)}")

    S, L = seq.shape
    query = seq[0]
    valid_len = first_pad_valid_len(query, args.pad_id)
    eoc_pos = (query[:valid_len] == args.eoc_id).nonzero(as_tuple=True)[0].tolist()
    pad_pos = (query == args.pad_id).nonzero(as_tuple=True)[0].tolist()

    src, dst = edge[0], edge[1]
    nonpad_edge = (src >= 0) & (dst >= 0)
    edge_valid = nonpad_edge & (src < L) & (dst < L)
    bad_edges = int((nonpad_edge & ~edge_valid).sum().item())
    good_edges = int(edge_valid.sum().item())

    eoc_set = set(map(int, eoc_pos))
    pad_set = set(map(int, pad_pos))
    touch_eoc = 0
    touch_pad = 0
    if good_edges > 0:
        for a, b in zip(src[edge_valid].tolist(), dst[edge_valid].tolist()):
            if int(a) in eoc_set or int(b) in eoc_set:
                touch_eoc += 1
            if int(a) in pad_set or int(b) in pad_set:
                touch_pad += 1

    eoc_label_counts = Counter()
    pad_label_counts = Counter()
    if label.ndim == 1 and label.shape[0] == L:
        for p in eoc_pos:
            eoc_label_counts[int(label[p].item())] += 1
        for p in pad_pos[:50]:
            pad_label_counts[int(label[p].item())] += 1
    elif label.ndim == 2 and label.shape[0] == L and label.shape[1] == L:
        for p in eoc_pos:
            vals = torch.cat([label[p, :], label[:, p]])
            eoc_label_counts.update(vals.view(-1).tolist())
        for p in pad_pos[:10]:
            vals = torch.cat([label[p, :], label[:, p]])
            pad_label_counts.update(vals.view(-1).tolist())

    bad_tokens = int(((seq < 0) | (seq >= args.vocab_size)).sum().item())
    lab = label.view(-1)
    valid_labels = int((lab != args.ignore_index).sum().item())

    return {
        "idx": idx,
        "seq_shape": tuple(seq.shape),
        "label_shape": tuple(label.shape),
        "edge_shape": tuple(edge.shape),
        "valid_len": valid_len,
        "first_pad": pad_pos[0] if pad_pos else None,
        "num_eoc": len(eoc_pos),
        "eoc_pos": list(map(int, eoc_pos)),
        "token_min": int(seq.min().item()),
        "token_max": int(seq.max().item()),
        "bad_tokens": bad_tokens,
        "label_counts": dict(label_counter(label)),
        "valid_labels": valid_labels,
        "total_labels": int(lab.numel()),
        "good_edges": good_edges,
        "bad_edges": bad_edges,
        "edge_touch_eoc": touch_eoc,
        "edge_touch_pad": touch_pad,
        "eoc_label_counts": dict(eoc_label_counts),
        "pad_label_counts": dict(pad_label_counts),
        "spans_fixed": spans_fixed(query, args.eoc_id, args.pad_id),
        "spans_right_eoc": spans_right_eoc(query, args.eoc_id, args.pad_id),
        "spans_left_eoc": spans_left_eoc(query, args.eoc_id, args.pad_id),
    }


def print_summary(s):
    print("\n" + "=" * 80)
    print(f"sample={s['idx']} seq={s['seq_shape']} label={s['label_shape']} edge={s['edge_shape']}")
    print(f"valid_len={s['valid_len']} first_pad={s['first_pad']} num_eoc={s['num_eoc']} eoc_pos={s['eoc_pos']}")
    print(f"token_range=[{s['token_min']},{s['token_max']}] bad_tokens={s['bad_tokens']}")
    print(f"labels valid/total={s['valid_labels']}/{s['total_labels']} counts={s['label_counts']}")
    print(f"edges good={s['good_edges']} bad={s['bad_edges']} touch_eoc={s['edge_touch_eoc']} touch_pad={s['edge_touch_pad']}")
    print(f"labels at/touching EOC={s['eoc_label_counts']}")
    print(f"labels at/touching PAD={s['pad_label_counts']}")
    print(f"spans fixed exclude EOC : {s['spans_fixed']} lens={[b-a for a,b in s['spans_fixed']]}")
    print(f"spans include EOC right: {s['spans_right_eoc']} lens={[b-a for a,b in s['spans_right_eoc']]}")
    print(f"spans include EOC left : {s['spans_left_eoc']} lens={[b-a for a,b in s['spans_left_eoc']]}")


def load_state_flexible(model, path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        for k in ("model_state_dict", "state_dict", "model", "model_state"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    if len(ckpt) > 0 and next(iter(ckpt.keys())).startswith("module."):
        ckpt = {k[len("module."):]: v for k, v in ckpt.items()}
    dst = model.state_dict()
    loaded, skip_missing, skip_shape = {}, [], []
    for k, v in ckpt.items():
        if k not in dst:
            skip_missing.append(k)
        elif tuple(dst[k].shape) != tuple(v.shape):
            skip_shape.append((k, tuple(v.shape), tuple(dst[k].shape)))
        else:
            loaded[k] = v
    dst.update(loaded)
    model.load_state_dict(dst, strict=True)
    print(f"checkpoint load: loaded={len(loaded)} skip_missing={len(skip_missing)} skip_shape={len(skip_shape)}")
    if skip_missing[:10]:
        print("first skip_missing:", skip_missing[:10])
    if skip_shape[:10]:
        print("first skip_shape:", skip_shape[:10])


def build_model(args):
    mod = importlib.import_module(args.model_file)
    cls = getattr(mod, "CTMModel" if args.task == "ctm" else "ClassificationModel")
    return cls(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_layers=args.num_layers,
        num_gnn_layers=args.num_gnn_layers,
        num_classes=args.num_classes,
        num_int_layers=args.num_int_layers,
        drop_path_rate=args.drop_path_rate,
    )


def main():
    args = parse_args()
    if args.dataloader_module is None:
        args.dataloader_module = "Dataloader_ctm" if args.task == "ctm" else "Dataloader_itf"

    print("ARGS")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    print("\n[1] NPZ open check")
    for p in (args.sequence_file, args.data_file, args.edge_file):
        print(f"  opening {p}")
        with np.load(p, allow_pickle=True) as z:
            print(f"    keys={len(z.files)} first={z.files[:5]}")

    print("\n[2] Dataset load")
    dl_mod = importlib.import_module(args.dataloader_module)
    Dataset = getattr(dl_mod, "SequenceParatopeDataset")
    dataset = Dataset(
        data_file=args.data_file,
        sequence_file=args.sequence_file,
        edge_file=args.edge_file,
        max_len=args.seq_len,
    )
    print("dataset length:", len(dataset))

    print("\n[3] Individual samples")
    agg = defaultdict(int)
    n = min(args.num_samples, len(dataset))
    for i in range(n):
        try:
            s = inspect_sample(i, dataset[i], args)
        except Exception:
            print(f"FAILED sample {i}")
            traceback.print_exc()
            raise
        print_summary(s)
        for k in ("bad_tokens", "bad_edges", "edge_touch_eoc", "edge_touch_pad", "num_eoc"):
            agg[k] += int(s[k])
        agg["valid_labels"] += int(s["valid_labels"])
        agg["total_labels"] += int(s["total_labels"])

    print("\nAggregate over inspected samples")
    for k, v in agg.items():
        print(f"  {k}: {v}")

    print("\n[4] DataLoader batch check")
    subset_n = min(len(dataset), max(args.batch_size * args.batch_checks, args.batch_size))
    loader = DataLoader(
        Subset(dataset, list(range(subset_n))),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    first_batch = None
    for bi, (edges, seqs, labels) in enumerate(loader):
        print(f"batch {bi}: edges={tuple(edges.shape)} seqs={tuple(seqs.shape)} labels={tuple(labels.shape)}")
        print(f"  seq range: {int(seqs.min())}..{int(seqs.max())}")
        print(f"  edge range: {int(edges.min())}..{int(edges.max())}")
        print(f"  label counts: {dict(label_counter(labels))}")
        if first_batch is None:
            first_batch = (edges, seqs, labels)
        if bi + 1 >= args.batch_checks:
            break

    if args.no_forward:
        print("\nDONE: skipped model forward")
        return

    print("\n[5] Model forward check")
    model = build_model(args).to(args.device).eval()
    if args.pretrained_ckpt:
        load_state_flexible(model, args.pretrained_ckpt, args.device)

    edges, seqs, labels = first_batch
    edges = edges.to(args.device)
    seqs = seqs.to(args.device)

    with torch.no_grad():
        if args.task == "ctm":
            out = model(sequences=seqs, padded_edges=edges, return_attention=True)
            logits = out[0] if isinstance(out, tuple) else out
            print("CTM logits shape:", tuple(logits.shape), "expected [B,C,L,L]")
        else:
            logits = model(sequences=seqs, padded_edges=edges, return_attention=False)
            print("classification logits shape:", tuple(logits.shape), "expected [B,1,L,C]")

    print("\nDONE: data and model forward look runnable")


if __name__ == "__main__":
    main()

