#!/usr/bin/env python3
"""
Check learned ESM fusion alpha values from a saved MGA checkpoint.

Works without instantiating the model, so it does not need to load ESM2.
It simply reads alpha parameters from the checkpoint state_dict.

Usage:
  python check_esm_alpha.py fold1_best_aupr.pth
  python check_esm_alpha.py checkpoints/fold1_best_aupr.pth --show-adapter
  python check_esm_alpha.py *.pth
"""

import argparse
import os
import sys
from typing import Any, Dict, Iterable, Tuple

import torch


def unwrap_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """Return a state_dict from common checkpoint formats."""
    if isinstance(obj, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "module",
        ]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]

        # Maybe the checkpoint itself is already a state_dict.
        if all(isinstance(k, str) for k in obj.keys()):
            tensor_like = [v for v in obj.values() if torch.is_tensor(v)]
            if len(tensor_like) > 0:
                return obj

    raise ValueError("Could not find a model state_dict inside checkpoint.")


def scalar_value(x: torch.Tensor) -> float:
    x = x.detach().cpu().float().reshape(-1)
    if x.numel() == 0:
        return float("nan")
    return float(x[0].item())


def clean_key(k: str) -> str:
    # Remove DataParallel/DDP prefix for display/matching.
    while k.startswith("module."):
        k = k[len("module."):]
    return k


def find_alpha_keys(sd: Dict[str, torch.Tensor]) -> Iterable[Tuple[str, torch.Tensor]]:
    for k, v in sd.items():
        ck = clean_key(k)
        if "esm_query_fusion" in ck and "alpha" in ck and torch.is_tensor(v):
            yield ck, v


def find_adapter_keys(sd: Dict[str, torch.Tensor]) -> Iterable[Tuple[str, torch.Tensor]]:
    for k, v in sd.items():
        ck = clean_key(k)
        if "esm_query_fusion" in ck and "adapter" in ck and torch.is_tensor(v):
            yield ck, v


def summarize_file(path: str, show_adapter: bool = False) -> None:
    print("=" * 80)
    print(f"Checkpoint: {path}")

    try:
        ckpt = torch.load(path, map_location="cpu")
        sd = unwrap_state_dict(ckpt)
    except Exception as e:
        print(f"ERROR: failed to load checkpoint: {e}")
        return

    alpha_items = list(find_alpha_keys(sd))

    if not alpha_items:
        print("No ESM alpha parameters found.")
        print("Expected keys like:")
        print("  mc_model.cg_model.esm_query_fusion.alpha")
        print("  mc_model.cg_model.esm_query_fusion.alpha_l")
        print("  mc_model.cg_model.esm_query_fusion.alpha_h")
        print("  mc_model.cg_model.esm_query_fusion.alpha_ag")
        print("\nAvailable ESM-related keys:")
        keys = [clean_key(k) for k in sd.keys() if "esm" in clean_key(k).lower()]
        for k in keys[:50]:
            print(" ", k)
        if len(keys) > 50:
            print(f"  ... {len(keys) - 50} more")
        return

    print("Found alpha parameters:")
    for k, v in sorted(alpha_items):
        val = scalar_value(v)
        print(f"  {k:70s} {val:+.8f}")

    # Friendly regional summary if names match regional-alpha version.
    vals = {k.split(".")[-1]: scalar_value(v) for k, v in alpha_items}
    if any(name in vals for name in ["alpha_l", "alpha_h", "alpha_ag"]):
        print("\nRegional alpha summary:")
        print(f"  Lchain:   {vals.get('alpha_l', float('nan')):+.8f}")
        print(f"  Hchain:   {vals.get('alpha_h', float('nan')):+.8f}")
        print(f"  AGchains: {vals.get('alpha_ag', float('nan')):+.8f}")
    elif "alpha" in vals:
        print("\nGlobal alpha summary:")
        print(f"  alpha: {vals['alpha']:+.8f}")

    abs_vals = [abs(scalar_value(v)) for _, v in alpha_items]
    max_abs = max(abs_vals) if abs_vals else 0.0
    print("\nInterpretation:")
    if max_abs < 1e-4:
        print("  Alpha is essentially zero: ESM branch was almost unused.")
    elif max_abs < 1e-2:
        print("  Alpha moved slightly: ESM branch is weakly used.")
    elif max_abs < 5e-2:
        print("  Alpha is clearly nonzero: ESM branch contributes modestly.")
    else:
        print("  Alpha is large enough that ESM likely affects predictions substantially.")

    if show_adapter:
        adapter_items = list(find_adapter_keys(sd))
        print("\nAdapter parameter norms:")
        if not adapter_items:
            print("  No adapter parameters found.")
        else:
            for k, v in sorted(adapter_items):
                vf = v.detach().cpu().float()
                norm = float(vf.norm().item())
                mean = float(vf.mean().item()) if vf.numel() else float("nan")
                std = float(vf.std().item()) if vf.numel() > 1 else 0.0
                print(f"  {k:70s} norm={norm:.6f} mean={mean:+.6e} std={std:.6e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ESM fusion alpha values in MGA checkpoints.")
    parser.add_argument("checkpoints", nargs="+", help="One or more .pth checkpoint files")
    parser.add_argument("--show-adapter", action="store_true", help="Also print ESM adapter parameter norms")
    args = parser.parse_args()

    for path in args.checkpoints:
        if not os.path.exists(path):
            print("=" * 80)
            print(f"Checkpoint: {path}")
            print("ERROR: file does not exist")
            continue
        summarize_file(path, show_adapter=args.show_adapter)


if __name__ == "__main__":
    main()

