#!/usr/bin/env python3
import torch
import sys
import os

def remove_module_prefix(checkpoint_dict):
    """
    Removes leading 'module.' from all keys in a checkpoint dictionary.
    """
    cleaned = {}
    for k, v in checkpoint_dict.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    return cleaned


def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python clean_module_prefix.py <input.pth> <output.pth>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {input_path}")
    ckpt = torch.load(input_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        print("❌ Expected a state_dict-like dictionary.")
        sys.exit(1)

    # Remove n_averaged (for SWA checkpoints)
    ckpt.pop("n_averaged", None)

    # Fix keys
    print("Cleaning 'module.' prefixes...")
    cleaned_ckpt = remove_module_prefix(ckpt)

    # Save
    torch.save(cleaned_ckpt, output_path)
    print(f"✅ Saved cleaned checkpoint → {output_path}")


if __name__ == "__main__":
    main()

