import torch

# ------------------------------------------------------------------
# 1.  Paths
# ------------------------------------------------------------------
in_ckpt  = "ctsr_ss_l1_g10_i5_do0.10_dpr0.10_lr0.0001_fold1_final.pth"
out_ckpt = "ctsr_ss_l1_g10_i5_do0.10_dpr0.10_lr0.0001_fold1_final_core.pth"

# ------------------------------------------------------------------
# 2.  Load full checkpoint
# ------------------------------------------------------------------
state = torch.load(in_ckpt, map_location="cpu")

# ------------------------------------------------------------------
# 3.  Keep only the mc_model weights
#     • drop leading "module." (DDP/DataParallel) if present
# ------------------------------------------------------------------
core_state = {
#    k.replace("module.", "", 1): v          # strip single leading "module."
    k: v
    for k, v in state.items()
    if "mc_model" in k                     # keep only mc_model sub-tree
}

# ------------------------------------------------------------------
# 4.  Save
# ------------------------------------------------------------------
torch.save(core_state, out_ckpt)
print(f"Saved mc_model-only weights to →  {out_ckpt}")

