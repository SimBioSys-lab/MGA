import torch

# Load the checkpoint file (using CPU)
state_dict = torch.load('iPara_balanced_l1_g10_i5_do0.10_dpr0.10_lr0.0002_heads16_fold8_core.pth', map_location=torch.device('cpu'))

# Get the keys 
keys = list(state_dict.keys())

# Print the keys
print(keys)

num_params = sum(p.numel() for p in state_dict.values())
print("Total parameters:", num_params)

