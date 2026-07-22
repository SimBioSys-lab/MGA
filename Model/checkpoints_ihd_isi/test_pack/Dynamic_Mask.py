import torch
import torch.nn as nn

class DyM(nn.Module):
    """
    Dynamic Mask: applies to all tokens (including CLS), 
    initialized to identity (mask=1), logs basic statistics each forward.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.mask_layer = nn.Linear(embed_dim, 1, bias=True)

        # Initialize so that 2 * sigmoid(0) == 1.0
        nn.init.zeros_(self.mask_layer.weight)
        nn.init.zeros_(self.mask_layer.bias)

        self.last_mask = None  # to store for inspection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.mask_layer(x)         # shape (B, L, 1)
        mask   = 2 * torch.sigmoid(logits)  # shape (B, L, 1)

        self.last_mask = mask.detach()      # store for external access
#        print(mask)
        gated = x * mask
        return gated

    def get_mask(self):
        """Return last forward-pass mask"""
        return self.last_mask

