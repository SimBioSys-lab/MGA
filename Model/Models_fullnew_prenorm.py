import math
import torch
import torch.nn.functional as F
from torch import einsum, nn
from torch_geometric.nn import GATConv
from einops import rearrange

from Dynamic_Mask import DyM
try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath


# ---------------------------
# Helpers
# ---------------------------

def exists(val):
    return val is not None


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)


def _find_chain_splits_1d(tokens: torch.Tensor, eoc_id: int = 24, pad_id: int = 1):
    """
    Locate chain spans in a 1-D token row.

    This version:
      - stops at padding tokens
      - excludes EOC tokens from the chain spans
      - avoids empty spans

    Returns list of (start, end) inclusive-exclusive spans.
    Example: [(0, 120), (121, 250)] if token 120 is EOC.
    """
    valid_len = int((tokens != pad_id).sum().item())
    toks = tokens[:valid_len]

    eocs = (toks == eoc_id).nonzero(as_tuple=True)[0].tolist()
    spans = []
    start = 0

    for eoc in eocs:
        if eoc > start:
            spans.append((start, eoc))  # exclude EOC
        start = eoc + 1                 # skip EOC

    if start < valid_len:
        spans.append((start, valid_len))

    return spans


# ---------------------------
# Attention modules
# ---------------------------

class Attention(nn.Module):
    """
    Pure multi-head attention operator.

    Important:
      - no residual connection here
      - no LayerNorm here
      - parent blocks own pre-norm + residual
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, gating=True):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_gating = gating

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.gating = nn.Linear(dim, inner_dim) if gating else None
        if gating:
            # sigmoid(5) ~= 0.993, so the gate is near identity at init.
            nn.init.constant_(self.gating.weight, 0.0)
            nn.init.constant_(self.gating.bias, 5.0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, tied=False, return_attention=False):
        """
        x:    (B, N, D)
        mask: optional bool tensor (B, N), True for valid tokens.
        """
        h = self.heads

        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
            (q, k, v),
        )

        q = q * self.scale
        dots = einsum("b h i d, b h j d -> b h i j", q, k)

        if tied:
            # Original code used sqrt(dots.size(0)); that scales by folded batch size.
            # Kept for behavioral compatibility, but you may want to revisit this.
            rowwise_average = torch.mean(dots, dim=3, keepdim=True)
            scaling_factor = math.sqrt(dots.size(0))
            dots = (rowwise_average / scaling_factor).expand_as(dots)

        if mask is not None:
            # mask keys; shape becomes (B, 1, 1, N)
            dots = dots.masked_fill(~mask[:, None, None, :], -torch.finfo(dots.dtype).max)

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        if self.use_gating:
            gates = self.gating(x)
            out = out * gates.sigmoid()

        out = self.to_out(out)

        if return_attention:
            return out, attn
        return out


class AxialAttention(nn.Module):
    """
    Pure axial attention operator.

    Important:
      - no residual connection here
      - no LayerNorm here
      - parent blocks own pre-norm + residual
    """
    def __init__(self, dim, heads, dropout=0.0, row_attn=True, col_attn=True, **kwargs):
        super().__init__()
        assert row_attn or col_attn, "Either row or column attention must be turned on."
        assert row_attn ^ col_attn, "Has to be either row or column attention, not both."

        self.row_attn = row_attn
        self.col_attn = col_attn
        self.attn = Attention(dim=dim, heads=heads, dropout=dropout, **kwargs)

    def forward(self, x, mask=None, return_attention=False, tied=False):
        """
        x:    (B, H, W, D)
        mask: optional bool tensor (B, H, W), True for valid tokens.
        """
        b, h, w, d = x.shape

        if self.col_attn:
            input_fold_eq = "b h w d -> (b w) h d"
            output_fold_eq = "(b w) h d -> b h w d"
            x_fold = rearrange(x, input_fold_eq)
            mask_fold = rearrange(mask, "b h w -> (b w) h") if mask is not None else None
            tied = False

        else:
            input_fold_eq = "b h w d -> (b h) w d"
            output_fold_eq = "(b h) w d -> b h w d"
            x_fold = rearrange(x, input_fold_eq)
            mask_fold = rearrange(mask, "b h w -> (b h) w") if mask is not None else None

        if return_attention:
            out, attn = self.attn(
                x_fold,
                mask=mask_fold,
                tied=tied,
                return_attention=True,
            )
        else:
            out = self.attn(
                x_fold,
                mask=mask_fold,
                tied=tied,
                return_attention=False,
            )
            attn = None

        out = rearrange(out, output_fold_eq, h=h, w=w)
        return (out, attn) if return_attention else out


class MSASelfAttentionBlock(nn.Module):
    """
    Pre-norm MSA row+column attention block.

    This block returns only the update: y - x.
    Parent CoreModel applies DyM/DropPath and residual.
    """
    def __init__(self, dim, heads, dim_head=64, dropout=0.0):
        super().__init__()
        self.row_norm = nn.LayerNorm(dim)
        self.col_norm = nn.LayerNorm(dim)

        self.row_attn = AxialAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            row_attn=True,
            col_attn=False,
        )
        self.col_attn = AxialAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            row_attn=False,
            col_attn=True,
        )

    def forward(self, x, mask=None, tied=False):
        y = x + self.row_attn(self.row_norm(x), mask=mask, tied=tied)
        y = y + self.col_attn(self.col_norm(y), mask=mask, tied=False)
        return y - x


# ---------------------------
# Core MSA model
# ---------------------------

class CoreModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
        drop_path_rate: float = 0.0,
        pad_id: int = 1,
        eoc_id: int = 24,
    ):
        super().__init__()
        print(f"Initializing CoreModel with {num_layers} layers, drop_path_rate={drop_path_rate}")

        self.pad_id = pad_id
        self.eoc_id = eoc_id

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)

        self.self_attention_layers = nn.ModuleList([
            MSASelfAttentionBlock(
                dim=embed_dim,
                heads=num_heads,
                dim_head=64,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.ffn_norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim),
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.attn_dym = nn.ModuleList([DyM(embed_dim) for _ in range(num_layers)])
        self.ffn_dym = nn.ModuleList([DyM(embed_dim) for _ in range(num_layers)])

        dpr = torch.linspace(0, drop_path_rate, num_layers).tolist()
        self.attn_drop_paths = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity()
            for i in range(num_layers)
        ])
        self.ffn_drop_paths = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity()
            for i in range(num_layers)
        ])

        # Final norm is useful for pre-norm residual stacks.
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        sequences: (B, S, L) integer IDs, commonly S == 64.
        returns:   (B, S, L, D)
        """
        B, S, L = sequences.shape
        D = self.embedding.embedding_dim

        # Robustness: nn.Embedding cannot accept -1 or token IDs >= vocab size.
        sequences = sequences.long()
        bad_tokens = (sequences < 0) | (sequences >= self.embedding.num_embeddings)
        if bad_tokens.any():
            sequences = sequences.masked_fill(bad_tokens, self.pad_id)

        x = self.embedding(sequences.reshape(B * S, L))
        x = x.reshape(B, S, L, D)

        valid_mask = sequences != self.pad_id  # (B, S, L)

        # Chain boundaries from the query/first MSA row.
        batch_splits = [
            _find_chain_splits_1d(sequences[b, 0], eoc_id=self.eoc_id, pad_id=self.pad_id)
            for b in range(B)
        ]

        for i in range(len(self.self_attention_layers)):
            # MSA axial attention per chain.
            # IMPORTANT: zeros_like, not clone(), because this is an update tensor.
            x_attn = torch.zeros_like(x)

            for b in range(B):
                for s, e in batch_splits[b]:
                    if e <= s:
                        continue

                    seg = x[b:b + 1, :, s:e, :]
                    seg_mask = valid_mask[b:b + 1, :, s:e]

                    seg_update = self.self_attention_layers[i](seg, mask=seg_mask)
                    x_attn[b, :, s:e, :] = seg_update.squeeze(0)

            x_attn = self.dropout(x_attn)
            x_attn = self.attn_dym[i](x_attn)
            x_attn = self.attn_drop_paths[i](x_attn)
            x = x + x_attn

            # FFN pre-norm.
            ffn_in = self.ffn_norm_layers[i](x)
            x_ffn = self.ffn_layers[i](ffn_in.reshape(B * S, L, D))
            x_ffn = x_ffn.reshape(B, S, L, D)

            x_ffn = self.dropout(x_ffn)
            x_ffn = self.ffn_dym[i](x_ffn)
            x_ffn = self.ffn_drop_paths[i](x_ffn)
            x = x + x_ffn

            # Keep padded positions from accumulating arbitrary residual values.
            x = x * valid_mask[..., None].to(dtype=x.dtype)

        x = self.final_norm(x)
        x = x * valid_mask[..., None].to(dtype=x.dtype)
        return x


# ---------------------------
# Core + graph model
# ---------------------------

class CGModel(nn.Module):
    """
    CoreModel + GNN layers + external DyM gating + DropPath.
    Uses pre-norm residual blocks.
    """
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
        num_gnn_layers: int,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        self.core_model = CoreModel(
            vocab_size,
            seq_len,
            embed_dim,
            num_heads,
            dropout,
            num_layers,
            drop_path_rate,
        )
        self.dropout = nn.Dropout(dropout)

        self.pos_emb = nn.Parameter(torch.zeros(seq_len, embed_dim))
        nn.init.xavier_uniform_(self.pos_emb)

        self.fc_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.fc_dym = DyM(embed_dim)
        self.fc_dp = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.gnn_layers = nn.ModuleList([
            GATConv(embed_dim, embed_dim // num_heads, heads=num_heads, concat=True)
            for _ in range(num_gnn_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim),
            )
            for _ in range(num_gnn_layers)
        ])

        self.gnn_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_gnn_layers)])
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_gnn_layers)])

        self.gnn_dym = nn.ModuleList([DyM(embed_dim) for _ in range(num_gnn_layers)])
        self.ffn_dym = nn.ModuleList([DyM(embed_dim) for _ in range(num_gnn_layers)])

        self.gnn_dp = nn.ModuleList([
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
            for _ in range(num_gnn_layers)
        ])
        self.ffn_dp = nn.ModuleList([
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
            for _ in range(num_gnn_layers)
        ])

        # Final norm after the pre-norm GNN/FFN stack.
        self.final_norm = nn.LayerNorm(embed_dim)

    def _build_edge_index(self, padded_edges, seq_len, device):
        """
        Build batched edge_index robustly.

        padded_edges: [B, 2, Emax], with -1 padding.
        Keeps only edges where both endpoints are in [0, seq_len).
        """
        B = padded_edges.shape[0]
        edge_indices = []
        padded_edges = padded_edges.long()

        for i in range(B):
            src = padded_edges[i, 0]
            dst = padded_edges[i, 1]

            valid_mask = (
                (src >= 0) & (dst >= 0) &
                (src < seq_len) & (dst < seq_len)
            )

            if not valid_mask.any():
                continue

            valid = padded_edges[i][:, valid_mask].long()
            valid = valid + i * seq_len
            edge_indices.append(valid)

        if len(edge_indices) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        edge_index = torch.cat(edge_indices, dim=1).long()

        # Defensive check before PyG CUDA kernels.
        num_nodes = B * seq_len
        if edge_index.numel() > 0:
            if int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= num_nodes:
                raise RuntimeError(
                    f"edge_index out of range: min={int(edge_index.min().item())}, "
                    f"max={int(edge_index.max().item())}, num_nodes={num_nodes}, "
                    f"B={B}, seq_len={seq_len}"
                )

        return edge_index

    def forward(self, sequences, padded_edges):
        B = sequences.shape[0]
        seq_len = sequences.shape[2]

        edge_index = self._build_edge_index(
            padded_edges=padded_edges,
            seq_len=seq_len,
            device=sequences.device,
        )

        # Core transformer: (B, S, L, D). Use first MSA row/query.
        x = self.core_model(sequences)
        x = self.dropout(x)
        x = x[:, 0, :, :]  # (B, L, D)

        pos = self.pos_emb[:seq_len].unsqueeze(0).expand(B, -1, -1)
        x = x + self.dropout(pos)

        # Initial FC pre-norm residual.
        fc_out = self.fc(self.fc_norm(x))
        fc_out = self.dropout(fc_out)
        fc_out = self.fc_dym(fc_out)
        fc_out = self.fc_dp(fc_out)
        x = x + fc_out

        x_flat = x.reshape(-1, x.size(-1))

        for i in range(len(self.gnn_layers)):
            # GNN pre-norm residual.
            g_in = self.gnn_norm[i](x_flat)
            g_out = self.gnn_layers[i](g_in, edge_index)
            g_out = self.dropout(g_out)
            g_out = self.gnn_dym[i](g_out)
            g_out = self.gnn_dp[i](g_out)
            x_flat = x_flat + g_out

            # FFN pre-norm residual.
            f_in = self.ffn_norm[i](x_flat)
            f_out = self.ffn_layers[i](f_in)
            f_out = self.dropout(f_out)
            f_out = self.ffn_dym[i](f_out)
            f_out = self.ffn_dp[i](f_out)
            x_flat = x_flat + f_out

        x_flat = self.final_norm(x_flat)
        refined = x_flat.reshape(B, 1, seq_len, -1)
        return refined


# ---------------------------
# MC model
# ---------------------------

class MCModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
        num_gnn_layers: int,
        num_int_layers: int,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        print(
            f"Initializing MCModel with {num_gnn_layers} GAT layers and "
            f"{num_int_layers} row attention layers (drop_path_rate={drop_path_rate})"
        )

        self.cg_model = CGModel(
            vocab_size,
            seq_len,
            embed_dim,
            num_heads,
            dropout,
            num_layers,
            num_gnn_layers,
            drop_path_rate,
        )

        self.row_attn_layers = nn.ModuleList([
            AxialAttention(
                dim=embed_dim,
                heads=num_heads,
                dropout=dropout,
                row_attn=True,
                col_attn=False,
            )
            for _ in range(num_int_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim),
            )
            for _ in range(num_int_layers)
        ])

        self.row_norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_int_layers)])
        self.ffn_norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_int_layers)])
        self.dropout = nn.Dropout(dropout)

        self.row_dym = nn.ModuleList([DyM(embed_dim) for _ in range(num_int_layers)])
        self.ffn_dym = nn.ModuleList([DyM(embed_dim) for _ in range(num_int_layers)])

        dpr = torch.linspace(0, drop_path_rate, num_int_layers).tolist()
        self.row_dp = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity()
            for i in range(num_int_layers)
        ])
        self.ffn_dp = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity()
            for i in range(num_int_layers)
        ])

        # Final norm after the pre-norm interaction stack.
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        sequences,
        padded_edges,
        pairwise_repr=None,
        return_attention=False,
        tied=False,
    ):
        output = self.cg_model(sequences, padded_edges)
        last_attn = None

        # output shape: (B, 1, L, D)
        # Row attention treats the L dimension as the sequence axis when H == 1.
        for idx, (row_attn, ffn, rnorm, fnorm) in enumerate(zip(
            self.row_attn_layers,
            self.ffn_layers,
            self.row_norm_layers,
            self.ffn_norm_layers,
        )):
            row_in = rnorm(output)

            if return_attention and idx == len(self.row_attn_layers) - 1:
                row_out, attn = row_attn(row_in, return_attention=True, tied=tied)
                last_attn = attn
            else:
                row_out = row_attn(row_in, tied=tied)

            row_out = self.dropout(row_out)
            row_out = self.row_dym[idx](row_out)
            row_out = self.row_dp[idx](row_out)
            output = output + row_out

            ffn_out = ffn(fnorm(output))
            ffn_out = self.dropout(ffn_out)
            ffn_out = self.ffn_dym[idx](ffn_out)
            ffn_out = self.ffn_dp[idx](ffn_out)
            output = output + ffn_out

        output = self.final_norm(output)

        if return_attention:
            return output, last_attn
        return output


# ---------------------------
# Task heads
# ---------------------------

class ClassificationModel(nn.Module):
    """
    Classification model using MCModel and a classification layer.
    """
    def __init__(
        self,
        vocab_size,
        seq_len,
        embed_dim,
        num_heads,
        dropout,
        num_layers,
        num_gnn_layers,
        num_classes,
        num_int_layers,
        drop_path_rate,
    ):
        super().__init__()
        print(f"Initializing ClassificationModel with {num_gnn_layers} GAT layers and residual connections...")

        self.mc_model = MCModel(
            vocab_size,
            seq_len,
            embed_dim,
            num_heads,
            dropout,
            num_layers,
            num_gnn_layers,
            num_int_layers,
            drop_path_rate,
        )
        # Extra head norm is often helpful with pre-norm residual stacks.
        self.head_norm = nn.LayerNorm(embed_dim)
        self.fc_classification = nn.Linear(embed_dim, num_classes)

    def forward(self, sequences, padded_edges, return_attention=False, tied=False):
        if return_attention:
            refined_embedding, last_attn = self.mc_model(
                sequences,
                padded_edges,
                return_attention=True,
                tied=tied,
            )
        else:
            refined_embedding = self.mc_model(sequences, padded_edges)

        refined_embedding = self.head_norm(refined_embedding)
        predictions = self.fc_classification(refined_embedding)

        if return_attention:
            return predictions, last_attn
        return predictions


class RegressionModel(nn.Module):
    """
    Regression model using MCModel and a regression layer.

    By default this uses softplus for non-negative regression.
    If your target can be negative, set positive_output=False.
    """
    def __init__(
        self,
        vocab_size,
        seq_len,
        embed_dim,
        num_heads,
        dropout,
        num_layers,
        num_gnn_layers,
        num_int_layers,
        drop_path_rate,
        positive_output=True,
    ):
        super().__init__()
        print(f"Initializing RegressionModel with {num_gnn_layers} GAT layers and residual connections...")

        self.mc_model = MCModel(
            vocab_size,
            seq_len,
            embed_dim,
            num_heads,
            dropout,
            num_layers,
            num_gnn_layers,
            num_int_layers,
            drop_path_rate,
        )
        self.head_norm = nn.LayerNorm(embed_dim)
        self.fc_regression = nn.Linear(embed_dim, 1)
        self.positive_output = positive_output

    def forward(self, sequences, padded_edges):
        refined_embedding = self.mc_model(sequences, padded_edges)
        refined_embedding = self.head_norm(refined_embedding)
        predictions = self.fc_regression(refined_embedding)

        if self.positive_output:
            predictions = F.softplus(predictions)

        return predictions


class CTMModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_len,
        embed_dim,
        num_heads,
        dropout,
        num_layers,
        num_gnn_layers,
        num_classes,
        num_int_layers,
        drop_path_rate,
    ):
        super().__init__()
        print(f"Initializing CTMModel with {num_gnn_layers} GAT layers and residual connections...")

        self.mc_model = MCModel(
            vocab_size,
            seq_len,
            embed_dim,
            num_heads,
            dropout,
            num_layers,
            num_gnn_layers,
            num_int_layers,
            drop_path_rate,
        )

        self.fc_linear = nn.Linear(num_heads, num_classes)

    def forward(self, sequences, padded_edges, return_attention=True, tied=False):
        # CTM needs attention to form contact-map logits, so force attention internally.
        refined_embedding, last_attn = self.mc_model(
            sequences,
            padded_edges,
            return_attention=True,
            tied=tied,
        )

        if last_attn is None:
            raise RuntimeError(
                "CTMModel requires num_int_layers > 0 so that MCModel can return row attention."
            )

        x = last_attn.permute(0, 2, 3, 1)
        x = self.fc_linear(x)
        logits = x.permute(0, 3, 1, 2)

        if return_attention:
            return logits, last_attn
        return logits

