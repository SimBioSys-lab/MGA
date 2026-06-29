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


def exists(val):
    return val is not None


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)


def _find_chain_spans_1d(tokens: torch.Tensor, eoc_id: int = 24, pad_id: int = 1):
    """
    Fixed chain splitting.

    This version:
      1. reads the query row only
      2. stops at the first PAD token
      3. splits at every EOC
      4. excludes EOC tokens from biological residue spans
      5. supports multiple antigen chains

    Example:
      tokens: L ... EOC H ... EOC AG1 ... EOC AG2 ... PAD
      spans:  L, H, AG1, AG2 without EOC tokens
    """
    tokens = tokens.long()

    pad_pos = (tokens == pad_id).nonzero(as_tuple=True)[0]
    valid_len = int(pad_pos[0].item()) if pad_pos.numel() > 0 else int(tokens.numel())

    if valid_len <= 0:
        return []

    toks = tokens[:valid_len]
    eocs = (toks == eoc_id).nonzero(as_tuple=True)[0].tolist()

    spans = []
    start = 0
    for eoc in eocs:
        eoc = int(eoc)
        if eoc > start:
            spans.append((start, eoc))  # exclude EOC itself
        start = eoc + 1                 # skip EOC

    if start < valid_len:
        spans.append((start, valid_len))

    return spans


def _boundary_context_start(query_tokens: torch.Tensor, start: int, eoc_id: int = 24):
    """
    Return context start for a chain span.

    If the residue span starts immediately after an EOC token, include that EOC
    as a left-context token for attention. For the first span, there is no
    previous EOC, so CoreModel prepends a learned BOS/context token instead.
    The caller should only write outputs back to residue positions [start:end],
    not to the context position.

    This recovers the useful boundary-token context from the old chain-split
    behavior while avoiding treating EOC as a biological residue.
    """
    if start > 0 and int(query_tokens[start - 1].item()) == int(eoc_id):
        return start - 1
    return start


class Attention(nn.Module):
    """
    Old-style attention block:
      Attention itself owns residual + LayerNorm.

    This intentionally keeps the old behavior:
        out = x + to_out(attention(x))
        out = norm(out)
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
            nn.init.constant_(self.gating.weight, 0.0)
            nn.init.constant_(self.gating.bias, 5.0)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None, tied=False, return_attention=False):
        h = self.heads
        residual = x

        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale
        dots = einsum("b h i d, b h j d -> b h i j", q, k)

        if tied:
            rowwise_average = torch.mean(dots, dim=3, keepdim=True)
            scaling_factor = math.sqrt(dots.size(0))
            dots = (rowwise_average / scaling_factor).expand_as(dots)

        if mask is not None:
            dots = dots.masked_fill(~mask[:, None, None, :], -torch.finfo(dots.dtype).max)

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        if self.use_gating:
            gates = self.gating(x)
            out = out * gates.sigmoid()

        out = self.to_out(out)

        # Old nested residual/norm behavior.
        out = residual + out
        out = self.norm(out)

        if return_attention:
            return out, attn

        return out


class AxialAttention(nn.Module):
    """
    Old-style axial attention:
      AxialAttention applies LayerNorm before folding/calling Attention.
      Attention then also applies its own residual + norm.
    """
    def __init__(self, dim, heads, dropout=0.0, row_attn=True, col_attn=True, **kwargs):
        super().__init__()

        assert row_attn or col_attn, "Either row or column attention must be turned on."
        assert row_attn ^ col_attn, "Has to be either row or column attention, not both."

        self.row_attn = row_attn
        self.col_attn = col_attn
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads, dropout=dropout, **kwargs)

    def forward(self, x, mask=None, return_attention=False, tied=False):
        b, h, w, d = x.shape

        x = self.norm(x)

        if self.col_attn:
            x_fold = rearrange(x, "b h w d -> (b w) h d")
            mask_fold = rearrange(mask, "b h w -> (b w) h") if mask is not None else None
            output_fold_eq = "(b w) h d -> b h w d"
            tied = False
        else:
            x_fold = rearrange(x, "b h w d -> (b h) w d")
            mask_fold = rearrange(mask, "b h w -> (b h) w") if mask is not None else None
            output_fold_eq = "(b h) w d -> b h w d"

        if return_attention:
            out, attn = self.attn(x_fold, mask=mask_fold, tied=tied, return_attention=True)
        else:
            out = self.attn(x_fold, mask=mask_fold, tied=tied, return_attention=False)
            attn = None

        out = rearrange(out, output_fold_eq, h=h, w=w)

        if return_attention:
            return out, attn

        return out


class MSASelfAttentionBlock(nn.Module):
    """
    Old-style MSA block:
      row axial attention followed by column axial attention.
    """
    def __init__(self, dim, heads, dim_head=64, dropout=0.0):
        super().__init__()

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
        x = self.row_attn(x, mask=mask, tied=tied)
        x = self.col_attn(x, mask=mask, tied=False)
        return x


class CoreModel(nn.Module):
    """
    Old-style CoreModel with fixed chain splitting plus boundary-context attention.

    Biological residue spans exclude EOC tokens, but if a chain starts immediately
    after an EOC, that EOC is included as a left-context token during attention.
    The EOC output is discarded and only residue positions are written back.

    This recovers the useful boundary-token context from the old buggy split while
    avoiding treating EOC as a predicted biological residue.
    """
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
        use_boundary_context: bool = True,
    ):
        super().__init__()

        print(f"Initializing CoreModel with {num_layers} layers, drop_path_rate={drop_path_rate}")

        self.pad_id = pad_id
        self.eoc_id = eoc_id
        self.use_boundary_context = use_boundary_context
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)

        # Learned boundary/BOS context token used for the first chain span.
        # For non-first chains, the real preceding EOC token is used as context.
        # Initialized to zero so the first-span context starts neutral.
        self.bos_context = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.self_attention_layers = nn.ModuleList([
            MSASelfAttentionBlock(
                dim=embed_dim,
                heads=num_heads,
                dim_head=64,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm_layers = nn.ModuleList([
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

        self.ffn_norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.attn_dym = nn.ModuleList([DyM(embed_dim) for _ in range(num_layers)])
        self.ffn_dym = nn.ModuleList([DyM(embed_dim) for _ in range(num_layers)])

        dpr = torch.linspace(0, drop_path_rate, num_layers).tolist()
        self.attn_drop_paths = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
            for i in range(num_layers)
        ])
        self.ffn_drop_paths = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
            for i in range(num_layers)
        ])

    def forward(self, sequences):
        """
        sequences: [B, S, L]
        returns:   [B, S, L, D]
        """
        if sequences.dim() != 3:
            raise ValueError(f"CoreModel expected sequences [B, S, L], got {tuple(sequences.shape)}")

        B, S, L = sequences.shape
        D = self.embedding.embedding_dim

        sequences = sequences.long()

        # Safety fix: avoid embedding CUDA assert on invalid token ids.
        bad_tokens = (sequences < 0) | (sequences >= self.embedding.num_embeddings)
        if bad_tokens.any():
            sequences = sequences.masked_fill(bad_tokens, self.pad_id)

        valid_mask = sequences != self.pad_id

        x = self.embedding(sequences.reshape(B * S, L))
        x = x.reshape(B, S, L, D)

        # Biological residue spans from query row, excluding EOC tokens.
        batch_spans = [
            _find_chain_spans_1d(sequences[b, 0], eoc_id=self.eoc_id, pad_id=self.pad_id)
            for b in range(B)
        ]

        for i in range(len(self.self_attention_layers)):
            res = x
            x_attn = torch.zeros_like(x)

            for b in range(B):
                query_tokens = sequences[b, 0]
                for start, end in batch_spans[b]:
                    if end <= start:
                        continue

                    ctx_start = start
                    prepend_bos = False

                    if self.use_boundary_context:
                        ctx_start = _boundary_context_start(
                            query_tokens=query_tokens,
                            start=start,
                            eoc_id=self.eoc_id,
                        )

                        # First biological chain has no preceding EOC.
                        # Give it the same kind of left-context slot using a learned BOS token.
                        if ctx_start == start:
                            prepend_bos = True

                    # Attend over optional boundary/BOS context token + residues.
                    seg = x[b:b + 1, :, ctx_start:end, :]
                    seg_mask = valid_mask[b:b + 1, :, ctx_start:end]

                    if prepend_bos:
                        bos = self.bos_context.expand(1, S, 1, D).to(dtype=seg.dtype, device=seg.device)
                        seg = torch.cat([bos, seg], dim=2)

                        bos_mask = torch.ones((1, S, 1), dtype=torch.bool, device=seg_mask.device)
                        seg_mask = torch.cat([bos_mask, seg_mask], dim=2)

                    seg_out = self.self_attention_layers[i](seg, mask=seg_mask)

                    # If a context token was included, discard its output and
                    # write only biological residue positions [start:end].
                    if prepend_bos:
                        seg_out = seg_out[:, :, 1:, :]
                    elif ctx_start < start:
                        seg_out = seg_out[:, :, (start - ctx_start):, :]

                    if seg_out.shape[2] != (end - start):
                        raise RuntimeError(
                            f"Boundary-context segment length mismatch: "
                            f"ctx_start={ctx_start}, start={start}, end={end}, "
                            f"prepend_bos={prepend_bos}, seg_out_len={seg_out.shape[2]}"
                        )

                    x_attn[b, :, start:end, :] = seg_out.squeeze(0)

            x_attn = self.dropout(x_attn)
            x_attn = self.attn_dym[i](x_attn)
            x_attn = self.attn_drop_paths[i](x_attn)

            # Old outer residual + norm.
            x = self.norm_layers[i](res + x_attn)

            res = x
            x_ffn = self.ffn_layers[i](x.reshape(B * S, L, D))
            x_ffn = x_ffn.reshape(B, S, L, D)
            x_ffn = self.dropout(x_ffn)
            x_ffn = self.ffn_dym[i](x_ffn)
            x_ffn = self.ffn_drop_paths[i](x_ffn)

            # Old FFN residual + norm.
            x = self.ffn_norm_layers[i](res + x_ffn)

            # Avoid padded positions accumulating arbitrary activations.
            x = x * valid_mask[..., None].to(dtype=x.dtype)

        return x


class CGModel(nn.Module):
    """
    Old-style Core + GAT model with residual connections.
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
        use_boundary_context: bool = True,
    ):
        super().__init__()

        self.core_model = CoreModel(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers,
            drop_path_rate=drop_path_rate,
            use_boundary_context=use_boundary_context,
        )

        self.dropout = nn.Dropout(dropout)

        self.pos_emb = nn.Parameter(torch.zeros(seq_len, embed_dim))
        nn.init.xavier_uniform_(self.pos_emb)

        self.fc = nn.Linear(embed_dim, embed_dim)
        self.fc_norm = nn.LayerNorm(embed_dim)
        self.fc_dym = DyM(embed_dim)
        self.fc_dp = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.gnn_layers = nn.ModuleList([
            GATConv(embed_dim, embed_dim // num_heads, heads=num_heads, concat=True)
            for _ in range(num_gnn_layers)
        ])

        self.gnn_norm = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_gnn_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim),
            )
            for _ in range(num_gnn_layers)
        ])

        self.ffn_norm = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_gnn_layers)
        ])

        self.gnn_dym = nn.ModuleList([DyM(embed_dim) for _ in range(num_gnn_layers)])
        self.ffn_dym = nn.ModuleList([DyM(embed_dim) for _ in range(num_gnn_layers)])

        self.gnn_dp = nn.ModuleList([
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
            for _ in range(num_gnn_layers)
        ])
        self.ffn_dp = nn.ModuleList([
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
            for _ in range(num_gnn_layers)
        ])

    def _build_edge_index(self, padded_edges, seq_len, device):
        B = padded_edges.shape[0]
        edge_indices = []

        padded_edges = padded_edges.long()

        for i in range(B):
            src = padded_edges[i, 0]
            dst = padded_edges[i, 1]

            valid = (
                (src >= 0) & (dst >= 0) &
                (src < seq_len) & (dst < seq_len)
            )

            if not valid.any():
                continue

            ei = padded_edges[i][:, valid].long()
            ei = ei + i * seq_len
            edge_indices.append(ei)

        if len(edge_indices) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        edge_index = torch.cat(edge_indices, dim=1).to(device=device, dtype=torch.long)

        num_nodes = B * seq_len
        if edge_index.numel() > 0:
            if int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= num_nodes:
                raise RuntimeError(
                    f"edge_index out of range: min={int(edge_index.min().item())}, "
                    f"max={int(edge_index.max().item())}, num_nodes={num_nodes}"
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

        x = self.core_model(sequences)
        x = self.dropout(x)

        # Query row only.
        x = x[:, 0, :, :]  # [B, L, D]

        pos = self.pos_emb[:seq_len].unsqueeze(0).expand(B, -1, -1)
        x = x + self.dropout(pos)

        # Old residual FC block.
        res = x
        x_fc = self.fc(x)
        x_fc = self.dropout(x_fc)
        x_fc = self.fc_dym(x_fc)
        x_fc = self.fc_dp(x_fc)
        x = self.fc_norm(res + x_fc)

        x_flat = x.reshape(B * seq_len, -1)

        for i in range(len(self.gnn_layers)):
            res = x_flat
            g = self.gnn_layers[i](x_flat, edge_index)
            g = self.dropout(g)
            g = self.gnn_dym[i](g)
            g = self.gnn_dp[i](g)
            x_flat = self.gnn_norm[i](res + g)

            res = x_flat
            f = self.ffn_layers[i](x_flat)
            f = self.dropout(f)
            f = self.ffn_dym[i](f)
            f = self.ffn_dp[i](f)
            x_flat = self.ffn_norm[i](res + f)

        refined = x_flat.reshape(B, 1, seq_len, -1)
        return refined


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
        use_boundary_context: bool = True,
    ):
        super().__init__()

        print(
            f"Initializing MCModel with {num_gnn_layers} GAT layers and "
            f"{num_int_layers} row attention layers (drop_path_rate={drop_path_rate})"
        )

        self.cg_model = CGModel(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers,
            num_gnn_layers=num_gnn_layers,
            drop_path_rate=drop_path_rate,
            use_boundary_context=use_boundary_context,
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

        self.row_norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_int_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim),
            )
            for _ in range(num_int_layers)
        ])

        self.ffn_norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_int_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.row_dym = nn.ModuleList([DyM(embed_dim) for _ in range(num_int_layers)])
        self.ffn_dym = nn.ModuleList([DyM(embed_dim) for _ in range(num_int_layers)])

        dpr = torch.linspace(0, drop_path_rate, num_int_layers).tolist()
        self.row_dp = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
            for i in range(num_int_layers)
        ])
        self.ffn_dp = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
            for i in range(num_int_layers)
        ])

    def forward(self, sequences, padded_edges, pairwise_repr=None, return_attention=False, tied=False):
        output = self.cg_model(sequences, padded_edges)
        last_attn = None

        for idx, (row_attn, row_norm, ffn, ffn_norm) in enumerate(zip(
            self.row_attn_layers,
            self.row_norm_layers,
            self.ffn_layers,
            self.ffn_norm_layers,
        )):
            res = output
            if return_attention and idx == len(self.row_attn_layers) - 1:
                row_out, attn = row_attn(output, return_attention=True, tied=tied)
                last_attn = attn
            else:
                row_out = row_attn(output, tied=tied)

            row_out = self.dropout(row_out)
            row_out = self.row_dym[idx](row_out)
            row_out = self.row_dp[idx](row_out)
            output = row_norm(res + row_out)

            res = output
            ffn_out = ffn(output)
            ffn_out = self.dropout(ffn_out)
            ffn_out = self.ffn_dym[idx](ffn_out)
            ffn_out = self.ffn_dp[idx](ffn_out)
            output = ffn_norm(res + ffn_out)

        if return_attention:
            return output, last_attn

        return output


class ClassificationModel(nn.Module):
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
        use_boundary_context: bool = True,
    ):
        super().__init__()

        print(f"Initializing ClassificationModel with {num_gnn_layers} GAT layers and residual connections...")

        self.mc_model = MCModel(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers,
            num_gnn_layers=num_gnn_layers,
            num_int_layers=num_int_layers,
            drop_path_rate=drop_path_rate,
            use_boundary_context=use_boundary_context,
        )

        self.fc_classification = nn.Linear(embed_dim, num_classes)

    def forward(self, sequences, padded_edges, return_attention=False, tied=False):
        if return_attention:
            refined_embedding, last_attn = self.mc_model(
                sequences,
                padded_edges,
                return_attention=True,
                tied=tied,
            )
            predictions = self.fc_classification(refined_embedding)
            return predictions, last_attn

        refined_embedding = self.mc_model(sequences, padded_edges)
        predictions = self.fc_classification(refined_embedding)
        return predictions


class RegressionModel(nn.Module):
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
        use_boundary_context: bool = True,
    ):
        super().__init__()

        print(f"Initializing RegressionModel with {num_gnn_layers} GAT layers and residual connections...")

        self.mc_model = MCModel(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers,
            num_gnn_layers=num_gnn_layers,
            num_int_layers=num_int_layers,
            drop_path_rate=drop_path_rate,
            use_boundary_context=use_boundary_context,
        )

        self.fc_regression = nn.Linear(embed_dim, 1)
        self.positive_output = positive_output

    def forward(self, sequences, padded_edges):
        refined_embedding = self.mc_model(sequences, padded_edges)
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
        use_boundary_context: bool = True,
    ):
        super().__init__()

        print(f"Initializing CTMModel with {num_gnn_layers} GAT layers and residual connections...")

        self.mc_model = MCModel(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers,
            num_gnn_layers=num_gnn_layers,
            num_int_layers=num_int_layers,
            drop_path_rate=drop_path_rate,
            use_boundary_context=use_boundary_context,
        )

        self.fc_linear = nn.Linear(num_heads, num_classes)

    def forward(self, sequences, padded_edges, return_attention=True, tied=False):
        refined_embedding, last_attn = self.mc_model(
            sequences,
            padded_edges,
            return_attention=True,
            tied=tied,
        )

        if last_attn is None:
            raise RuntimeError("CTMModel requires num_int_layers > 0 to return row attention.")

        x = last_attn.permute(0, 2, 3, 1)
        x = self.fc_linear(x)
        logits = x.permute(0, 3, 1, 2)

        if return_attention:
            return logits, last_attn

        return logits

