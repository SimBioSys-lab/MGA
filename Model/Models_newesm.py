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


# IMPORTANT:
# This default map is only used by the ESM2 fusion branch to convert your internal
# token ids back to amino-acid letters. If your Dataloader uses a different id
# mapping, edit DEFAULT_ID_TO_AA or pass esm_id_to_aa=... to the model constructor.
# pad_id=1 and eoc_id=24 are skipped by the chain spans before conversion.
DEFAULT_ID_TO_AA = {
    0: "X",
    1: "",   # PAD
    2: "A",
    3: "C",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H",
    9: "I",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "P",
    15: "Q",
    16: "R",
    17: "S",
    18: "T",
    19: "V",
    20: "W",
    21: "Y",
    22: "B",
    23: "Z",
    24: "",  # EOC / chain separator
    25: "X",
    26: "U",
    27: "O",
    28: "X",
    29: "X",
    30: "X",
}

_VALID_ESM_AA = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


def _find_chain_spans_1d(tokens: torch.Tensor, eoc_id: int = 24, pad_id: int = 1):
    """
    Fixed chain splitting.

    This version:
      1. reads the query row only
      2. stops at the first PAD token
      3. splits at every EOC
      4. excludes EOC tokens from attended chain spans
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


class FrozenESM2QueryFusion(nn.Module):
    """
    Frozen small ESM2 branch for the query row only.

    It encodes each chain span from the query row with ESM2, projects ESM2
    representations to embed_dim through a small adapter, and returns a
    chain-role-scaled residual update:

        alpha_L  * adapter(ESM2) for light-chain positions
        alpha_H  * adapter(ESM2) for heavy-chain positions
        alpha_AG * adapter(ESM2) for antigen-chain positions

    The three alpha parameters are initialized to 0.0 by default, so the model
    starts exactly as the MSA-only baseline. ESM2 is frozen by default; the
    adapter and alpha parameters are trainable.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        dropout: float = 0.1,
        esm_model_name: str = "esm2_t6_8M_UR50D",
        esm_repr_layer: int = None,
        alpha_init: float = 0.0,
        freeze_esm: bool = True,
        max_esm_len: int = 1022,
        pad_id: int = 1,
        eoc_id: int = 24,
        id_to_aa=None,
    ):
        super().__init__()

        try:
            import esm
        except ImportError as exc:
            raise ImportError(
                "ESM fusion requires fair-esm. Install it with `pip install fair-esm` "
                "or set use_esm=False in the model constructor."
            ) from exc

        if not hasattr(esm.pretrained, esm_model_name):
            raise ValueError(f"Unknown esm.pretrained model: {esm_model_name}")

        self.esm_model, self.esm_alphabet = getattr(esm.pretrained, esm_model_name)()
        self.batch_converter = self.esm_alphabet.get_batch_converter()

        self.esm_repr_layer = esm_repr_layer
        if self.esm_repr_layer is None:
            self.esm_repr_layer = int(getattr(self.esm_model, "num_layers", 6))

        esm_dim = int(getattr(self.esm_model, "embed_dim", 320))

        self.freeze_esm = freeze_esm
        if self.freeze_esm:
            self.esm_model.eval()
            for p in self.esm_model.parameters():
                p.requires_grad = False

        self.max_esm_len = int(max_esm_len)
        self.pad_id = int(pad_id)
        self.eoc_id = int(eoc_id)
        self.id_to_aa = dict(DEFAULT_ID_TO_AA if id_to_aa is None else id_to_aa)

        self.adapter = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Starts as exact MSA baseline: x_query = x_query + 0 * adapter(ESM).
        # Signed scalars are intentional: the model can learn to add or subtract
        # an ESM-derived direction separately for antibody and antigen regions.
        self.alpha_l = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_h = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha_ag = nn.Parameter(torch.tensor(float(alpha_init)))

    def _tokens_to_aa(self, token_slice: torch.Tensor) -> str:
        chars = []
        for tok in token_slice.detach().cpu().long().tolist():
            tok = int(tok)
            if tok == self.pad_id or tok == self.eoc_id:
                continue
            aa = self.id_to_aa.get(tok, "X")
            if aa is None or aa == "":
                continue
            aa = str(aa)[0].upper()
            if aa not in _VALID_ESM_AA:
                aa = "X"
            chars.append(aa)
        return "".join(chars)

    def _encode_piece_batch(self, pieces, device):
        """
        pieces: list of (piece_id, aa_string)
        returns list of tensors [len(seq), esm_dim] on device, same order.
        """
        if len(pieces) == 0:
            return []

        _, _, toks = self.batch_converter(pieces)
        toks = toks.to(device)

        if self.freeze_esm:
            self.esm_model.eval()

        with torch.no_grad() if self.freeze_esm else torch.enable_grad():
            out = self.esm_model(
                toks,
                repr_layers=[self.esm_repr_layer],
                return_contacts=False,
            )
            reps = out["representations"][self.esm_repr_layer]

        seq_reps = []
        for i, (_, seq) in enumerate(pieces):
            n = len(seq)
            seq_reps.append(reps[i, 1:1 + n, :])  # remove BOS/EOS
        return seq_reps

    def forward(self, query_tokens: torch.Tensor, chain_spans, out_len: int):
        """
        query_tokens: [B, L] internal token IDs from sequences[:, 0, :]
        chain_spans:  list[list[(start, end)]] from _find_chain_spans_1d
        out_len:      L

        returns: [B, L, embed_dim], region-alpha-scaled ESM adapter output.
        """
        if query_tokens.dim() != 2:
            raise ValueError(f"query_tokens expected [B, L], got {tuple(query_tokens.shape)}")

        device = query_tokens.device
        B = query_tokens.shape[0]
        esm_dim = int(getattr(self.esm_model, "embed_dim", 320))

        esm_repr = torch.zeros(B, out_len, esm_dim, device=device)

        # Encode per-chain spans. Long chains are split into chunks to avoid ESM2
        # max-position issues.
        pieces = []
        locations = []  # (b, abs_start, abs_end, role_id), role_id: 0=L, 1=H, 2=AG

        for b in range(B):
            for span_idx, (start, end) in enumerate(chain_spans[b]):
                # By construction of your query sequence: span 0 = L chain,
                # span 1 = H chain, span >= 2 = antigen chain(s).
                role_id = 0 if span_idx == 0 else (1 if span_idx == 1 else 2)
                if end <= start:
                    continue
                cur = int(start)
                end = int(end)
                while cur < end:
                    nxt = min(cur + self.max_esm_len, end)
                    aa_seq = self._tokens_to_aa(query_tokens[b, cur:nxt])
                    if len(aa_seq) > 0:
                        pieces.append((f"b{b}_{cur}_{nxt}", aa_seq))
                        # If token conversion skipped any special token, lengths can differ.
                        # For normal chain spans this should match nxt-cur.
                        locations.append((b, cur, cur + len(aa_seq), role_id))
                    cur = nxt

        # Process pieces one by one or in small batches. The batch size is kept
        # conservative because sequence lengths can be large.
        batch_size = 4
        offset = 0
        while offset < len(pieces):
            sub_pieces = pieces[offset:offset + batch_size]
            sub_locs = locations[offset:offset + batch_size]
            sub_reps = self._encode_piece_batch(sub_pieces, device=device)

            for rep, (b, start, end, role_id) in zip(sub_reps, sub_locs):
                n = min(rep.shape[0], end - start, out_len - start)
                if n > 0:
                    esm_repr[b, start:start + n, :] = rep[:n]

            offset += batch_size

        esm_repr = esm_repr.to(dtype=self.adapter[1].weight.dtype)
        esm_update = self.adapter(esm_repr)

        # Build [B, L, 1] alpha map from chain spans. All alphas start at zero,
        # so this branch is initially exactly disabled.
        alpha_map = torch.zeros(B, out_len, 1, device=device, dtype=esm_update.dtype)
        for b in range(B):
            for span_idx, (start, end) in enumerate(chain_spans[b]):
                if end <= start:
                    continue
                role_alpha = self.alpha_l if span_idx == 0 else (self.alpha_h if span_idx == 1 else self.alpha_ag)
                alpha_map[b, int(start):min(int(end), out_len), 0] = role_alpha.to(dtype=esm_update.dtype)

        return alpha_map * esm_update

    def get_alpha_values(self):
        return {
            "Lchain": float(self.alpha_l.detach().cpu().item()),
            "Hchain": float(self.alpha_h.detach().cpu().item()),
            "AGchains": float(self.alpha_ag.detach().cpu().item()),
        }


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
    Old-style CoreModel with fixed chain splitting.
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

        bad_tokens = (sequences < 0) | (sequences >= self.embedding.num_embeddings)
        if bad_tokens.any():
            sequences = sequences.masked_fill(bad_tokens, self.pad_id)

        valid_mask = sequences != self.pad_id

        x = self.embedding(sequences.reshape(B * S, L))
        x = x.reshape(B, S, L, D)

        batch_spans = [
            _find_chain_spans_1d(sequences[b, 0], eoc_id=self.eoc_id, pad_id=self.pad_id)
            for b in range(B)
        ]

        for i in range(len(self.self_attention_layers)):
            res = x
            x_attn = torch.zeros_like(x)

            for b in range(B):
                for start, end in batch_spans[b]:
                    if end <= start:
                        continue

                    seg = x[b:b + 1, :, start:end, :]
                    seg_mask = valid_mask[b:b + 1, :, start:end]

                    seg_out = self.self_attention_layers[i](seg, mask=seg_mask)
                    x_attn[b, :, start:end, :] = seg_out.squeeze(0)

            x_attn = self.dropout(x_attn)
            x_attn = self.attn_dym[i](x_attn)
            x_attn = self.attn_drop_paths[i](x_attn)
            x = self.norm_layers[i](res + x_attn)

            res = x
            x_ffn = self.ffn_layers[i](x.reshape(B * S, L, D))
            x_ffn = x_ffn.reshape(B, S, L, D)
            x_ffn = self.dropout(x_ffn)
            x_ffn = self.ffn_dym[i](x_ffn)
            x_ffn = self.ffn_drop_paths[i](x_ffn)
            x = self.ffn_norm_layers[i](res + x_ffn)

            x = x * valid_mask[..., None].to(dtype=x.dtype)

        return x


class CGModel(nn.Module):
    """
    Old-style Core + GAT model with optional frozen ESM2 query-row fusion.
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
        use_esm: bool = True,
        esm_model_name: str = "esm2_t6_8M_UR50D",
        esm_repr_layer: int = None,
        esm_alpha_init: float = 0.0,
        esm_adapter_dropout: float = 0.1,
        esm_freeze: bool = True,
        esm_max_len: int = 1022,
        esm_id_to_aa=None,
        pad_id: int = 1,
        eoc_id: int = 24,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.eoc_id = eoc_id
        self.use_esm = bool(use_esm)

        self.core_model = CoreModel(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers,
            drop_path_rate=drop_path_rate,
            pad_id=pad_id,
            eoc_id=eoc_id,
        )

        if self.use_esm:
            self.esm_query_fusion = FrozenESM2QueryFusion(
                embed_dim=embed_dim,
                dropout=esm_adapter_dropout,
                esm_model_name=esm_model_name,
                esm_repr_layer=esm_repr_layer,
                alpha_init=esm_alpha_init,
                freeze_esm=esm_freeze,
                max_esm_len=esm_max_len,
                pad_id=pad_id,
                eoc_id=eoc_id,
                id_to_aa=esm_id_to_aa,
            )
        else:
            self.esm_query_fusion = None

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

        # Query row only from the MSA output.
        x = x[:, 0, :, :]  # [B, L, D]

        # Optional frozen ESM2 residual fusion into the query row.
        # Because alpha_L/H/AG start at 0, this is initially exactly the MSA-only model.
        if self.use_esm and self.esm_query_fusion is not None:
            query_tokens = sequences[:, 0, :].long()
            batch_spans = [
                _find_chain_spans_1d(query_tokens[b], eoc_id=self.eoc_id, pad_id=self.pad_id)
                for b in range(B)
            ]
            esm_update = self.esm_query_fusion(query_tokens, batch_spans, out_len=seq_len)
            esm_update = esm_update.to(device=x.device, dtype=x.dtype)
            x = x + esm_update

        pos = self.pos_emb[:seq_len].unsqueeze(0).expand(B, -1, -1)
        x = x + self.dropout(pos)

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
        use_esm: bool = True,
        esm_model_name: str = "esm2_t6_8M_UR50D",
        esm_repr_layer: int = None,
        esm_alpha_init: float = 0.0,
        esm_adapter_dropout: float = 0.1,
        esm_freeze: bool = True,
        esm_max_len: int = 1022,
        esm_id_to_aa=None,
        pad_id: int = 1,
        eoc_id: int = 24,
    ):
        super().__init__()

        print(
            f"Initializing MCModel with {num_gnn_layers} GAT layers and "
            f"{num_int_layers} row attention layers (drop_path_rate={drop_path_rate}, use_esm={use_esm})"
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
            use_esm=use_esm,
            esm_model_name=esm_model_name,
            esm_repr_layer=esm_repr_layer,
            esm_alpha_init=esm_alpha_init,
            esm_adapter_dropout=esm_adapter_dropout,
            esm_freeze=esm_freeze,
            esm_max_len=esm_max_len,
            esm_id_to_aa=esm_id_to_aa,
            pad_id=pad_id,
            eoc_id=eoc_id,
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
        use_esm: bool = True,
        esm_model_name: str = "esm2_t6_8M_UR50D",
        esm_repr_layer: int = None,
        esm_alpha_init: float = 0.0,
        esm_adapter_dropout: float = 0.1,
        esm_freeze: bool = True,
        esm_max_len: int = 1022,
        esm_id_to_aa=None,
        pad_id: int = 1,
        eoc_id: int = 24,
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
            use_esm=use_esm,
            esm_model_name=esm_model_name,
            esm_repr_layer=esm_repr_layer,
            esm_alpha_init=esm_alpha_init,
            esm_adapter_dropout=esm_adapter_dropout,
            esm_freeze=esm_freeze,
            esm_max_len=esm_max_len,
            esm_id_to_aa=esm_id_to_aa,
            pad_id=pad_id,
            eoc_id=eoc_id,
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
        use_esm: bool = True,
        esm_model_name: str = "esm2_t6_8M_UR50D",
        esm_repr_layer: int = None,
        esm_alpha_init: float = 0.0,
        esm_adapter_dropout: float = 0.1,
        esm_freeze: bool = True,
        esm_max_len: int = 1022,
        esm_id_to_aa=None,
        pad_id: int = 1,
        eoc_id: int = 24,
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
            use_esm=use_esm,
            esm_model_name=esm_model_name,
            esm_repr_layer=esm_repr_layer,
            esm_alpha_init=esm_alpha_init,
            esm_adapter_dropout=esm_adapter_dropout,
            esm_freeze=esm_freeze,
            esm_max_len=esm_max_len,
            esm_id_to_aa=esm_id_to_aa,
            pad_id=pad_id,
            eoc_id=eoc_id,
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
        use_esm: bool = True,
        esm_model_name: str = "esm2_t6_8M_UR50D",
        esm_repr_layer: int = None,
        esm_alpha_init: float = 0.0,
        esm_adapter_dropout: float = 0.1,
        esm_freeze: bool = True,
        esm_max_len: int = 1022,
        esm_id_to_aa=None,
        pad_id: int = 1,
        eoc_id: int = 24,
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
            use_esm=use_esm,
            esm_model_name=esm_model_name,
            esm_repr_layer=esm_repr_layer,
            esm_alpha_init=esm_alpha_init,
            esm_adapter_dropout=esm_adapter_dropout,
            esm_freeze=esm_freeze,
            esm_max_len=esm_max_len,
            esm_id_to_aa=esm_id_to_aa,
            pad_id=pad_id,
            eoc_id=eoc_id,
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

