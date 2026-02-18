import math, os, torch, numpy as np, random
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from Dataloader_itf import SequenceParatopeDataset
from Models_fullnew import ClassificationModel
import torch.nn.functional as F
# ───────────────────────────────────────── SEEDS
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
set_seed(42)

# ───────────────────────────────────────── CONFIG
config = {
    # data / model
    'sequence_file':  'para_tv_esmsequences_1600.npz',
    'data_file':      'para_tv_esminterfaces_1600.npz',
    'edge_file':      'para_tv_esmedges_1600.npz',
    'vocab_size':     31,
    'seq_len':        1600,
    'embed_dim':      256,
    'num_heads':      16,
    'dropout':        0.25,
    'num_layers':     1,
    'num_gnn_layers': 10,
    'num_int_layers': 5,
    'drop_path_rate': 0.20,
    'num_classes':    2,

    # optimisation
    'batch_size':     4,
    'num_epochs':     50,
    'warmup_epochs':  10,
    'swa_start':      20,
    'learning_rate':  1e-4,
    'weight_decay':   1e-4,
    'max_grad_norm':  0.5,
    'accum_steps':    1,

    # early-stop & CV
    'n_splits':       10,
    'early_stop':     20,

    # pos-weight (exp decay for recall)
    'weight_start':   20.0,
    'weight_end':     1.0,
    'weight_anneal_tau': 8.0,

    # graph-specific regularization (ramped later)
    'dropedge_min':   0.05,
    'dropedge_max':   0.15,
    'token_drop_p':   0.05,   # PAD=1
    'apply_feature_mask_on_train_only': True,

    # region-smoothing loss (graph-based)
    'smooth_lambda':  0.05,
    'smooth_ramp_start': 5,
    'smooth_ramp_len':   10,

    # AUCPR surrogate loss
    'aucpr_alpha':    0.4,    # mix weight for AUCPR loss (0.2–0.5 typical)
    'aucpr_bins':     64,     # number of soft thresholds (32–128)
    'aucpr_temp':     0.05,   # sigmoid temperature for soft thresholding (0.03–0.1)

    # loss
    'label_smoothing': 0.05,
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(config)
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

# ───────────────────────────────────────── Collate
def custom_collate_fn(batch):
    seqs, labels, edges = zip(*batch)
    seqs   = torch.stack(seqs)                               # [B, M, L] or [B, L]
    labels = torch.tensor(np.array(labels), dtype=torch.long)  # [B, L]
    max_e  = max(e.shape[0] for e in edges)
    pads   = []
    for e in edges:
        pad = -torch.ones((2, max_e), dtype=torch.long)      # -1: no-edge
        pad[:, :e.shape[0]] = e.T.clone().detach()
        pads.append(pad)
    edges = torch.stack(pads)                                # [B, 2, Emax]
    return edges, seqs, labels

# ───────────────────────────────────────── Graph regs
def random_p(min_p, max_p):
    return min_p + (max_p - min_p) * random.random()

def dropedge_padded(padded_edges, p=0.05):
    if p <= 0: return padded_edges
    B, _, Emax = padded_edges.shape
    pe = padded_edges.clone()
    for b in range(B):
        col = pe[b]
        valid_mask = (col[0] >= 0) & (col[1] >= 0)
        idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0: continue
        k = int(p * idx.numel())
        if k <= 0: continue
        drop_idx = idx[torch.randperm(idx.numel(), device=pe.device)[:k]]
        pe[b, :, drop_idx] = -1
    return pe

def token_dropout(seqs, p=0.05, pad_id=1):
    """Randomly replace tokens with PAD in [B, M, L] indices. Protect query row M=0."""
    if p <= 0:
        return seqs
    s = seqs.clone()
    mask = (torch.rand_like(s.float()) < p) & (s != pad_id) & (s != -1)
    if s.dim() == 3:  # [B, M, L]
        mask[:, 0, :] = False
    s[mask] = pad_id
    return s

# ───────────────────────────────────────── Region smoothing loss
def graph_smooth_loss(logits, padded_edges, query_row=0, reduction="mean"):
    """
    Smoothness regularizer for binary classification logits.

    Args:
        logits:        [B, M, L, 2] tensor (class logits).
        padded_edges:  [B, 2, Emax] edge list with -1 as padding.
        query_row:     which MSA row to use (default 0 = main sequence).
        reduction:     "mean" or "sum".

    Returns:
        Scalar smoothness penalty.
    """
    # extract query row → [B, L, 2]
    p_pos = F.softmax(logits[:, query_row, :, :], dim=-1)[..., 1]  # [B, L]
    B, L = p_pos.shape

    total_loss = p_pos.new_zeros(())
    valid_batches = 0

    for b in range(B):
        e = padded_edges[b]  # [2, Emax]
        valid = (e[0] >= 0) & (e[1] >= 0)
        if not valid.any():
            continue

        i = e[0, valid].long()
        j = e[1, valid].long()
        in_range = (i < L) & (j < L)
        if not in_range.any():
            continue

        i = i[in_range]
        j = j[in_range]

        diff = (p_pos[b, i] - p_pos[b, j]) ** 2
        total_loss += diff.mean()
        valid_batches += 1

    if valid_batches == 0:
        return p_pos.new_tensor(0.0)

    return total_loss / valid_batches if reduction == "mean" else total_loss
# ───────────────────────────────────────── AUCPR surrogate (soft PR curve)
def soft_pr_auc_loss(probs: torch.Tensor, labels: torch.Tensor, bins: int = 64, temp: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    """
    Differentiable approximation to AUPR over flattened tokens.
    probs:  [N]   (positive class probabilities)
    labels: [N]   (0/1)
    Returns: loss = 1 - AUPR_hat  (scalar)
    """
    # Soft thresholds in [0,1]
    t = torch.linspace(0.0, 1.0, steps=bins, device=probs.device)  # [K]
    # Soft indicator: predicted positive at threshold t_k
    # shape [N, K]
    I = torch.sigmoid((probs.unsqueeze(1) - t.unsqueeze(0)) / max(1e-6, temp))

    y = labels.float().unsqueeze(1)  # [N,1]
    y_pos = y
    y_neg = 1.0 - y

    # Soft counts per threshold
    TP = (I * y_pos).sum(dim=0)                     # [K]
    FP = (I * y_neg).sum(dim=0)                     # [K]
    FN = ((1.0 - I) * y_pos).sum(dim=0)             # [K]

    precision = TP / (TP + FP + eps)                # [K]
    recall    = TP / (TP + FN + eps)                # [K]

    # Sort by recall for a valid integral
    recall, idx = torch.sort(recall)
    precision = precision[idx]

    # Trapezoidal area under PR
    # torch.trapz integrates y w.r.t. x; x must be increasing
    auc_hat = torch.trapz(precision, recall)        # scalar

    return 1.0 - auc_hat

# ───────────────────────────────────────── Train / Eval
def train_one_epoch(model, loader, ce_criterion, optimizer, scaler,
                    token_p, dropedge_min, dropedge_max, smooth_w,
                    auc_alpha, auc_bins, auc_temp):
    model.train()
    running = 0.0
    for i,(e,s,l) in enumerate(loader):
        e = e.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        l = l.to(device, non_blocking=True)

        # Train-time regularization (ramped)
        e = dropedge_padded(e, p=random_p(dropedge_min, dropedge_max))
        if config['apply_feature_mask_on_train_only']:
            s = token_dropout(s, p=token_p, pad_id=1)

        if i % config['accum_steps']==0:
            optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits,_ = model(sequences=s, padded_edges=e, return_attention=True)  # [B, L, 2]
            ce = ce_criterion(logits.view(-1,2), l.view(-1))

            # AUCPR surrogate on VALID tokens only
            probs = torch.softmax(logits, dim=-1)[..., 1]  # [B, L]
            labels_flat = l.reshape(-1)
            probs_flat  = probs.reshape(-1)
            valid = labels_flat != -1
            if valid.any():
                aucpr_loss = soft_pr_auc_loss(
                    probs_flat[valid],
                    labels_flat[valid],
                    bins=auc_bins,
                    temp=auc_temp
                )
            else:
                aucpr_loss = ce.detach()*0.0

            # Region smoothing on positive logits
            smooth = graph_smooth_loss(logits, e) if smooth_w > 0 else 0.0

            loss = ((1.0 - auc_alpha) * ce + auc_alpha * aucpr_loss + smooth_w * smooth) / config['accum_steps']

        scaler.scale(loss).backward()
        if (i+1)%config['accum_steps']==0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()

        running += loss.item()*config['accum_steps']
    return running/len(loader)

@torch.no_grad()
def evaluate_with_aupr(model, loader, ce_criterion):
    """Return (loss, AUPR) over all tokens with labels != -1. No train-time regs here."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    for e,s,l in loader:
        e = e.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)
        l = l.to(device, non_blocking=True)

        logits = model(sequences=s, padded_edges=e)  # [B, L, 2]
        total_loss += ce_criterion(logits.view(-1,2), l.view(-1)).item()
        probs = torch.softmax(logits, dim=-1)[..., 1]  # [B, L]
        all_probs.append(probs.reshape(-1).detach().cpu())
        all_labels.append(l.reshape(-1).detach().cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    mask = all_labels != -1
    aupr = average_precision_score(all_labels[mask], all_probs[mask]) if mask.sum() else 0.0
    return total_loss/len(loader), aupr

# ───────────────────────────────────────── Dataset & CV
dataset = SequenceParatopeDataset(
    data_file=config['data_file'],
    sequence_file=config['sequence_file'],
    edge_file=config['edge_file'],
    max_len=config['seq_len'])

kf = KFold(n_splits=config['n_splits'], shuffle=True, random_state=42)

# ───────────────────────────────────────── Main loop
scaler = GradScaler('cuda')
for fold,(tr_idx,vl_idx) in enumerate(kf.split(np.arange(len(dataset))),1):
    print(f"\n── Fold {fold}/{config['n_splits']}")
    tr_loader=DataLoader(Subset(dataset,tr_idx),
        batch_size=config['batch_size'], shuffle=True,
        collate_fn=custom_collate_fn)
    vl_loader=DataLoader(
        Subset(dataset,vl_idx),
        batch_size=config['batch_size'], shuffle=False,
        collate_fn=custom_collate_fn)

    model = ClassificationModel(
        vocab_size=config['vocab_size'], seq_len=config['seq_len'],
        embed_dim=config['embed_dim'], num_heads=config['num_heads'],
        dropout=config['dropout'], num_layers=config['num_layers'],
        num_gnn_layers=config['num_gnn_layers'], num_int_layers=config['num_int_layers'],
        num_classes=2, drop_path_rate=config['drop_path_rate'])

    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel.")
        model = nn.DataParallel(model)
    model = model.to(device)
    # Load parameters
    core_params = torch.load('isicParareg_l1_g10_i5_do0.1_dpr0.1_fold5_core.pth', map_location=device)
    # Update model parameters
    model_state = model.state_dict()
    model_state.update(core_params)
    model.load_state_dict(model_state)

    # parameter groups
    base_lr = config['learning_rate']
    core_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if "mc_model" in name:
            core_params.append(param)
        else:
            other_params.append(param)
    optimizer = optim.AdamW(
        [
            {"params": core_params,  "lr": base_lr * 0.5},
            {"params": other_params, "lr": base_lr * 1.0},
        ],
        lr=base_lr,
        weight_decay=config['weight_decay']
    )

    # warm-up → cosine
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch+1)/config['warmup_epochs']
        progress = (epoch-config['warmup_epochs'])/max(1,(config['num_epochs']-config['warmup_epochs']))
        return 0.5*(1+math.cos(math.pi*progress))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # SWA
    swa_model      = AveragedModel(model)
    swa_scheduler  = SWALR(optimizer, swa_lr=config['learning_rate']*0.05)

    best_aupr = -1.0
    patience = 0
    best_state = None

    for epoch in range(config['num_epochs']):
        # Pos-weight: exp decay (strong early → smooth taper)
        pw = config['weight_end'] + (config['weight_start'] - config['weight_end']) * \
             math.exp(-epoch / max(1e-6, config['weight_anneal_tau']))
        class_weights = torch.tensor([1.0, pw], device=device)

        try:
            ce_criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=-1,
                label_smoothing=config['label_smoothing']
            )
        except TypeError:
            ce_criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=-1
            )

        # Reg ramp: start after 15 → full by 25
        if epoch <= 5:
            phase = 0.0
        else:
            phase = min(1.0, (epoch - 5) / 10.0)
        curr_token_p  = config['token_drop_p'] * phase
        curr_drop_min = config['dropedge_min'] * phase
        curr_drop_max = config['dropedge_max'] * phase

        # Smooth ramp
        if epoch <= config['smooth_ramp_start']:
            s_phase = 0.0
        else:
            s_phase = min(1.0, (epoch - config['smooth_ramp_start']) / max(1, config['smooth_ramp_len']))
        curr_smooth_lambda = config['smooth_lambda'] * s_phase

        tr_loss = train_one_epoch(
            model, tr_loader, ce_criterion, optimizer, scaler,
            token_p=curr_token_p,
            dropedge_min=curr_drop_min,
            dropedge_max=curr_drop_max,
            smooth_w=curr_smooth_lambda,
            auc_alpha=config['aucpr_alpha'],
            auc_bins=config['aucpr_bins'],
            auc_temp=config['aucpr_temp']
        )

        # schedule / SWA
        if epoch < config['swa_start']:
            scheduler.step()
            eval_model = model
        else:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            eval_model = swa_model

        # Train / Val evaluation (AUPR)
        train_loss_eval, train_aupr = evaluate_with_aupr(model, tr_loader, ce_criterion)
        val_loss_eval,   val_aupr   = evaluate_with_aupr(eval_model, vl_loader, ce_criterion)

        print(
            f"Ep{epoch:03d}  LR={optimizer.param_groups[0]['lr']:.2e}  "
            f"pw={pw:.2f}  token_p={curr_token_p:.3f}  dropE=({curr_drop_min:.3f},{curr_drop_max:.3f})  "
            f"smooth={curr_smooth_lambda:.3f}  train_loss={tr_loss:.4f}  "
            f"train_AUPR={train_aupr:.4f}  val_loss={val_loss_eval:.4f}  val_AUPR={val_aupr:.4f}"
        )

        # Early stop on VAL AUPR
        if val_aupr > best_aupr + 1e-4:
            best_aupr = val_aupr
            patience = 0
            best_state = eval_model.state_dict()
        else:
            patience += 1
            if patience >= config['early_stop']:
                print("Early stop triggered (val AUPR)")
                break

    # final BN update & save
    if epoch >= config['swa_start']:
        update_bn(tr_loader, swa_model)
        best_state = swa_model.state_dict() if best_state is None else best_state

    ckpt=(f"isiciParareg_l{config['num_layers']}_g{config['num_gnn_layers']}"
          f"_i{config['num_int_layers']}_do{config['dropout']:.2f}"
          f"_dpr{config['drop_path_rate']:.2f}_lr{config['learning_rate']}"
          f"_heads{config['num_heads']}_fold{fold}.pth")
    torch.save(best_state, ckpt)
    print(f"Saved {ckpt}  (best val AUPR={best_aupr:.4f})")

