import re
import matplotlib.pyplot as plt
import numpy as np


def set_big_style(scale=2.0):
    base = 10
    plt.rcParams.update({
        "font.size": base * scale,
        "axes.titlesize": base * scale * 1.08,
        "axes.labelsize": base * scale * 1.05,
        "xtick.labelsize": base * scale * 0.82,
        "ytick.labelsize": base * scale * 0.82,
        "legend.fontsize": base * scale * 0.62,
        "lines.linewidth": 2.3,
        "axes.linewidth": 1.4,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "savefig.dpi": 400,
        "figure.dpi": 140,
    })


def style_axes(ax):
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_points(ax, x, y, fmt="{:.3f}", fontsize=8, dy=4, color="0.35"):
    for xi, yi in zip(x, y):
        ax.annotate(
            fmt.format(yi),
            (xi, yi),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
        )


def add_panel_label(ax, label, x=-0.12, y=1.02, fontsize=22):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold"
    )


def style_legend(ax, **kwargs):
    defaults = dict(
        frameon=True,
        framealpha=0.9,
        borderaxespad=0.3,
        handlelength=2.0,
        labelspacing=0.35,
        borderpad=0.35,
        fancybox=True,
    )
    defaults.update(kwargs)
    return ax.legend(**defaults)


def plot_ablation(ax_auc, ax_aupr):
    variants = ["w/o MSA", "w/o GNN", "w/o Attn", "w/o DyM", "Full"]
    x = np.arange(len(variants))

    paratope_auc = [0.982, 0.964, 0.980, 0.982, 0.982]
    epitope_auc = [0.801, 0.766, 0.798, 0.809, 0.812]
    paratope_aupr = [0.737, 0.584, 0.726, 0.743, 0.768]
    epitope_aupr = [0.449, 0.409, 0.454, 0.458, 0.472]

    # Top: AUC
    l1, = ax_auc.plot(
        x, paratope_auc,
        marker="o", markersize=5.5,
        color="tab:blue",
        label="Paratope AUC"
    )
    l2, = ax_auc.plot(
        x, epitope_auc,
        marker="o", markersize=5.5,
        color="tab:green",
        label="Epitope AUC"
    )

    annotate_points(ax_auc, x, paratope_auc, fontsize=12, dy=6)
    annotate_points(ax_auc, x, epitope_auc, fontsize=12, dy=6)

    ax_auc.set_title("VASCIF Ablations: AUC", pad=10)
    ax_auc.set_ylabel("AUC")
    ax_auc.set_ylim(0.75, 1.00)
    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels([])
    style_axes(ax_auc)
    style_legend(
        ax_auc,
        handles=[l1, l2],
        loc="center right",
        bbox_to_anchor=(0.98, 0.50),
    )

    # Bottom: AUPR
    l3, = ax_aupr.plot(
        x, paratope_aupr,
        marker="s", markersize=5.5,
        linestyle="--", dashes=(4, 2),
        color="tab:orange",
        label="Paratope AUPR"
    )
    l4, = ax_aupr.plot(
        x, epitope_aupr,
        marker="s", markersize=5.5,
        linestyle="--", dashes=(4, 2),
        color="tab:red",
        label="Epitope AUPR"
    )

    annotate_points(ax_aupr, x, paratope_aupr, fontsize=12, dy=6)
    annotate_points(ax_aupr, x, epitope_aupr, fontsize=12, dy=6)

    ax_aupr.set_title("VASCIF Ablations: AUPR", pad=10)
    ax_aupr.set_ylabel("AUPR")
    ax_aupr.set_xticks(x)
    ax_aupr.set_xticklabels(variants)
    ax_aupr.set_ylim(0.39, 0.80)
    style_axes(ax_aupr)
    style_legend(
        ax_aupr,
        handles=[l3, l4],
        loc="center right",
        bbox_to_anchor=(0.98, 0.50),
    )


def plot_ctsr(ax1, ax2):
    x = np.array([0, 1, 2, 3])

    # CTSR results
    ctsr_paratope = [0.768, 0.777, 0.785, 0.786]
    ctsr_epitope = [0.463, 0.472, 0.488, 0.490]

    # Normal soft-restart results
    # Replace these example values with your actual measurements.
    restart_paratope = [0.768, 0.771, 0.772, 0.771]
    restart_epitope = [0.463, 0.475, 0.467, 0.468]

    # -------------------------
    # Paratope
    # -------------------------
    ax1.plot(
        x,
        ctsr_paratope,
        marker="o",
        markersize=6,
        linestyle="-",
        linewidth=2.3,
        color="tab:blue",
        label="CTSR",
    )

    ax1.plot(
        x,
        restart_paratope,
        marker="s",
        markersize=6,
        linestyle="--",
        linewidth=2.3,
        color="tab:orange",
        label="Normal soft restart",
    )

    # Highlight best CTSR result
    max_idx_y1 = int(np.argmax(ctsr_paratope))
    max_y1 = ctsr_paratope[max_idx_y1]

    ax1.scatter(
        x[max_idx_y1],
        max_y1,
        color="red",
        s=55,
        zorder=4,
    )

    ax1.annotate(
        f"Best CTSR: {max_y1:.3f}",
        xy=(x[max_idx_y1], max_y1),
        xytext=(x[max_idx_y1] - 1.25, max_y1 - 0.011),
        arrowprops=dict(
            color="red",
            arrowstyle="->",
            lw=1.2,
        ),
        fontsize=12,
        color="red",
    )

    annotate_points(
        ax1,
        x,
        ctsr_paratope,
        fontsize=11,
        dy=6,
        color="tab:blue",
    )

    annotate_points(
        ax1,
        x,
        restart_paratope,
        fontsize=11,
        dy=-16,
        color="tab:orange",
    )

    ax1.set_ylabel("AUC-PR")
    ax1.set_title("Paratope Prediction AUC-PR", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([])

    # Adjust after inserting your actual values if necessary.
    para_min = min(ctsr_paratope + restart_paratope)
    para_max = max(ctsr_paratope + restart_paratope)
    para_margin = max(0.006, (para_max - para_min) * 0.35)
    ax1.set_ylim(para_min - para_margin, para_max + para_margin)

    style_axes(ax1)
    style_legend(
        ax1,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
    )
    # -------------------------
    # Epitope
    # -------------------------
    ax2.plot(
        x,
        ctsr_epitope,
        marker="o",
        markersize=6,
        linestyle="-",
        linewidth=2.3,
        color="tab:green",
        label="CTSR",
    )

    ax2.plot(
        x,
        restart_epitope,
        marker="s",
        markersize=6,
        linestyle="--",
        linewidth=2.3,
        color="tab:red",
        label="Normal soft restart",
    )

    # Highlight best CTSR result
    max_idx_y2 = int(np.argmax(ctsr_epitope))
    max_y2 = ctsr_epitope[max_idx_y2]

    ax2.scatter(
        x[max_idx_y2],
        max_y2,
        color="red",
        s=55,
        zorder=4,
    )

    ax2.annotate(
        f"Best CTSR: {max_y2:.3f}",
        xy=(x[max_idx_y2], max_y2),
        xytext=(x[max_idx_y2] - 1.25, max_y2 - 0.011),
        arrowprops=dict(
            color="red",
            arrowstyle="->",
            lw=1.2,
        ),
        fontsize=12,
        color="red",
    )

    annotate_points(
        ax2,
        x,
        ctsr_epitope,
        fontsize=11,
        dy=6,
        color="tab:green",
    )

    annotate_points(
        ax2,
        x,
        restart_epitope,
        fontsize=11,
        dy=-16,
        color="tab:red",
    )

    ax2.set_xlabel("Restart step", labelpad=6)
    ax2.set_ylabel("AUC-PR")
    ax2.set_title("Epitope Prediction AUC-PR", pad=8)
    ax2.set_xticks(x)

    epi_min = min(ctsr_epitope + restart_epitope)
    epi_max = max(ctsr_epitope + restart_epitope)
    epi_margin = max(0.006, (epi_max - epi_min) * 0.35)
    ax2.set_ylim(epi_min - epi_margin, epi_max + epi_margin)

    style_axes(ax2)
    style_legend(
        ax2,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
    )

def parse_train_val_blocks(log_file="tv_curve.txt"):
    save_pat = re.compile(r"Saved\s+(.+?\.pth)")
    epoch_pat = re.compile(r"Ep(\d+)\s+LR=.*?train=([0-9.]+)\s+val=([0-9.]+)")

    blocks = []
    current_epochs = []

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = epoch_pat.match(line)
            if m:
                ep = int(m.group(1))
                tr = float(m.group(2))
                va = float(m.group(3))
                current_epochs.append((ep, tr, va))
                continue

            s = save_pat.match(line)
            if s:
                model_name = s.group(1)
                blocks.append((model_name, current_epochs))
                current_epochs = []

    if current_epochs:
        blocks.append(("last_block", current_epochs))

    if not blocks:
        raise ValueError(f"No training blocks were parsed from {log_file}")

    return blocks


def plot_train_val(ax_train, ax_val, log_file="tv_curve.txt"):
    blocks = parse_train_val_blocks(log_file)

    for i, (_, epochs) in enumerate(blocks, start=1):
        xs = [e[0] for e in epochs]
        trains = [e[1] for e in epochs]
        vals = [e[2] for e in epochs]

        ax_train.plot(
            xs, trains,
            marker="o", linewidth=1.2, markersize=2.0,
            label=f"CTSR step {i}"
        )
        ax_val.plot(
            xs, vals,
            marker="o", linewidth=1.2, markersize=2.0,
            label=f"CTSR step {i}"
        )

    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Training loss")
    ax_train.set_title("Training loss across CTSR steps", pad=8)
    style_axes(ax_train)
    style_legend(
        ax_train,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
    )

    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Validation loss")
    ax_val.set_title("Validation loss across CTSR steps", pad=8)
    style_axes(ax_val)
    style_legend(
        ax_val,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
    )


def main():
    set_big_style(scale=2.0)

    fig = plt.figure(figsize=(15.5, 14.8))
    outer = fig.add_gridspec(
        2, 2,
        width_ratios=[1.55, 1.0],
        height_ratios=[1.08, 0.98],
        wspace=0.22,
        hspace=0.28,
    )

    # Top-left: ablation with more separation between AUC and AUPR
    gs_ab = outer[0, 0].subgridspec(2, 1, hspace=0.28)
    ax_ab_auc = fig.add_subplot(gs_ab[0, 0])
    ax_ab_aupr = fig.add_subplot(gs_ab[1, 0])
    plot_ablation(ax_ab_auc, ax_ab_aupr)
    add_panel_label(ax_ab_auc, "A")

    # Top-right: CTSR with extra separation
    gs_ctsr = outer[0, 1].subgridspec(2, 1, hspace=0.34)
    ax_ctsr_para = fig.add_subplot(gs_ctsr[0, 0])
    ax_ctsr_epi = fig.add_subplot(gs_ctsr[1, 0])
    plot_ctsr(ax_ctsr_para, ax_ctsr_epi)
    add_panel_label(ax_ctsr_para, "B")

    # Bottom: training / validation
    gs_tv = outer[1, :].subgridspec(1, 2, wspace=0.14)
    ax_train = fig.add_subplot(gs_tv[0, 0])
    ax_val = fig.add_subplot(gs_tv[0, 1])
    plot_train_val(ax_train, ax_val, log_file="tv_curve.txt")
    add_panel_label(ax_train, "C")

    plt.tight_layout()
    plt.savefig("combined_all.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
