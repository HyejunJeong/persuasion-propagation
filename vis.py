import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.stats import mannwhitneyu


sns.set_style("white")

palette = {
    0: "steelblue",   # Not persuaded
    1: "darkorange"   # Persuaded
}


# ======================================
# PLOTTING FUNCTION
# ======================================
def plot_coding_delta_boxplot(df, delta_col):
    means = (
        df
        .groupby("persuaded")[delta_col]
        .mean()
        .reindex([0, 1])  # ensure order: not persuaded, persuaded
    )

    stds = (
        df
        .groupby("persuaded")[delta_col]
        .std()
        .reindex([0, 1])
    )

    labels = ["Not Persuaded", "Persuaded"]
    x = np.arange(len(labels))

    plt.figure(figsize=(4, 4))
    plt.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=["steelblue", "darkorange"],
        alpha=0.8,
    )

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, labels)
    plt.ylabel(f"Δ {delta_col.replace('d_', '').replace('_', ' ')}")
    plt.title(f"{delta_col.replace('d_', '').replace('_', ' ').title()}")

    plt.tight_layout()
    plt.show()
    
def plot_pct_box_grid(
    df,
    pc_pct="dPC_brd_pct",
    backbone="gpt",
    figsize_per_cell=(1.8, 1.8),
):
    personas = sorted(df["persona"].unique())
    tactics = sorted(t for t in df["tactic"].unique() if t != "baseline")

    n_rows = len(personas)
    n_cols = len(tactics)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols,
                 figsize_per_cell[1] * n_rows),
        dpi=600,
        sharey=True
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    for i, persona in enumerate(personas):
        for j, tactic in enumerate(tactics):
            ax = axes[i, j]

            sub = df[
                (df["persona"] == persona) &
                (df["tactic"] == tactic) &
                (df["backbone"] == backbone)
            ]

            if sub["persuaded"].nunique() < 1:
                ax.axis("off")
                continue

            x = sub[sub["persuaded"] == 0][pc_pct]
            y = sub[sub["persuaded"] == 1][pc_pct]

            if len(x) < 1 or len(y) < 1:
                ax.axis("off")
                continue

            u, p = mannwhitneyu(x, y, alternative="two-sided")

            sns.boxplot(
                data=sub,
                x="persuaded",
                y=pc_pct,
                hue="persuaded",          # ← add this
                palette=palette,
                legend=False,             # ← suppress legend
                width=0.6,
                showfliers=False,
                ax=ax,
                boxprops=dict(edgecolor="black", linewidth=0.8),
                whiskerprops=dict(color="black", linewidth=0.8),
                capprops=dict(color="black", linewidth=0.8),
                medianprops=dict(color="black", linewidth=0.8),
            )

            ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["NP", "P"], fontsize=10)

            if j == 0:
                ax.set_ylabel(f"{persona}\n% Δ Breadth", fontsize=10)
                ax.set_xlabel("")
                
            else:
                ax.set_ylabel("")
                ax.set_xlabel("")

            if i == 0:
                ax.set_title(tactic.replace("_", "\n"), fontsize=10)

            # ax.text(
            #     0.5, 0.95,
            #     f"p={p:.2e}",
            #     ha="center", va="top",
            #     transform=ax.transAxes,
            #     fontsize=8
            # )

            sns.despine(ax=ax)

    plt.suptitle(
        "Percentage Change in Exploration Breadth (Persuaded vs Not)",
        fontsize=12,
        y=1.0
    )

    plt.tight_layout()
    plt.show()

def plot_coding_delta_side_by_side(df_delta, metric):

    df_bb = df_delta.copy()
    df_bb["row_label"] = df_bb["persona"]

    # Split by persuasion outcome
    df_np = df_bb[df_bb["persuaded"] == 0]
    df_p  = df_bb[df_bb["persuaded"] == 1]

    personas = sorted(df_bb["persona"].unique())
    tactics  = sorted(df_bb["tactic"].unique())

    # Pivot tables (median Δ from baseline)
    mat_np = df_np.pivot_table(
        index="row_label",
        columns="tactic",
        values=metric,
        aggfunc="mean"
    ).reindex(index=personas, columns=tactics)

    mat_p = df_p.pivot_table(
        index="row_label",
        columns="tactic",
        values=metric,
        aggfunc="mean"
    ).reindex(index=personas, columns=tactics)

    # ===== Shared color scale =====
    all_vals = np.concatenate([
        mat_np.values.flatten(),
        mat_p.values.flatten()
    ])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

    # ===== Layout =====
    fig = plt.figure(figsize=(20, 4), dpi=600)
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[1, 1, 0.05],
        wspace=0
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    # cax = fig.add_subplot(gs[0, 2])

    # LEFT — not persuaded
    sns.heatmap(
        mat_np,
        annot=True, fmt=".2f",
        cmap="coolwarm", norm=norm,
        linewidths=0.5,
        ax=ax1,
        # cbar=False
    )
    ax1.set_title(f"NOT PERSUADED — Δ from Baseline\n({metric})")
    ax1.set_xlabel("Tactic")
    ax1.set_ylabel("Persona")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

    # RIGHT — persuaded
    hm = sns.heatmap(
        mat_p,
        annot=True, fmt=".2f",
        cmap="coolwarm", norm=norm,
        linewidths=0.5,
        ax=ax2,
        # cbar=True,
        # cbar_ax=cax
    )
    ax2.set_title(f"PERSUADED — Δ from Baseline\n({metric})")
    ax2.set_xlabel("Tactic")
    ax2.set_ylabel("")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

    # cax.set_ylabel("Mean Δ (Attempt − Baseline)")

    plt.tight_layout()
    plt.show()

def plot_pc_side_by_side(delta_df, pc, backbone):

    df_bb = delta_df[delta_df["backbone"] == backbone].copy()
    df_bb["row_label"] = df_bb["persona"]

    df_np = df_bb[df_bb["condition"] == "not_persuaded"]
    df_p  = df_bb[df_bb["condition"] == "persuaded"]

    personas = sorted(df_bb["persona"].unique())
    tactics  = sorted(df_bb["tactic"].unique())

    mat_np = df_np.pivot_table(
        index="row_label", columns="tactic", values=pc
    ).reindex(index=personas, columns=tactics)

    mat_p = df_p.pivot_table(
        index="row_label", columns="tactic", values=pc
    ).reindex(index=personas, columns=tactics)

    # ===== Shared color scale =====
    all_vals = np.concatenate([mat_np.values.flatten(), mat_p.values.flatten()])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

    # ======== FIXED LAYOUT with GridSpec ========
    fig = plt.figure(figsize=(20, 4), dpi=600)
    gs = gridspec.GridSpec(
        1, 3, 
        width_ratios=[1, 1, 0.05],  # Fix equal subplot widths
        wspace=0
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    # cax = fig.add_subplot(gs[0, 2])   # fixed colorbar location

    # LEFT — not persuaded
    sns.heatmap(
        mat_np, annot=True, fmt=".2f",
        cmap="coolwarm", norm=norm,
        linewidths=0.5, ax=ax1
    )
    ax1.set_title(f"{backbone.upper()} — NOT PERSUADED ({pc})")
    ax1.set_xlabel("Tactic")
    ax1.set_ylabel("Persona")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

    # RIGHT — persuaded
    hm = sns.heatmap(
        mat_p, annot=True, fmt=".2f",
        cmap="coolwarm", norm=norm,
        linewidths=0.5, ax=ax2
    )
    ax2.set_title(f"{backbone.upper()} — PERSUADED ({pc})")
    ax2.set_xlabel("Tactic")
    ax2.set_ylabel("Persona")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    
    # Single aligned colorbar
    # fig.colorbar(hm.collections[0], cax=cax, label=f"{pc} value")

    plt.show()

def plot_pc_difference(
    delta_df,
    pc,
    backbone,
    mode="percent",        # "diff" | "percent"
    clip_pct=100000,          # cap % change to avoid blow-up
    figsize=(5, 3),
):
    """
    Plot persuasion-induced shift heatmap for a given dPC / axis.

    mode="diff"    → raw difference (P - NP)
    mode="percent" → relative % change: (P - NP) / |NP|
    """

    df_bb = delta_df[delta_df["backbone"] == backbone].copy()
    df_bb["row_label"] = df_bb["persona"]

    # Split groups
    df_np = df_bb[df_bb["condition"] == "not_persuaded"]
    df_p  = df_bb[df_bb["condition"] == "persuaded"]

    personas = sorted(df_bb["persona"].unique())
    tactics  = sorted(df_bb["tactic"].unique())

    # Pivot matrices
    mat_np = df_np.pivot_table(
        index="row_label", columns="tactic", values=pc
    ).reindex(index=personas, columns=tactics)

    mat_p = df_p.pivot_table(
        index="row_label", columns="tactic", values=pc
    ).reindex(index=personas, columns=tactics)

    # ===============================
    # Difference / Percentage Matrix
    # ===============================
    if mode == "diff":
        mat = mat_p - mat_np
        title_suffix = "Shift (P − NP)"
        fmt = ".2f"

    elif mode == "percent":
        mat = (mat_p - mat_np) / (np.abs(mat_np) + 1e-6) * 100
        mat = mat.clip(-clip_pct, clip_pct)
        title_suffix = "% Change (P vs NP)"
        fmt = ".1f"

    else:
        raise ValueError("mode must be 'diff' or 'percent'")

    # Shared color scale centered at 0
    v_abs = np.nanpercentile(np.abs(mat.values), 95)
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-v_abs, vmax=v_abs)

    # ===============================
    # Plot
    # ===============================
    plt.figure(figsize=figsize, dpi=600)
    ax = sns.heatmap(
        mat,
        annot=True,
        fmt=fmt,
        cmap="coolwarm",
        norm=norm,
        linewidths=0.5,
        cbar=False,
        annot_kws={"size": 12},
    )

    def pretty_tactic(t):
        mapping = {
            "authority_endorsement": "authority",
            "logical_appeal": "logical",
            "evidence_based": "evidence",
            "priming_urgency": "priming",
        }
        return mapping.get(t, t)

    ax.set_xticklabels(
        [pretty_tactic(t.get_text()) for t in ax.get_xticklabels()],
        rotation=0,
        fontsize=11,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

    plt.title(f"{backbone.upper()} — {pc} {title_suffix}", fontsize=12)
    plt.xlabel("Tactic", fontsize=12)
    plt.ylabel("Persona", fontsize=12)
    plt.tight_layout()
    plt.show()
