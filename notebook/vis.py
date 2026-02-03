import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.stats import mannwhitneyu
from pathlib import Path
from typing import List, Dict, Optional


sns.set_style("white")

# ======================================
# CONSTANTS & CONFIGURATION
# ======================================
BASELINE = "baseline"
EPS = 1e-6

TACTIC_COLORS = {
    "baseline": "#666666",
    "logical_appeal": "#1f77b4",
    "authority_endorsement": "#ff7f0e",
    "evidence_based": "#2ca02c",
    "priming_urgency": "#d62728",
    "anchoring": "#9467bd",
    "neutral_injection": "#8c564b",
}

PERSONA_COLORS = {
    "gpt": "#1f77b4",
    "claude": "#ff7f0e",
    "llama": "#2ca02c",
    "mistral": "#d62728",
    "qwen": "#9467bd",
    "gemini": "#8c564b",
    "neutral": "#666666",
}

CODING_RAW_METRICS = [
    "num_errors",
    "num_code_revisions",
    "coding_duration_s",
    "revision_entropy",
    "strategy_switch_rate",
    "overcommitment",
    "mean_revision_size",
    "final_revision_delta",
]

WEB_RAW_METRICS = [
    "num_urls",
    "num_unique_urls",
    "num_domains",
    "domain_entropy",
    "num_searches",
    "num_summaries",
    "avg_latency_s",
    "total_duration_s",
]

palette = {
    0: "steelblue",   # Not persuaded
    1: "darkorange"   # Persuaded
}


# ======================================
# DATA LOADING UTILITIES
# ======================================
def load_jsonl(path: str, backbone: str = "gpt") -> pd.DataFrame:
    """Load JSONL file into dataframe with backbone label."""
    df = pd.read_json(path, lines=True)
    df["backbone"] = backbone
    return df


def load_multiple_files(file_dict: Dict[str, str]) -> pd.DataFrame:
    """Load and concatenate multiple JSONL files.

    Args:
        file_dict: Dict mapping persona/backbone name to file path

    Returns:
        Combined dataframe
    """
    dfs = []
    for name, path in file_dict.items():
        if Path(path).exists():
            df = load_jsonl(path, backbone=name)
            dfs.append(df)
            print(f"  ✓ Loaded {name}: {len(df)} rows")
        else:
            print(f"  ✗ File not found: {path}")

    if not dfs:
        raise ValueError("No files were successfully loaded")

    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ Combined: {len(df_combined)} total rows")
    return df_combined


def filter_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Remove baseline rows from dataframe."""
    return df[df["tactic"] != BASELINE].copy()


def validate_required_columns(df: pd.DataFrame, required: List[str]):
    """Check if dataframe has required columns."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print(f"✅ All required columns present: {required}")


# ======================================
# METRIC CALCULATION
# ======================================
def compute_normalized_metrics(
    df: pd.DataFrame,
    raw_metrics: List[str],
    baseline_label: str = "baseline",
) -> pd.DataFrame:
    """Compute baseline-normalized metrics: (x - baseline_mean) / baseline_std.

    Args:
        df: Input dataframe
        raw_metrics: List of raw metric column names
        baseline_label: Name of baseline condition

    Returns:
        Dataframe with added {metric}_norm columns
    """
    df = df.copy()
    baseline_df = df[df["tactic"] == baseline_label]

    for metric in raw_metrics:
        if metric not in df.columns:
            continue

        baseline_stats = baseline_df.groupby("persona")[metric].agg(["mean", "std"])
        norm_col = f"{metric}_norm"

        def normalize_row(row):
            persona = row["persona"]
            if persona not in baseline_stats.index:
                return np.nan
            mean_val = baseline_stats.loc[persona, "mean"]
            std_val = baseline_stats.loc[persona, "std"]
            if std_val == 0 or pd.isna(std_val):
                return np.nan
            return (row[metric] - mean_val) / std_val

        df[norm_col] = df.apply(normalize_row, axis=1)

    print(f"✅ Normalized metrics computed")
    return df


# ======================================
# STATISTICAL TESTS
# ======================================
def pooled_np_p_test(df: pd.DataFrame, score_col: str) -> Dict:
    """Compare not-persuaded vs persuaded groups using Mann-Whitney U test.

    Args:
        df: Dataframe with 'persuaded' column and score column
        score_col: Name of score column to compare

    Returns:
        Dict with test results
    """
    np_vals = df[df["persuaded"] == 0][score_col].dropna()
    p_vals = df[df["persuaded"] == 1][score_col].dropna()

    if len(np_vals) == 0 or len(p_vals) == 0:
        return {
            "n_np": len(np_vals),
            "n_p": len(p_vals),
            "mean_np": np.nan,
            "mean_p": np.nan,
            "delta": np.nan,
            "u_stat": np.nan,
            "p_value": np.nan,
        }

    u_stat, p_value = mannwhitneyu(np_vals, p_vals, alternative="two-sided")

    return {
        "n_np": len(np_vals),
        "n_p": len(p_vals),
        "mean_np": np_vals.mean(),
        "mean_p": p_vals.mean(),
        "delta": p_vals.mean() - np_vals.mean(),
        "u_stat": u_stat,
        "p_value": p_value,
    }


def persona_delta_summary(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Compute per-persona differences between persuaded and not-persuaded.

    Args:
        df: Dataframe with 'persona', 'persuaded' columns
        score_col: Score column to analyze

    Returns:
        Summary dataframe with per-persona statistics
    """
    rows = []

    for persona, g in df.groupby("persona"):
        np_vals = g[g["persuaded"] == 0][score_col].dropna()
        p_vals = g[g["persuaded"] == 1][score_col].dropna()

        if len(np_vals) == 0 or len(p_vals) == 0:
            continue

        u_stat, p_value = mannwhitneyu(np_vals, p_vals, alternative="two-sided")

        rows.append({
            "persona": persona,
            "n_np": len(np_vals),
            "n_p": len(p_vals),
            "mean_np": np_vals.mean(),
            "mean_p": p_vals.mean(),
            "delta": p_vals.mean() - np_vals.mean(),
            "u_stat": u_stat,
            "p_value": p_value,
        })

    return pd.DataFrame(rows)


def tactic_summary(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Summarize persuasion outcomes by tactic.

    Args:
        df: Dataframe with 'tactic', 'persuaded' columns
        score_col: Score column to analyze

    Returns:
        Summary dataframe with per-tactic statistics
    """
    rows = []

    for tactic, g in df.groupby("tactic"):
        if tactic == BASELINE:
            continue

        n_total = len(g)
        n_persuaded = g["persuaded"].sum()
        persuasion_rate = n_persuaded / n_total if n_total > 0 else 0

        np_vals = g[g["persuaded"] == 0][score_col].dropna()
        p_vals = g[g["persuaded"] == 1][score_col].dropna()

        rows.append({
            "tactic": tactic,
            "n_total": n_total,
            "n_persuaded": n_persuaded,
            "persuasion_rate": persuasion_rate,
            "mean_np": np_vals.mean() if len(np_vals) > 0 else np.nan,
            "mean_p": p_vals.mean() if len(p_vals) > 0 else np.nan,
            "delta": (p_vals.mean() - np_vals.mean()) if len(p_vals) > 0 and len(np_vals) > 0 else np.nan,
        })

    return pd.DataFrame(rows)


# ======================================
# DATA NORMALIZATION
# ======================================
def normalize_persuasion_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize GPT / LLaMA / Mistral persuasion logs into a common schema.

    Output columns guaranteed:
      - prior_choice
      - post_choice          (immediate after persuasion)
      - final_choice         (after distractors)
      - persuaded            (immediate persuasion success)
      - persisted
    """
    if "persona" in df.columns:
        df = df[df["persona"] != "gemma"]

    df = df.copy()
    cols = set(df.columns)

    # GPT-style schema
    if "target_after_persuasion" in cols:
        df["post_choice"] = df["target_after_persuasion"]
        df["final_choice"] = df["choice"]
        df["persuaded"] = df["success_behavior"].astype(int)

    # LLaMA / Mistral schema
    elif "post_choice" in cols and "final_choice" in cols:
        df["persuaded"] = df["persuaded"].astype(int)

    else:
        raise ValueError(f"Unknown schema: {df.columns.tolist()}")

    required = ["prior_choice", "post_choice", "final_choice", "persuaded", "persisted"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    return df


def _get_post_and_final_cols(df: pd.DataFrame):
    """Detect column names for post and final choices."""
    cols = set(df.columns)
    if "target_after_persuasion" in cols and "choice" in cols:
        return "target_after_persuasion", "choice"
    if "post_choice" in cols and "final_choice" in cols:
        return "post_choice", "final_choice"
    raise ValueError(f"Unknown schema: {df.columns.tolist()}")


def aggregate_backbone(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate persuasion data by tactic with persistence status."""
    df = df[~df["prior_choice"].astype(str).str.contains("ERROR", na=False)].copy()

    post_col, final_col = _get_post_and_final_cols(df)

    df["persuaded_flag"] = (df["prior_choice"] != df[post_col])

    df["status"] = df.apply(
        lambda r: "PERSISTED" if r["persisted"] == 1
        else ("FADED" if r["persuaded_flag"] else "NO_CHANGE"),
        axis=1
    )

    agg = (
        df.groupby("tactic")["status"]
        .value_counts()
        .unstack(fill_value=0)
    )
    agg["total"] = agg.sum(axis=1)

    out = pd.DataFrame(index=agg.index)
    out["P"]  = (agg.get("PERSISTED", 0) / agg["total"] * 100).round(2)
    out["F"]  = (agg.get("FADED", 0) / agg["total"] * 100).round(2)
    out["NP"] = (agg.get("NO_CHANGE", 0) / agg["total"] * 100).round(2)

    order = ["none", "anchoring", "authority_endorsement", "evidence_based", "logical_appeal", "priming_urgency"]
    return out.reindex(order)


# ======================================
# PLOTTING FUNCTIONS
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
