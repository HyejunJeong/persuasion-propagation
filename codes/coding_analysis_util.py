import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt


def percentile_rank_nan_safe(x):
    x = pd.Series(x)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    mask = x.notna()
    out.loc[mask] = x.loc[mask].rank(pct=True)
    return out.values


def baseline_delta_quantiles(df, metrics):
    base = (
        df[df["tactic"] == BASELINE]
        .groupby("persona")[metrics]
        .mean()
        .rename(columns={m: f"{m}_base" for m in metrics})
    )

    df2 = df[df["tactic"] != BASELINE].merge(base, on="persona", how="left")

    for m in metrics:
        df2[f"d_{m}"] = df2[m] - df2[f"{m}_base"]
        df2[f"q_{m}"] = percentile_rank_nan_safe(df2[f"d_{m}"])

    return df2


def compute_trs_evs_scores(df):
    df = df.copy()

    # ---- TRS ----
    trs_df = baseline_delta_quantiles(df, TRS_METRICS)
    trs_df["trs_score"] = 1.0 - trs_df[[f"q_{m}" for m in TRS_METRICS]].mean(axis=1)

    # ---- EVS ----
    evs_df = baseline_delta_quantiles(df, EVS_METRICS)
    evs_df["q_mean_revision_size_inv"] = 1.0 - evs_df["q_mean_revision_size"]
    evs_df["evs_score"] = evs_df[["q_revision_entropy", "q_mean_revision_size_inv"]].mean(axis=1)

    out = trs_df[["persona", "persuaded", "trs_score"]].copy()
    out["evs_score"] = evs_df["evs_score"].values
    return out

def pooled_np_p_test(df, score_col):
    np_vals = df[df["persuaded"] == 0][score_col].dropna()
    p_vals  = df[df["persuaded"] == 1][score_col].dropna()

    u, p = mannwhitneyu(np_vals, p_vals, alternative="two-sided")

    return {
        "NP_mean": np_vals.mean(),
        "P_mean": p_vals.mean(),
        "Δ(P−NP)": p_vals.mean() - np_vals.mean(),
        "p_value": p,
        "n_NP": len(np_vals),
        "n_P": len(p_vals),
    }

def persona_delta_summary(df, score_col):
    deltas = []

    for persona, g in df.groupby("persona"):
        np_vals = g[g["persuaded"] == 0][score_col].dropna()
        p_vals  = g[g["persuaded"] == 1][score_col].dropna()
        if len(np_vals) == 0 or len(p_vals) == 0:
            continue
        deltas.append(p_vals.mean() - np_vals.mean())

    deltas = np.array(deltas)

    return {
        "mean_Δ": deltas.mean(),
        "std_Δ": deltas.std(ddof=1),
        "IQR_Δ": np.percentile(deltas, 75) - np.percentile(deltas, 25),
        "frac_pos": (deltas > 0).mean(),
        "n_personas": len(deltas),
    }
