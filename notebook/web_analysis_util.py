import os, re, json, glob
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


# ----------------------------
# config
# ----------------------------
BASELINE = "baseline"
BASELINE_KEYS = ["persona", "backbone"]   # pb baseline

TOOLS = [
    "visit_url","web_search","history_back","click","input_text",
    "answer_question","summarize_page","sleep","hover"
]

# NOTE: these are RAW metric names from expanded behavioral_metrics (m_*)
ACTIVITY = ["m_num_web_events", "m_total_duration_s", "tool_drift"]
BREADTH  = ["m_domain_entropy", "m_num_searches", "m_unique_url_ratio",
            "m_num_domains", "domain_kl", "domain_jaccard"]
DEPTH    = ["m_num_unique_urls", "m_num_summaries",
            "m_avg_latency_s", "query_cosine"]

# sign anchors (use RAW metrics here; we’ll anchor via correlation with this metric)
ANCHOR = {
    "act": "m_num_web_events",
    "brd": "m_num_domains",
    "dpt": "m_num_unique_urls",
}


BEHAVIOR_POLICY_PAIRS = {
    "breadth": {
        "topic": "Web research strategy (source diversity)",
        "A": "I should rely on a small number of sources and finish quickly.",
        "B": "I should use diverse sources across domains before concluding.",
    },
    "depth": {
        "topic": "Web research strategy (verification depth)",
        "A": "I should skim sources quickly once I get the main idea.",
        "B": "I should carefully extract details and verify claims before concluding.",
    },
}

# ============================================================
# 1) CLAIM-BASED LOADER (gpt-behavior_traces / neutral / irrelevant)
# ============================================================
FILENAME_RE = re.compile(
    r"(?P<persona>[a-zA-Z0-9]+)_"
    r"(?P<task_id>t\d+)_"
    r"(?P<tactic>[a-zA-Z_]+)_"
    r"(?P<pair_id>c\d+)_"
    r"(?:(?P<run_id>\d+)_)?"
    r"traces\.json$"
)

def parse_filename(fname: str) -> dict:
    m = FILENAME_RE.search(fname)
    if not m:
        return {}
    d = m.groupdict()
    d["run_id"] = int(d["run_id"] or 0)
    return d

def load_traces(folder, backbone):
    rows = []
    for fp in Path(folder).glob("*.json"):
        meta = parse_filename(fp.name)
        if not meta:
            continue
        with open(fp) as f:
            js = json.load(f)
        rows.append({
            **meta,
            **js,
            "backbone": backbone,
            "file": fp.name,
            "folder": os.path.basename(folder),
        })
    return pd.DataFrame(rows)


# ============================================================
# 2) POLICY-AXIS LOADER (UPDATED)
#   online:  gpt_t03_evidence_based_breadth_16_traces.json
#   prefill: gpt_t03_evidence_based_breadth_A_P_45_traces.json
# ============================================================

POLICY_ONLINE_RE = re.compile(
    r'^(?P<persona>[a-zA-Z0-9]+)_'
    r't(?P<task>\d+)_'
    r'(?P<tactic>[a-zA-Z0-9_]+)_'
    r'(?P<policy>breadth|depth|activity)_'
    r'(?P<run>\d+)_traces\.json$'
)

POLICY_PREFILL_RE = re.compile(
    r'^(?P<persona>[a-zA-Z0-9]+)_'
    r't(?P<task>\d+)_'
    r'(?P<tactic>[a-zA-Z0-9_]+)_'
    r'(?P<policy>breadth|depth|activity)_'
    r'(?P<arm>[AB])_'
    r'(?P<persuasion>P|NP)_'
    r'(?P<run>\d+)_traces\.json$'
)

OPINION_PREFILL_RE = re.compile(
    r"^(?P<persona>[a-zA-Z0-9]+)_"          # gpt
    r"(?P<claim>c\d{2})_"                  # c05
    r"(?P<prefill_condition>P|NP|C0)_"     # P / NP / C0
    r"(?P<task>t\d{2})_"                   # t33
    r"(?P<run>\d{2})"                      # 01
    r"_traces\.json$"
)

def _read_json_flexible(path: str) -> pd.DataFrame:
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        with open(path) as f:
            return pd.DataFrame([json.load(f)])

def load_policy_dir(folder: str, backbone_override=None, prefill: bool=False) -> pd.DataFrame:
    rows = []
    for fp in glob.glob(os.path.join(folder, "*_traces.json")):
        fname = os.path.basename(fp)

        m = (POLICY_PREFILL_RE.match(fname) if prefill else POLICY_ONLINE_RE.match(fname))
        if not m:
            continue

        df = _read_json_flexible(fp)

        df["persona"] = m["persona"]
        df["backbone"] = backbone_override or m["persona"]
        df["task_id"] = int(m["task"])
        df["tactic"] = m["tactic"]
        df["policy_axis"] = m["policy"]
        df["run_id"] = int(m["run"])
        df["pair_id"] = -1
        df["file"] = fname
        df["folder"] = os.path.basename(folder)
        df["prefill"] = bool(prefill)

        if prefill:
            df["policy_goal"] = m["arm"]               # A/B
            df["persuaded"] = 1 if m["persuasion"] == "P" else 0
        else:
            df["policy_goal"] = None                  # unknown in filename
            df["persuaded"] = np.nan                  # compute later from trajectory if available

        rows.append(df)

    if not rows:
        raise RuntimeError(f"No policy files found in {folder}")

    return pd.concat(rows, ignore_index=True)

def load_prefill_opinion_dir(folder: str, backbone_override=None) -> pd.DataFrame:
    rows = []
    for fp in glob.glob(os.path.join(folder, "*_traces.json")):
        fname = os.path.basename(fp)
        m = OPINION_PREFILL_RE.match(fname)
        if not m:
            continue

        df = _read_json_flexible(fp)

        persona = m["persona"]
        claim_str = m["claim"]        # "c05"
        cond = m["prefill_condition"] # "P" / "NP" / "C0"
        task_str = m["task"]          # "t33"
        run_id = int(m["run"])        # 1..10 etc

        # ---- filename metadata ----
        df["persona"] = persona
        df["backbone"] = backbone_override or persona

        # # IMPORTANT: task id is now "t33"
        # # Option A (recommended): keep string
        # df["task_id"] = task_str

        # Option B: convert to numeric 33
        df["task_id"] = int(task_str[1:])

        df["tactic"] = "prefill_opinion"    # no tactic in filename anymore
        df["pair_id"] = int(claim_str[1:])  # c05 -> 5
        df["run_id"] = run_id
        df["file"] = fname
        df["folder"] = os.path.basename(folder)

        # ---- prefill labels ----
        df["prefill"] = True
        df["persuasion_mode"] = df.get("persuasion_mode", "opinion")
        df["prefill_condition"] = cond

        # DO NOT call this "persuaded" anymore unless you really mean "P vs not-P"
        # For compatibility with your downstream code:
        # persuaded=1 if P, else 0 (NP and C0 are both non-belief)
        df["persuaded"] = df["prefill_condition"].map({"P": 1, "NP": 0, "C0": 0})

        # optional fields (in case JSON contains them)
        df["target_side"] = df.get("target_side", None)
        df["target_text"] = df.get("target_text", None)
        df["claim_id"] = df.get("claim_id", claim_str)

        rows.append(df)

    if not rows:
        raise RuntimeError(f"No prefill-opinion files found in {folder}")

    return pd.concat(rows, ignore_index=True)

# ============================================================
# 3) PARSING HELPERS
# ============================================================
def get_domain(u):
    try:
        return urlparse(u).netloc
    except Exception:
        return None

def add_parsed_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "visited_urls" not in df.columns:
        df["visited_urls"] = df.get("raw_events", pd.Series([[]]*len(df))).apply(
            lambda evs: [e.get("url") for e in evs if isinstance(e, dict) and e.get("url")]
            if isinstance(evs, list) else []
        )

    df["domains"] = df["visited_urls"].apply(lambda xs: [get_domain(u) for u in xs if u])

    if "search_queries" in df.columns:
        df["queries"] = df["search_queries"].apply(lambda xs: xs if isinstance(xs, list) else [])
    else:
        df["queries"] = df.get("raw_events", pd.Series([[]]*len(df))).apply(
            lambda evs: [e.get("query") for e in evs if isinstance(e, dict) and e.get("query")]
            if isinstance(evs, list) else []
        )

    df["actions"] = df.get("raw_events", pd.Series([[]]*len(df))).apply(
        lambda evs: [e.get("action") for e in evs if isinstance(e, dict) and e.get("action")]
        if isinstance(evs, list) else []
    )

    return df


# ============================================================
# 4) DRIFT METRICS
# ============================================================
def jaccard(a, b):
    A, B = set(a), set(b)
    return np.nan if not (A | B) else len(A & B) / len(A | B)

def cosine_tfidf(q1, q2):
    if not q1 and not q2:
        return 1.0
    if not q1 or not q2:
        return 0.0
    vec = TfidfVectorizer().fit(q1 + q2)
    return cosine_similarity(
        vec.transform([" ".join(q1)]),
        vec.transform([" ".join(q2)])
    )[0, 0]

def kl_divergence(p, q):
    P, Q = Counter(p), Counter(q)
    keys = set(P) | set(Q)
    p = np.array([P[k] for k in keys], float)
    q = np.array([Q[k] for k in keys], float)
    p /= p.sum() + 1e-12
    q /= q.sum() + 1e-12
    return np.sum(p * np.log((p + 1e-12) / (q + 1e-12)))

def tool_counts(actions):
    c = Counter(a for a in actions if a in TOOLS)
    return np.array([c[t] for t in TOOLS], float)

def build_baseline_refs(df_train: pd.DataFrame):
    base = df_train[df_train["tactic"] == BASELINE]
    q, d, t = {}, {}, {}
    for k, g in base.groupby(BASELINE_KEYS):
        q[k] = sum(g["queries"], [])
        d[k] = sum(g["domains"], [])
        t[k] = tool_counts(sum(g["actions"], []))
    return q, d, t

def compute_drift(df: pd.DataFrame, baseline_q: dict, baseline_d: dict, baseline_t: dict) -> pd.DataFrame:
    out = []
    for _, r in df.iterrows():
        key = tuple(r[k] for k in BASELINE_KEYS)
        out.append({
            "query_cosine": cosine_tfidf(r["queries"], baseline_q.get(key, [])),
            "domain_jaccard": jaccard(r["domains"], baseline_d.get(key, [])),
            "domain_kl": kl_divergence(r["domains"], baseline_d.get(key, [])),
            "tool_drift": float(np.abs(tool_counts(r["actions"]) - baseline_t.get(key, 0)).sum()),
        })
    return df.join(pd.DataFrame(out))


# ============================================================
# 5) EXPAND behavioral_metrics -> m_*
#    + normalize key-name variants (fixes m_total_duration_s issue)
# ============================================================
def _coalesce_cols(df, target, candidates):
    if target in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            df[target] = df[c]
            return df
    return df

def expand_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "behavioral_metrics" in df.columns:
        bm = pd.json_normalize(df["behavioral_metrics"]).add_prefix("m_")
        df = pd.concat([df.drop(columns=["behavioral_metrics"]), bm], axis=1)

    # normalize common variants
    df = _coalesce_cols(df, "m_total_duration_s", ["m_total_duration", "m_duration_s", "m_total_time_s"])
    df = _coalesce_cols(df, "m_avg_latency_s",    ["m_avg_latency", "m_latency_s"])

    # derived ratios
    eps = 1e-6
    if "m_num_unique_urls" in df.columns and "m_num_urls" in df.columns:
        df["m_unique_url_ratio"] = df["m_num_unique_urls"] / (df["m_num_urls"] + eps)
    if "m_num_domains" in df.columns and "m_num_urls" in df.columns:
        df["m_url_domain_ratio"] = df["m_num_domains"] / (df["m_num_urls"] + eps)

    return df


# ============================================================
# 6) BASELINE NORMALIZATION -> d_m_*
# ============================================================
def add_delta_metrics(df_all: pd.DataFrame, df_train: pd.DataFrame, baseline_keys: list[str]) -> pd.DataFrame:
    df_all = df_all.copy()
    df_train = df_train.copy()

    metric_cols = [c for c in df_all.columns if c.startswith("m_") and c != "m_num_urls"]
    metric_cols = [c for c in metric_cols if c in df_train.columns]  # safety

    base_means = (
        df_train[df_train["tactic"] == BASELINE]
        .groupby(baseline_keys)[metric_cols]
        .mean()
        .reset_index()
    )

    df = df_all.merge(
        base_means,
        on=baseline_keys,
        how="left",
        suffixes=("", "_base")
    )

    for m in metric_cols:
        base_col = f"{m}_base"
        if base_col in df.columns:
            df[f"d_{m}"] = df[m] - df[base_col]

    return df


# ============================================================
# 7) NEW PCA -> dPC (fit on BASELINE only; center baseline to 0)
# ============================================================
def fit_pca_and_score_baseline(df_train: pd.DataFrame, df_all: pd.DataFrame) -> pd.DataFrame:
    df_all = df_all.copy()

    def _fit_one_construct(name: str, metrics: list[str]) -> None:
        out_col = f"dPC_{name}"
        cols = metrics[:]

        # keep only columns that exist
        cols = [c for c in cols if c in df_all.columns]
        if len(cols) == 0:
            df_all[out_col] = np.nan
            return

        df_all[out_col] = np.nan

        for bb in df_all["backbone"].dropna().unique():
            base_mask = ((df_train["backbone"] == bb) & (df_train["tactic"] == BASELINE))
            all_mask  = (df_all["backbone"] == bb)

            if base_mask.sum() < 3 or all_mask.sum() == 0:
                continue

            X_base = df_train.loc[base_mask, cols]
            X_all  = df_all.loc[all_mask, cols]

            imp = SimpleImputer(strategy="median").fit(X_base)
            Xb = imp.transform(X_base)
            Xa = imp.transform(X_all)

            sc = StandardScaler().fit(Xb)
            Xb = sc.transform(Xb)
            Xa = sc.transform(Xa)

            pca = PCA(n_components=1, random_state=0).fit(Xb)

            pc_base = pca.transform(Xb).ravel()
            pc_all  = pca.transform(Xa).ravel()

            # sign anchor: corr(pc_base, anchor_metric_in_baseline) >= 0
            anchor = ANCHOR.get(name)
            if anchor in cols:
                a_idx = cols.index(anchor)
                a_base = Xb[:, a_idx]
                corr = np.corrcoef(pc_base, a_base)[0, 1]
                if np.isfinite(corr) and corr < 0:
                    pc_base *= -1
                    pc_all  *= -1

            pc_all_centered = pc_all - float(np.nanmean(pc_base))
            df_all.loc[all_mask, out_col] = pc_all_centered

    _fit_one_construct("act", ACTIVITY)
    _fit_one_construct("brd", BREADTH)
    _fit_one_construct("dpt", DEPTH)

    return df_all


# ============================================================
# 8) CONDITION + PERSUADED + DIRECTION LABELS
# ============================================================
def label_condition(row):
    folder = str(row.get("folder", ""))

    if "gpt-prefill-behavior" in folder:
        return "task_relevant"
    if "gpt-prefill-opinion" in folder:
        return "task_relevant"

    if "gpt-behavior-gpt-t3" in folder:
        return "task_relevant"
    if "gpt-no-p-" in folder:
        return "neutral_injection"
    if row.get("tactic") == BASELINE:
        return "no_persuasion"
    return "task_irrelevant"

def persuaded_from_trajectory(row):
    # prefill loader set 0/1
    # if "persuaded" in row and pd.notna(row["persuaded"]):
    #     return int(bool(row["persuaded"]))

    # baseline & neutral are not "persuaded"
    if row.get("tactic") == BASELINE:
        return 0
    if row.get("persuasion_condition") in ("no_persuasion", "neutral_injection"):
        return 0

    tr = row.get("opinion_trajectory", None)
    if not isinstance(tr, dict):
        return 0
    return int(tr.get("prior") != tr.get("post") and tr.get("post") == tr.get("final"))

def behavioral_direction(row):
    """
    Direction-aware label for task_relevant runs only.
    - prefill: uses policy_goal A/B + persuaded (P/NP) to determine target direction.
    - online: policy_goal is None; we can only say "accepted/rejected" if trajectory exists,
             but we CANNOT map to narrower/broader without knowing whether the injected claim was A or B.
    """
    if row.get("persuasion_condition") != "task_relevant":
        return None

    axis = row.get("policy_axis", None)
    goal = row.get("policy_goal", None)    # A/B only for prefill
    p = int(row.get("persuaded", 0) == 1)

    if axis not in ("breadth", "depth", "activity") or goal not in ("A", "B"):
        return None  # online can't be direction-mapped without A/B

    if axis == "breadth":
        if goal == "A":  # target narrower
            return "narrower" if p else "not_narrower"
        else:            # target broader
            return "broader" if p else "not_broader"

    if axis == "depth":
        if goal == "A":  # target shallower
            return "shallower" if p else "not_shallower"
        else:            # target deeper
            return "deeper" if p else "not_deeper"

    return None

# stats

def summarize_series(x):
    x = pd.to_numeric(x, errors="coerce").dropna().values
    n = len(x)
    if n == 0:
        return dict(n=0, mu=np.nan, sd=np.nan)
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n >= 2 else 0.0
    return dict(n=n, mu=mu, sd=sd)

def fmt_stat(s):
    if not np.isfinite(s["mu"]):
        return "nan"
    return f'{s["mu"]:.3f} ± {s["sd"]:.3f} (n={s["n"]})'

def welch_p(a, b):
    a = pd.to_numeric(a, errors="coerce").dropna().values
    b = pd.to_numeric(b, errors="coerce").dropna().values
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)

def delta_ci(base_s, grp_s):
    if base_s["n"] < 2 or grp_s["n"] < 2:
        return dict(delta=np.nan, lo=np.nan, hi=np.nan)
    delta = grp_s["mu"] - base_s["mu"]
    se = np.sqrt((base_s["sd"]**2 / base_s["n"]) + (grp_s["sd"]**2 / grp_s["n"]))
    lo = delta - 1.96 * se
    hi = delta + 1.96 * se
    return dict(delta=float(delta), lo=float(lo), hi=float(hi))

def fmt_ci(lo, hi):
    if not np.isfinite(lo):
        return "nan"
    return f"[{lo:.3f}, {hi:.3f}]"
