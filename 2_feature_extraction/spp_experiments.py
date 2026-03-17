
from __future__ import annotations
import os
import json
import math
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

RESULTS_DIR = "experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
EPS = 1e-9

@dataclass
class SplitConfig:
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42
    mode: str = "group"

@dataclass
class WindowConfig:
    window_s: int = 60
    attack_ratio_thr: float = 0.2
    time_origin: str = "global_min"

@dataclass
class ThresholdConfig:
    mode: str = "low_fpr"
    target_fpr: float = 0.01
    fixed_threshold: float = 0.5
    grid_size: int = 400

@dataclass
class SPPConfig:
    enabled: bool = True
    levels: Tuple[int, ...] = (1, 2)
    lookback_windows: int = 20
    context: str = "past"
    mode_bins: int = 10
    mode_strategy: str = "quantile"

@dataclass
class ModelConfig:
    n_estimators: int = 300
    max_depth: int = 12
    min_samples_leaf: int = 2
    random_state: int = 42
    class_weight: str = "balanced"
    n_jobs: int = -1

@dataclass
class ExperimentConfig:
    csv_file: str = "traffic_logs.csv"
    label_col: str = "label"
    type_col: str = "type"
    api_key_col: str = "api_key"
    timestamp_col: str = "timestamp"
    group_col: str = "api_key"
    strong_negative_types: Tuple[str, ...] = ("LegitBatch", "Monitoring", "RetryStorm")

def _to_numeric_series(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)

def load_and_prepare_df(cfg: ExperimentConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_file)
    for col in [cfg.api_key_col, cfg.timestamp_col]:
        if col not in df.columns: raise ValueError(f"Missing required column: {col}")
    df[cfg.timestamp_col] = _to_numeric_series(df[cfg.timestamp_col], default=0.0)
    for c, dflt in [("fwd_bytes", 0), ("bwd_bytes", 0), ("proc_duration", 0.0)]:
        if c not in df.columns: df[c] = dflt
        df[c] = _to_numeric_series(df[c], default=dflt).clip(lower=0)
    if "status_code" not in df.columns: df["status_code"] = 0
    df["status_code"] = _to_numeric_series(df["status_code"], default=0).astype(int)
    if "endpoint" not in df.columns: df["endpoint"] = "/unknown"
    if "method" not in df.columns: df["method"] = "UNKNOWN"
    if "age" not in df.columns: df["age"] = np.nan
    if "income" not in df.columns: df["income"] = np.nan
    if cfg.label_col in df.columns:
        df[cfg.label_col] = df[cfg.label_col].fillna("Normal").astype(str)
        df["is_attack"] = (df[cfg.label_col].str.lower() == "attack").astype(int)
    else: df["is_attack"] = 0
    if cfg.type_col not in df.columns: df[cfg.type_col] = np.where(df["is_attack"] == 1, "Attack", "Legit")
    df[cfg.type_col] = df[cfg.type_col].fillna("Unknown").astype(str)
    if cfg.group_col not in df.columns: df[cfg.group_col] = df[cfg.api_key_col].fillna("unknown").astype(str)
    df[cfg.api_key_col] = df[cfg.api_key_col].fillna("unknown").astype(str)
    return df

def add_window_id(df: pd.DataFrame, win_cfg: WindowConfig, cfg: ExperimentConfig) -> pd.DataFrame:
    t = df[cfg.timestamp_col].to_numpy(dtype=float)
    origin = float(np.nanmin(t)) if (win_cfg.time_origin == "global_min" and len(t)) else 0.0
    win_id = np.floor((t - origin) / float(win_cfg.window_s)).astype(int)
    out = df.copy()
    out["win_id"] = win_id
    out["win_start"] = origin + out["win_id"].astype(float) * float(win_cfg.window_s)
    return out

def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts.astype(float)
    s = counts.sum()
    if s <= 0: return 0.0
    p = counts / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def _binned_entropy(x: np.ndarray, bins: List[float]) -> Tuple[float, int]:
    if len(x) == 0: return 0.0, 0
    hist, _ = np.histogram(x, bins=bins)
    return _entropy_from_counts(hist), int((hist > 0).sum())

def compute_window_features(df_win: pd.DataFrame, cfg: ExperimentConfig, window_s: int, include_enhanced: bool, include_payload_diversity_proxy: bool) -> pd.DataFrame:
    grp = cfg.group_col
    g = df_win.groupby([grp, "win_id"], sort=False)
    rows = []
    for (gkey, win_id), x in g:
        x = x.sort_values(cfg.timestamp_col)
        ts = x[cfg.timestamp_col].to_numpy(dtype=float)
        fwd, bwd = x["fwd_bytes"].to_numpy(dtype=float), x["bwd_bytes"].to_numpy(dtype=float)
        n = int(len(x))
        dur = float(max(ts.max() - ts.min(), 0.0)) if n else 0.0
        dur_safe = max(dur, 1e-3)
        pkt_per_s_win = float(n / max(float(window_s), 1.0))
        bytes_sum = float(fwd.sum() + bwd.sum())
        bytes_per_s_win = float(bytes_sum / max(float(window_s), 1.0))
        
        if n >= 2: iat = np.diff(ts); iat = iat[iat >= 0]
        else: iat = np.array([], dtype=float)
        iat_mean = float(iat.mean()) if len(iat) else 0.0
        iat_std = float(iat.std(ddof=0)) if len(iat) else 0.0
        iat_min = float(iat.min()) if len(iat) else 0.0
        iat_max = float(iat.max()) if len(iat) else 0.0
        
        attack_ratio = float(x["is_attack"].mean()) if n else 0.0
        endpoint_counts = x["endpoint"].astype(str).value_counts()
        method_counts = x["method"].astype(str).value_counts()
        endpoint_entropy = _entropy_from_counts(endpoint_counts.to_numpy())
        method_entropy = _entropy_from_counts(method_counts.to_numpy())
        top_endpoint_frac = float(endpoint_counts.iloc[0] / max(endpoint_counts.sum(), 1)) if len(endpoint_counts) else 0.0
        sc = x["status_code"].to_numpy(dtype=int)
        frac_2xx = float(np.mean((200 <= sc) & (sc <= 299))) if n else 0.0
        frac_4xx = float(np.mean((400 <= sc) & (sc <= 499))) if n else 0.0
        frac_5xx = float(np.mean((500 <= sc) & (sc <= 599))) if n else 0.0
        size_bins = [0, 50, 100, 200, 400, 800, 1600, 3200, 6400, 20000, 1e12]
        fwd_ent, fwd_bins_nunique = _binned_entropy(fwd, size_bins)
        bwd_ent, bwd_bins_nunique = _binned_entropy(bwd, size_bins)
        fwd_sum, bwd_sum = float(fwd.sum()), float(bwd.sum())
        tot_sum = fwd_sum + bwd_sum
        fwd_max, bwd_max = float(fwd.max()) if n else 0.0, float(bwd.max()) if n else 0.0
        fwd_mean, bwd_mean = float(fwd.mean()) if n else 0.0, float(bwd.mean()) if n else 0.0
        fwd_std, bwd_std = float(fwd.std(ddof=0)) if n else 0.0, float(bwd.std(ddof=0)) if n else 0.0
        
        base23 = {
            "sport": 0.0, "dport": 0.0, "syn": 0.0, "ack": 0.0, "rst": 0.0, "fin": 0.0,
            "fwd_pkts": float(n), "fwd_bytes": fwd_sum, "fwd_bytes_max": fwd_max, "fwd_bytes_avg": fwd_mean, "fwd_bytes_std": fwd_std, "fwd_pkt_s": pkt_per_s_win,
            "bwd_pkts": float(n), "bwd_bytes": bwd_sum, "bwd_bytes_max": bwd_max, "bwd_bytes_avg": bwd_mean, "bwd_bytes_std": bwd_std, "bwd_pkt_s": pkt_per_s_win,
            "pkts": float(2 * n), "tot_bytes": tot_sum, "fwd_bwd_ratio": float(fwd_sum / (bwd_sum + EPS)), "pkt_s": pkt_per_s_win, "duration": dur_safe,
        }

        enhanced = {}
        if include_enhanced:
            credit_reqs = x[x["endpoint"] == "/api/v1/credit_score"]
            ages = pd.to_numeric(credit_reqs["age"], errors="coerce").dropna().to_numpy()
            incomes = pd.to_numeric(credit_reqs["income"], errors="coerce").dropna().to_numpy()
            
            if len(ages) > 0:
              
                age_bins = np.floor(ages / 2.0)
                inc_bins = np.floor(incomes / 5000.0)
                age_ent = _entropy_from_counts(pd.Series(age_bins).value_counts().to_numpy())
                inc_ent = _entropy_from_counts(pd.Series(inc_bins).value_counts().to_numpy())
                
                age_steps = np.abs(np.diff(ages))
                inc_steps = np.abs(np.diff(incomes))
                age_step_mean = float(np.mean(age_steps)) if len(age_steps) > 0 else 0.0
                inc_step_mean = float(np.mean(inc_steps)) if len(inc_steps) > 0 else 0.0
                
                unique_pairs = float(len(set(zip(age_bins, inc_bins))))
            else:
                age_ent = inc_ent = age_step_mean = inc_step_mean = unique_pairs = 0.0

            enhanced = {
                "req_cnt": float(n), "proc_dur_mean": float(x["proc_duration"].mean()) if n else 0.0, "proc_dur_std": float(x["proc_duration"].std(ddof=0)) if n else 0.0,
                "iat_mean": iat_mean, "iat_std": iat_std, "iat_min": iat_min, "iat_max": iat_max,
                "endpoint_nunique": float(endpoint_counts.size), "endpoint_entropy": float(endpoint_entropy), "top_endpoint_frac": float(top_endpoint_frac),
                "method_nunique": float(method_counts.size), "method_entropy": float(method_entropy),
                "status_2xx_frac": frac_2xx, "status_4xx_frac": frac_4xx, "status_5xx_frac": frac_5xx,
                "bytes_per_s": bytes_per_s_win,
                "credit_unique_age": float(len(np.unique(ages))),
                "credit_unique_income": float(len(np.unique(incomes))),
                "credit_unique_pairs": unique_pairs, 
                "credit_age_entropy": float(age_ent),
                "credit_income_entropy": float(inc_ent),
                "credit_age_step_mean": float(age_step_mean),
                "credit_income_step_mean": float(inc_step_mean),
            }
            if include_payload_diversity_proxy:
                enhanced.update({
                    "payload_size_entropy_fwd": float(fwd_ent), "payload_size_bins_nunique_fwd": float(fwd_bins_nunique),
                    "payload_size_entropy_bwd": float(bwd_ent), "payload_size_bins_nunique_bwd": float(bwd_bins_nunique),
                })
                
        type_counts = x[cfg.type_col].value_counts(dropna=False)
        row = {
            cfg.group_col: str(gkey),
            cfg.api_key_col: str(x[cfg.api_key_col].iloc[0]) if cfg.api_key_col in x.columns else str(gkey),
            "win_id": int(win_id), "win_start": float(x["win_start"].iloc[0]), "n_rows": int(n),
            "attack_ratio": float(attack_ratio), "y_binary": 0,
            "type_mode": str(type_counts.index[0]) if len(type_counts) else "Unknown",
        }
        row.update(base23)
        row.update(enhanced)
        rows.append(row)
    return pd.DataFrame(rows)

def apply_window_label(windows_df: pd.DataFrame, attack_ratio_thr: float) -> pd.DataFrame:
    out = windows_df.copy()
    out["y_binary"] = (out["attack_ratio"] >= float(attack_ratio_thr)).astype(int)
    return out

def select_feature_columns(windows_df: pd.DataFrame, feature_set: str) -> List[str]:
    base23 = [
        "sport","dport","syn","ack","rst","fin","fwd_pkts","fwd_bytes","fwd_bytes_max","fwd_bytes_avg","fwd_bytes_std","fwd_pkt_s",
        "bwd_pkts","bwd_bytes","bwd_bytes_max","bwd_bytes_avg","bwd_bytes_std","bwd_pkt_s","pkts","tot_bytes","fwd_bwd_ratio","pkt_s","duration",
    ]
    ours_extra = [
        "req_cnt","proc_dur_mean","proc_dur_std","iat_mean","iat_std","iat_min","iat_max",
        "endpoint_nunique","endpoint_entropy","top_endpoint_frac","method_nunique","method_entropy",
        "status_2xx_frac","status_4xx_frac","status_5xx_frac","bytes_per_s",
        "payload_size_entropy_fwd","payload_size_bins_nunique_fwd","payload_size_entropy_bwd","payload_size_bins_nunique_bwd",
        "credit_unique_age", "credit_unique_income", "credit_unique_pairs", "credit_age_entropy", "credit_income_entropy", "credit_age_step_mean", "credit_income_step_mean"
    ]
    def exists(cols: List[str]) -> List[str]: return [c for c in cols if c in windows_df.columns]
        
    if feature_set == "baseline23": return exists(base23)
    if feature_set == "ours_all": return exists(base23 + ours_extra)
    
    if feature_set == "ours_no_payload": 
        no_payload = [c for c in ours_extra if not c.startswith("payload_")]
        return exists(base23 + no_payload)
        
    if feature_set == "ours_no_semantic":
        no_semantic = [c for c in ours_extra if not c.startswith("credit_")]
        return exists(base23 + no_semantic)
        
    if feature_set == "iat_rate_only":
        small = ["req_cnt","iat_mean","iat_std","iat_min","iat_max","fwd_bytes","bwd_bytes","tot_bytes","fwd_bwd_ratio","pkt_s","duration","bytes_per_s","proc_dur_mean"]
        return exists(small)
        
    raise ValueError(f"Unknown feature_set: {feature_set}")

def _segment_indices(n: int, level: int) -> List[Tuple[int, int]]:
    if n <= 0: return [(0, 0)]
    segs = 2 ** (level - 1)
    bounds = []
    for i in range(segs):
        s = int(math.floor(i * n / segs))
        e = int(math.floor((i + 1) * n / segs))
        bounds.append((s, e))
    return bounds

def _compute_bins_from_train(X_train: np.ndarray, n_bins: int, strategy: str) -> List[np.ndarray]:
    d = X_train.shape[1]
    edges = []
    for j in range(d):
        col = X_train[:, j].astype(float)
        if np.all(np.isfinite(col)) and np.nanmax(col) == np.nanmin(col):
            edges.append(np.array([-np.inf, np.inf], dtype=float)); continue
        col = col[np.isfinite(col)]
        if len(col) < max(20, n_bins * 2):
            lo = float(np.nanmin(col)) if len(col) else 0.0
            hi = float(np.nanmax(col)) if len(col) else 1.0
            if hi == lo: e = np.array([-np.inf, np.inf], dtype=float)
            else: e = np.linspace(lo, hi, num=n_bins + 1, dtype=float); e[0] = -np.inf; e[-1] = np.inf
            edges.append(e); continue
        if strategy == "quantile":
            qs = np.linspace(0, 1, n_bins + 1)
            e = np.unique(np.quantile(col, qs).astype(float))
            if len(e) <= 2: e = np.array([-np.inf, np.inf], dtype=float)
            else: e[0] = -np.inf; e[-1] = np.inf
            edges.append(e)
        elif strategy == "uniform":
            lo, hi = float(np.nanmin(col)), float(np.nanmax(col))
            if hi == lo: e = np.array([-np.inf, np.inf], dtype=float)
            else: e = np.linspace(lo, hi, num=n_bins + 1, dtype=float); e[0] = -np.inf; e[-1] = np.inf
            edges.append(e)
    return edges

def _mode_from_binned(x: np.ndarray, edges: np.ndarray) -> float:
    if x.size == 0: return 0.0
    x = x[np.isfinite(x)]
    if x.size == 0: return 0.0
    bin_ids = np.digitize(x, edges[1:-1], right=False)
    if bin_ids.size == 0: return 0.0
    counts = np.bincount(bin_ids)
    m = int(np.argmax(counts))
    left = float(edges[m]) if np.isfinite(edges[m]) else float(np.nanmin(x))
    right = float(edges[m + 1]) if np.isfinite(edges[m + 1]) else float(np.nanmax(x))
    if not np.isfinite(left): left = float(np.nanmin(x))
    if not np.isfinite(right): right = float(np.nanmax(x))
    if right == left: return float(left)
    return float((left + right) / 2.0)

def build_spp_context_features(windows_df: pd.DataFrame, feature_cols: List[str], group_col: str, time_col: str, spp_cfg: SPPConfig, train_idx_for_bins: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    df = windows_df.copy()
    df["_row_id"] = np.arange(len(df))
    df = df.sort_values([group_col, time_col, "_row_id"])
    X_train_for_bins = windows_df.loc[train_idx_for_bins, feature_cols].to_numpy(dtype=float)
    bin_edges = _compute_bins_from_train(X_train_for_bins, n_bins=spp_cfg.mode_bins, strategy=spp_cfg.mode_strategy)
    d = len(feature_cols)
    n_segments_total = sum(2 ** (lv - 1) for lv in spp_cfg.levels)
    ctx = np.zeros((len(df), n_segments_total * 3 * d), dtype=float)
    col_names = []
    for lv in spp_cfg.levels:
        for seg in range(2 ** (lv - 1)):
            for op in ["mean", "max", "mode"]:
                for f in feature_cols: col_names.append(f"spp_l{lv}_seg{seg}_{op}_{f}")
    for _, g in df.groupby(group_col, sort=False):
        idx = g.index.to_numpy()
        Xg = g[feature_cols].to_numpy(dtype=float)
        for local_t in range(len(g)):
            if spp_cfg.context == "full": Xctx = Xg
            else:
                end_t = local_t + 1
                lb = int(spp_cfg.lookback_windows) if spp_cfg.lookback_windows is not None else 0
                beg_t = max(0, end_t - lb) if lb > 0 else 0
                Xctx = Xg[beg_t:end_t, :]
            parts = []
            for lv in spp_cfg.levels:
                for (s, e) in _segment_indices(len(Xctx), lv):
                    segX = Xctx[s:e, :]
                    if segX.size == 0: parts.append(np.zeros((3 * d,), dtype=float)); continue
                    mean_v, max_v = np.mean(segX, axis=0), np.max(segX, axis=0)
                    mode_v = np.zeros((d,), dtype=float)
                    for j in range(d): mode_v[j] = _mode_from_binned(segX[:, j], bin_edges[j])
                    parts.append(np.concatenate([mean_v, max_v, mode_v], axis=0))
            ctx[int(df.loc[idx[local_t], "_row_id"]), :] = np.concatenate(parts, axis=0)
    return ctx, col_names

def build_windows_df(df_raw: pd.DataFrame, cfg: ExperimentConfig, win_cfg: WindowConfig, include_enhanced: bool, include_payload_diversity_proxy: bool) -> pd.DataFrame:
    df2 = add_window_id(df_raw, win_cfg, cfg)
    w = compute_window_features(df2, cfg, window_s=win_cfg.window_s, include_enhanced=include_enhanced, include_payload_diversity_proxy=include_payload_diversity_proxy)
    w = apply_window_label(w, win_cfg.attack_ratio_thr)
    return w.sort_values([cfg.group_col, "win_start", "win_id"]).reset_index(drop=True)

def train_rf(X: np.ndarray, y: np.ndarray, mcfg: ModelConfig) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=mcfg.n_estimators, max_depth=mcfg.max_depth, min_samples_leaf=mcfg.min_samples_leaf, random_state=mcfg.random_state, class_weight=mcfg.class_weight, n_jobs=mcfg.n_jobs)
    clf.fit(X, y.astype(int))
    return clf