
import numpy as np
import pandas as pd
import hashlib
import xgboost as xgb
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2_feature_extraction')))

from spp_experiments import (
    ExperimentConfig, WindowConfig, SPPConfig, ModelConfig,
    load_and_prepare_df, build_windows_df, train_rf,
    select_feature_columns, build_spp_context_features
)
from window_relabel_credit_only import relabel_windows_credit_only

TARGET_FPR = 0.01

def hash_api_key(key: str) -> str:
    if pd.isna(key): return "unknown"
    return hashlib.sha256(str(key).encode('utf-8')).hexdigest()[:16]

def evaluate_with_xgboost():
    print("=" * 100)
    print(" EXPERIMENT: Entity-Disjoint Split with XGBoost Baseline")
    print("=" * 100)

    cfg = ExperimentConfig(csv_file="traffic_logs.csv", group_col="api_key")
    win_cfg = WindowConfig(window_s=60, attack_ratio_thr=0.1)
    spp_cfg = SPPConfig(enabled=True, levels=(1, 2), lookback_windows=20, context="past")
    mcfg = ModelConfig(class_weight="balanced_subsample", random_state=42)

    df_raw = load_and_prepare_df(cfg)
    original_keys = df_raw['api_key'].unique()
    atk_keys_original = df_raw[df_raw['label'].str.lower() == 'attack']['api_key'].unique()
    key_mapping = {k: hash_api_key(k) for k in original_keys}
    atk_keys_hashed = set([key_mapping[k] for k in atk_keys_original])
    df_raw['api_key'] = df_raw['api_key'].map(key_mapping)

    windows_df = build_windows_df(df_raw, cfg, win_cfg, True, True)
    windows_df = relabel_windows_credit_only(
        df_raw=df_raw, windows_df=windows_df, window_s=win_cfg.window_s,
        group_col=cfg.group_col, credit_endpoint="/api/v1/credit_score",
    ).sort_values("win_start").reset_index(drop=True)

    is_atk_all = windows_df[cfg.group_col].isin(atk_keys_hashed).to_numpy()
    all_hashed_keys = windows_df[cfg.group_col].unique()
    hashed_atk_keys = [k for k in all_hashed_keys if k in atk_keys_hashed]
    hashed_norm_keys = [k for k in all_hashed_keys if k not in atk_keys_hashed]

    train_atk, test_val_atk = train_test_split(hashed_atk_keys, test_size=0.3, random_state=42)
    val_atk, test_atk = train_test_split(test_val_atk, test_size=0.66, random_state=42)
    train_norm, test_val_norm = train_test_split(hashed_norm_keys, test_size=0.3, random_state=42)
    val_norm, test_norm = train_test_split(test_val_norm, test_size=0.66, random_state=42)

    train_keys = set(train_atk + train_norm)
    val_keys = set(val_atk + val_norm)
    test_keys = set(test_atk + test_norm)

    tr = windows_df.index[windows_df[cfg.group_col].isin(train_keys)].to_numpy()
    val = windows_df.index[windows_df[cfg.group_col].isin(val_keys)].to_numpy()
    te = windows_df.index[windows_df[cfg.group_col].isin(test_keys)].to_numpy()

    variants = [
        {"name": "RF: Baseline23 + SPP", "feat": "baseline23", "model": "rf"},
        {"name": "RF: Full(Enhanced) + SPP", "feat": "ours_all", "model": "rf"},
        {"name": "XGB: Full(Enhanced) + SPP", "feat": "ours_all", "model": "xgb"},
    ]

    print(f"\n{'Model':<28} | {'Macro-TPR':<10} | {'S1 TPR':<8} | {'S2 TPR':<8} | {'Win FPR'}")
    print("-" * 80)

    for var in variants:
        feat_cols = select_feature_columns(windows_df, var["feat"])
        X_base = windows_df[feat_cols].to_numpy(dtype=float)
        X_spp, _ = build_spp_context_features(windows_df, feat_cols, cfg.group_col, "win_start", spp_cfg, tr)
        X_all = np.hstack([X_base, X_spp])
        y_all = windows_df["y_binary"].to_numpy(dtype=int)

        if var["model"] == "rf":
            clf = train_rf(X_all[tr], y_all[tr], mcfg)
            p_val = clf.predict_proba(X_all[val])[:, 1]
            p_te = clf.predict_proba(X_all[te])[:, 1]
        else:
            pos_weight = (len(y_all[tr]) - sum(y_all[tr])) / max(sum(y_all[tr]), 1)
            clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05, 
                scale_pos_weight=pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss'
            )
            clf.fit(X_all[tr], y_all[tr])
            p_val = clf.predict_proba(X_all[val])[:, 1]
            p_te = clf.predict_proba(X_all[te])[:, 1]

        neg_val = (y_all[val] == 0) & (~is_atk_all[val])
        if neg_val.sum() > 0:
            try: thr = float(np.quantile(p_val[neg_val], 1 - TARGET_FPR, method="higher")) + 1e-6
            except TypeError: thr = float(np.quantile(p_val[neg_val], 1 - TARGET_FPR, interpolation="higher")) + 1e-6
        else: thr = 0.5

        df_test = windows_df.iloc[te].copy().reset_index(drop=True)
        df_test["pred"] = (p_te > thr).astype(int)

        test_neg = (y_all[te] == 0) & (~is_atk_all[te])
        actual_win_fpr = (df_test.loc[test_neg, "pred"] == 1).sum() / max(test_neg.sum(), 1)

        tprs = {}
        for s_type in ["S1", "S2"]:
            mask = df_test["type_mode"].astype(str).str.contains(s_type) & (df_test["y_binary"] == 1)
            tprs[s_type] = (df_test.loc[mask, "pred"] == 1).sum() / mask.sum() if mask.sum() > 0 else float("nan")
        
        valid_tprs = [v for v in tprs.values() if not np.isnan(v)]
        macro_tpr = np.mean(valid_tprs) if valid_tprs else float("nan")

        print(
            f"{var['name']:<28} | {macro_tpr*100:>9.1f}% | "
            f"{(tprs['S1']*100 if tprs['S1']==tprs['S1'] else float('nan')):>5.1f}% | "
            f"{(tprs['S2']*100 if tprs['S2']==tprs['S2'] else float('nan')):>5.1f}% | "
            f"{actual_win_fpr*100:>6.2f}%"
        )

if __name__ == "__main__":
    evaluate_with_xgboost()