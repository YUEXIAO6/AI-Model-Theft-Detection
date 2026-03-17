
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2_feature_extraction')))

from spp_experiments import (
    ExperimentConfig, WindowConfig, SPPConfig, ModelConfig,
    load_and_prepare_df, build_windows_df, train_rf,
    select_feature_columns, build_spp_context_features
)
from window_relabel_credit_only import relabel_windows_credit_only

TARGET_FPR = 0.01

def evaluate_drift():
    cfg = ExperimentConfig(csv_file="traffic_logs_drift.csv", group_col="api_key")
    win_cfg = WindowConfig(window_s=60, attack_ratio_thr=0.1)
    spp_cfg = SPPConfig(enabled=True, levels=(1, 2), lookback_windows=20, context="past")
    mcfg = ModelConfig(class_weight="balanced_subsample", random_state=42)

    try:
        df_raw = load_and_prepare_df(cfg)
    except FileNotFoundError:
        print("can not find traffic_logs_drift.csv！Please run server_drift.py and traffic_generator_drift.py")
        return

    windows_df = build_windows_df(df_raw, cfg, win_cfg, True, True)
    windows_df = relabel_windows_credit_only(
        df_raw=df_raw, windows_df=windows_df, window_s=win_cfg.window_s,
        group_col=cfg.group_col, credit_endpoint="/api/v1/credit_score",
    ).sort_values("win_start").reset_index(drop=True)

    n = len(windows_df)
    tr_idx = int(n * 0.5)
    val_idx = int(n * 0.6)
    
    tr = np.arange(0, tr_idx)
    val = np.arange(tr_idx, val_idx)
    te = np.arange(val_idx, n)

    is_atk_all = windows_df[cfg.group_col].astype(str).str.contains("atk", na=False).to_numpy()

    variants = [
        {"name": "IAT/RateOnly + SPP", "feat": "iat_rate_only"},
        {"name": "Baseline23 + SPP", "feat": "baseline23"},
        {"name": "Full(Enhanced) + SPP", "feat": "ours_all"},
    ]

    print("=" * 100)
    print(f"{'Table: Concept Drift Robustness (Train on Phase 1, Test on Phase 2)':^100}")
    print("=" * 100)
    print(f"{'Model':<22} | {'Macro-TPR':<10} | {'S1 TPR':<8} | {'S2 TPR':<8} | {'Win FPR'}")
    print("-" * 100)

    for var in variants:
        feat_cols = select_feature_columns(windows_df, var["feat"])
        X_base = windows_df[feat_cols].to_numpy(dtype=float)
        X_spp, _ = build_spp_context_features(windows_df, feat_cols, cfg.group_col, "win_start", spp_cfg, tr)
        X_all = np.hstack([X_base, X_spp])
        y_all = windows_df["y_binary"].to_numpy(dtype=int)

        clf = train_rf(X_all[tr], y_all[tr], mcfg)
        p_val = clf.predict_proba(X_all[val])[:, 1]
        y_val = y_all[val]
        neg_val = (y_val == 0) & (~is_atk_all[val])
        
        if neg_val.sum() > 0:
            try: thr = float(np.quantile(p_val[neg_val], 1 - TARGET_FPR, method="higher")) + 1e-6
            except TypeError: thr = float(np.quantile(p_val[neg_val], 1 - TARGET_FPR, interpolation="higher")) + 1e-6
        else: thr = 0.5

        p_te = clf.predict_proba(X_all[te])[:, 1]
        y_te = y_all[te]
        df_test = windows_df.iloc[te].copy().reset_index(drop=True)
        df_test["pred"] = (p_te > thr).astype(int)

        test_neg = (y_te == 0) & (~is_atk_all[te])
        actual_win_fpr = (df_test.loc[test_neg, "pred"] == 1).sum() / max(test_neg.sum(), 1)

        tprs = {}
        for s_type in ["S1", "S2"]: 
            mask = df_test["type_mode"].astype(str).str.contains(s_type) & (df_test["y_binary"] == 1)
            tprs[s_type] = (df_test.loc[mask, "pred"] == 1).sum() / mask.sum() if mask.sum() > 0 else float("nan")
        
        valid_tprs = [v for v in tprs.values() if not np.isnan(v)]
        macro_tpr = np.mean(valid_tprs) if valid_tprs else float("nan")

        print(
            f"{var['name']:<22} | {macro_tpr*100:>9.1f}% | "
            f"{(tprs['S1']*100 if tprs['S1']==tprs['S1'] else float('nan')):>5.1f}% | "
            f"{(tprs['S2']*100 if tprs['S2']==tprs['S2'] else float('nan')):>5.1f}% | "
            f"{actual_win_fpr*100:>6.2f}%"
        )

if __name__ == "__main__":
    evaluate_drift()