
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2_feature_extraction')))

from spp_experiments import (
    ExperimentConfig, WindowConfig, SPPConfig, ModelConfig,
    load_and_prepare_df, build_windows_df, train_rf,
    select_feature_columns, build_spp_context_features
)

from window_relabel_credit_only import relabel_windows_credit_only

TARGET_FPR = 0.01

def get_operational_metrics(df_test, p_test, thr, group_col="api_key", cooldown_s=600):
    df_t = df_test.copy()
    df_t["pred"] = (p_test > thr).astype(int)
    df_t["is_attacker_entity"] = df_t[group_col].astype(str).str.contains("atk", na=False)

    def dedup_alerts(dfx):
        alerts = 0
        for _, g in dfx[dfx["pred"] == 1].groupby(group_col):
            g = g.sort_values("win_start")
            last = -1e18
            for t in g["win_start"].to_numpy():
                if t - last >= cooldown_s:
                    alerts += 1
                    last = t
        return alerts

    total_hours = max((df_t["win_start"].max() - df_t["win_start"].min()) / 3600.0, 1e-5)
    alerts_normal = dedup_alerts(df_t[~df_t["is_attacker_entity"]]) / total_hours

    normal_entities = df_t.loc[~df_t["is_attacker_entity"], group_col].unique()
    fp_entities = df_t.loc[(~df_t["is_attacker_entity"]) & (df_t["pred"] == 1), group_col].unique()
    entity_fpr = len(fp_entities) / max(len(normal_entities), 1)

    return entity_fpr, alerts_normal

def evaluate_ablation():
    print("=" * 145)
    print(f"{'Table: Comprehensive Ablation Study (Strict Entity-Disjoint Split, Target Normal-FPR=1%)':^145}")
    print("=" * 145)

    cfg = ExperimentConfig(csv_file="traffic_logs.csv", group_col="api_key")
    win_cfg = WindowConfig(window_s=60, attack_ratio_thr=0.1)
    spp_cfg = SPPConfig(enabled=True, levels=(1, 2), lookback_windows=20, context="past")
    mcfg = ModelConfig(class_weight="balanced_subsample", random_state=42)

    df_raw = load_and_prepare_df(cfg)
    windows_df = build_windows_df(df_raw, cfg, win_cfg, True, True)
    windows_df = relabel_windows_credit_only(
        df_raw=df_raw, windows_df=windows_df, window_s=win_cfg.window_s,
        group_col=cfg.group_col, credit_endpoint="/api/v1/credit_score",
    ).sort_values("win_start").reset_index(drop=True)

    unique_keys = windows_df[cfg.group_col].unique()
    atk_keys = [k for k in unique_keys if "atk" in str(k)]
    norm_keys = [k for k in unique_keys if "atk" not in str(k)]

    train_atk, test_val_atk = train_test_split(atk_keys, test_size=0.3, random_state=42)
    val_atk, test_atk = train_test_split(test_val_atk, test_size=0.66, random_state=42) 
    train_norm, test_val_norm = train_test_split(norm_keys, test_size=0.3, random_state=42)
    val_norm, test_norm = train_test_split(test_val_norm, test_size=0.66, random_state=42)

    train_keys = set(train_atk + train_norm)
    val_keys = set(val_atk + val_norm)
    test_keys = set(test_atk + test_norm)

    tr = windows_df.index[windows_df[cfg.group_col].isin(train_keys)].to_numpy()
    val = windows_df.index[windows_df[cfg.group_col].isin(val_keys)].to_numpy()
    te = windows_df.index[windows_df[cfg.group_col].isin(test_keys)].to_numpy()

    is_atk_all = windows_df[cfg.group_col].astype(str).str.contains("atk", na=False).to_numpy()

    variants = [
        {"name": "IAT/RateOnly + SPP", "feat": "iat_rate_only"},
        {"name": "Baseline23 + SPP", "feat": "baseline23"},
        {"name": "Ours w/o Semantic (No Credit)", "feat": "ours_no_semantic"}, 
        {"name": "Ours w/o Payload Div", "feat": "ours_no_payload"},           
        {"name": "Full(Enhanced) + SPP", "feat": "ours_all"},
    ]

    print(f"{'Model':<30} | {'Macro-TPR':<10} | {'S0 TPR':<8} | {'S1 TPR':<8} | {'S2 TPR':<8} | {'S3 TPR':<8} | {'Win FPR':<8} | {'Ent FPR':<8} | {'Alert/Hr(Norm)'}")
    print("-" * 145)

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
        for s_type in ["S0", "S1", "S2", "S3"]:
            mask = df_test["type_mode"].astype(str).str.contains(s_type) & (df_test["y_binary"] == 1)
            tprs[s_type] = (df_test.loc[mask, "pred"] == 1).sum() / mask.sum() if mask.sum() > 0 else float("nan")
        
        valid_tprs = [v for v in tprs.values() if not np.isnan(v)]
        macro_tpr = np.mean(valid_tprs) if valid_tprs else float("nan")

        ent_fpr, alerts_normal = get_operational_metrics(df_test, p_te, thr, cfg.group_col)

        print(
            f"{var['name']:<30} | {macro_tpr*100:>9.1f}% | "
            f"{(tprs['S0']*100 if tprs['S0']==tprs['S0'] else float('nan')):>5.1f}% | "
            f"{(tprs['S1']*100 if tprs['S1']==tprs['S1'] else float('nan')):>5.1f}% | "
            f"{(tprs['S2']*100 if tprs['S2']==tprs['S2'] else float('nan')):>5.1f}% | "
            f"{(tprs['S3']*100 if tprs['S3']==tprs['S3'] else float('nan')):>5.1f}%"
            f" | {actual_win_fpr*100:>6.2f}% | {ent_fpr*100:>6.1f}% | {alerts_normal:>14.1f}"
        )

if __name__ == "__main__":
    evaluate_ablation()