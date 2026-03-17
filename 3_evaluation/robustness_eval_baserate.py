
import numpy as np
import pandas as pd
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
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

def get_alerts_per_hour(df_test, group_col="api_key", cooldown_s=600):
    alerts = 0
    for _, g in df_test[df_test["pred"] == 1].groupby(group_col):
        g = g.sort_values("win_start")
        last = -1e18
        for t in g["win_start"].to_numpy():
            if t - last >= cooldown_s:
                alerts += 1
                last = t
    total_hours = max((df_test["win_start"].max() - df_test["win_start"].min()) / 3600.0, 1e-5)
    return alerts / total_hours

def evaluate_base_rate_sensitivity():
    print("=" * 110)
    print(" EXPERIMENT: Base Rate Sensitivity Analysis (Precision/PPV Decay)")
    print("=" * 110)

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
        {"name": "Baseline23 + SPP", "feat": "baseline23"},
        {"name": "Full(Enhanced) + SPP", "feat": "ours_all"},
    ]

    target_base_rates = ["Original", 0.01, 0.005, 0.001]

    for var in variants:
        print(f"\n>>> Evaluating Model: {var['name']}")
        print(f"{'Base Rate':<12} | {'TPR':<8} | {'Normal-FPR':<12} | {'Precision(PPV)':<16} | {'Alert/Hr(Normal)'}")
        print("-" * 75)

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
        
        test_df_full = windows_df.iloc[te].copy().reset_index(drop=True)
        test_df_full["pred"] = (p_te > thr).astype(int)
        
        test_normal_mask = (test_df_full["y_binary"] == 0)
        test_attack_mask = (test_df_full["y_binary"] == 1)
        
        df_normal = test_df_full[test_normal_mask]
        df_attack = test_df_full[test_attack_mask]
        
        n_normal = len(df_normal)
        
        for rate in target_base_rates:
            if rate == "Original":
                df_eval = test_df_full
                actual_rate = len(df_attack) / len(test_df_full)
                rate_str = f"{actual_rate*100:.1f}% (Orig)"
            else:
                # rate = n_attack / (n_normal + n_attack)  =>  n_attack = rate * n_normal / (1 - rate)
                n_attack_needed = int((rate * n_normal) / (1.0 - rate))
                
                if n_attack_needed < len(df_attack):
                    df_attack_sampled = df_attack.sample(n=n_attack_needed, random_state=42)
                else:
                    df_attack_sampled = df_attack 
                    
                df_eval = pd.concat([df_normal, df_attack_sampled]).sort_values("win_start")
                rate_str = f"{rate*100:.1f}%"


            y_true_eval = df_eval["y_binary"].to_numpy()
            y_pred_eval = df_eval["pred"].to_numpy()
            
            # TPR
            tpr = recall_score(y_true_eval, y_pred_eval, zero_division=0)
            
            # Normal-FPR 
            eval_normal_mask = (y_true_eval == 0)
            normal_fpr = y_pred_eval[eval_normal_mask].sum() / max(eval_normal_mask.sum(), 1)
            
            # Precision (PPV)
            precision = precision_score(y_true_eval, y_pred_eval, zero_division=0)
            
            # Alert/Hr (Normal) 
            df_eval_normal_only = df_eval[~df_eval[cfg.group_col].isin(atk_keys_hashed)]
            alert_hr_normal = get_alerts_per_hour(df_eval_normal_only, group_col=cfg.group_col)

            print(f"{rate_str:<12} | {tpr*100:>7.1f}% | {normal_fpr*100:>11.2f}% | {precision*100:>15.1f}% | {alert_hr_normal:>12.2f}")

if __name__ == "__main__":
    evaluate_base_rate_sensitivity()