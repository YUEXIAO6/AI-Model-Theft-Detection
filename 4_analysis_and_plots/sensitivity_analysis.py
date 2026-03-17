
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2_feature_extraction')))

from spp_experiments import (
    ExperimentConfig, WindowConfig, SPPConfig, ModelConfig,
    load_and_prepare_df, build_windows_df, train_rf,
    select_feature_columns, build_spp_context_features
)
from window_relabel_credit_only import relabel_windows_credit_only
import hashlib

TARGET_FPR = 0.01

def hash_api_key(key: str) -> str:
    if pd.isna(key): return "unknown"
    return hashlib.sha256(str(key).encode('utf-8')).hexdigest()[:16]

def run_sensitivity():
    print("[*] Running Sensitivity Analysis for Window Size (W) and Lookback (K)...")
    cfg = ExperimentConfig(csv_file="traffic_logs.csv", group_col="api_key")
    df_raw = load_and_prepare_df(cfg)
    
    original_keys = df_raw['api_key'].unique()
    atk_keys_original = df_raw[df_raw['label'].str.lower() == 'attack']['api_key'].unique()
    key_mapping = {k: hash_api_key(k) for k in original_keys}
    atk_keys_hashed = set([key_mapping[k] for k in atk_keys_original])
    df_raw['api_key'] = df_raw['api_key'].map(key_mapping)

    Ws = [30, 60, 120]
    Ks = [5, 10, 20, 40]
    
    results = {w: [] for w in Ws}

    for w in Ws:
        print(f"\n>>> Processing Window Size W = {w}s")
        win_cfg = WindowConfig(window_s=w, attack_ratio_thr=0.1)
        windows_df = build_windows_df(df_raw, cfg, win_cfg, True, True)
        windows_df = relabel_windows_credit_only(
            df_raw=df_raw, windows_df=windows_df, window_s=w,
            group_col=cfg.group_col, credit_endpoint="/api/v1/credit_score"
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

        feat_cols = select_feature_columns(windows_df, "ours_all")
        X_base = windows_df[feat_cols].to_numpy(dtype=float)
        y_all = windows_df["y_binary"].to_numpy(dtype=int)

        for k in Ks:
            print(f"    - Training with Lookback K = {k}")
            spp_cfg = SPPConfig(enabled=True, levels=(1, 2), lookback_windows=k, context="past")
            X_spp, _ = build_spp_context_features(windows_df, feat_cols, cfg.group_col, "win_start", spp_cfg, tr)
            X_all = np.hstack([X_base, X_spp])

            mcfg = ModelConfig(class_weight="balanced_subsample", random_state=42)
            clf = train_rf(X_all[tr], y_all[tr], mcfg)

            p_val = clf.predict_proba(X_all[val])[:, 1]
            neg_val = (y_all[val] == 0) & (~is_atk_all[val])
            try: thr = float(np.quantile(p_val[neg_val], 1 - TARGET_FPR, method="higher")) + 1e-6
            except TypeError: thr = float(np.quantile(p_val[neg_val], 1 - TARGET_FPR, interpolation="higher")) + 1e-6

            p_te = clf.predict_proba(X_all[te])[:, 1]
            y_pred = (p_te > thr).astype(int)
            
            # Calculate Macro-TPR for S1 and S2 only (Adaptive attackers)
            df_test = windows_df.iloc[te].copy().reset_index(drop=True)
            df_test["pred"] = y_pred
            tprs = []
            for s_type in ["S1", "S2"]:
                mask = df_test["type_mode"].astype(str).str.contains(s_type) & (df_test["y_binary"] == 1)
                if mask.sum() > 0:
                    tprs.append((df_test.loc[mask, "pred"] == 1).sum() / mask.sum())
            
            macro_tpr = np.mean(tprs) if tprs else 0
            results[w].append(macro_tpr * 100)

    # Plotting
    plt.figure(figsize=(8, 5))
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, w in enumerate(Ws):
        plt.plot(Ks, results[w], marker=markers[i], color=colors[i], linewidth=2, label=f'Window Size W={w}s')

    plt.title('Sensitivity of Detection Performance to SPP Parameters')
    plt.xlabel('Lookback Windows (K)')
    plt.ylabel('Macro-TPR for Adaptive Attacks (%)')
    plt.xticks(Ks)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('sensitivity_w_k.png', dpi=300)
    print("[*] Sensitivity plot saved to sensitivity_w_k.png")

if __name__ == "__main__":
    run_sensitivity()