
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import hashlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2_feature_extraction')))

from spp_experiments import (
    ExperimentConfig, WindowConfig, SPPConfig,
    load_and_prepare_df, build_windows_df, select_feature_columns, build_spp_context_features
)
from window_relabel_credit_only import relabel_windows_credit_only

def hash_api_key(key: str) -> str:
    if pd.isna(key): return "unknown"
    return hashlib.sha256(str(key).encode('utf-8')).hexdigest()[:16]

def run_information_gain_analysis():
    print("=" * 80)
    print(" EXPERIMENT: SPP Level Information Gain (Mutual Information) Analysis")
    print("=" * 80)

    cfg = ExperimentConfig(csv_file="traffic_logs.csv", group_col="api_key")
    win_cfg = WindowConfig(window_s=60, attack_ratio_thr=0.1)
    
    df_raw = load_and_prepare_df(cfg)
    
    original_keys = df_raw['api_key'].unique()
    atk_keys_original = df_raw[df_raw['label'].str.lower() == 'attack']['api_key'].unique()
    key_mapping = {k: hash_api_key(k) for k in original_keys}
    df_raw['api_key'] = df_raw['api_key'].map(key_mapping)

    windows_df = build_windows_df(df_raw, cfg, win_cfg, True, True)
    windows_df = relabel_windows_credit_only(
        df_raw=df_raw, windows_df=windows_df, window_s=win_cfg.window_s,
        group_col=cfg.group_col, credit_endpoint="/api/v1/credit_score"
    ).sort_values("win_start").reset_index(drop=True)

    feat_cols = select_feature_columns(windows_df, "ours_all")
    y_all = windows_df["y_binary"].to_numpy(dtype=int)
    
    train_idx = np.arange(len(windows_df))

    spp_configs = {
        "Level 1 Only": (1,),
        "Level 2 Only": (2,),
        "Levels 1+2": (1, 2),
        "Levels 1+2+3": (1, 2, 3)
    }

    print(f"{'SPP Configuration':<20} | {'Total Features':<15} | {'Mean Top-20 MI':<15} | {'Max MI (Info Gain)'}")
    print("-" * 75)

    results = {}

    for name, levels in spp_configs.items():
        spp_cfg = SPPConfig(enabled=True, levels=levels, lookback_windows=20, context="past")
        
        X_spp, spp_col_names = build_spp_context_features(
            windows_df, feat_cols, cfg.group_col, "win_start", spp_cfg, train_idx
        )
        
        mi_scores = mutual_info_classif(X_spp, y_all, discrete_features=False, random_state=42)
        
        total_feats = len(mi_scores)
        max_mi = np.max(mi_scores)
        
        top_20_mi = np.sort(mi_scores)[-20:]
        mean_top_20_mi = np.mean(top_20_mi)
        
        results[name] = {
            "max_mi": max_mi,
            "mean_top_20_mi": mean_top_20_mi
        }
        
        print(f"{name:<20} | {total_feats:<15} | {mean_top_20_mi:<15.4f} | {max_mi:.4f}")

    print("\n[*] Analysis Complete.")

if __name__ == "__main__":
    run_information_gain_analysis()