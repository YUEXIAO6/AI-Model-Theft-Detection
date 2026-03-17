
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2_feature_extraction')))

from spp_experiments import ExperimentConfig, WindowConfig, load_and_prepare_df, build_windows_df
from window_relabel_credit_only import relabel_windows_credit_only

def plot_distribution():
    cfg = ExperimentConfig(csv_file="traffic_logs.csv", group_col="api_key")
    win_cfg = WindowConfig(window_s=60, attack_ratio_thr=0.1)
    
    df_raw = load_and_prepare_df(cfg)
    windows_df = build_windows_df(df_raw, cfg, win_cfg, True, True)
    windows_df = relabel_windows_credit_only(
        df_raw=df_raw, windows_df=windows_df, window_s=win_cfg.window_s,
        group_col=cfg.group_col, credit_endpoint="/api/v1/credit_score"
    )

    windows_df['minute'] = (windows_df['win_start'] - windows_df['win_start'].min()) / 60.0
    
    agg_df = windows_df.groupby(['minute', 'type_mode']).size().unstack(fill_value=0)
    
    attack_cols = [c for c in agg_df.columns if "S" in c]
    normal_cols = [c for c in agg_df.columns if c not in attack_cols]
    
    agg_df['Normal Traffic'] = agg_df[normal_cols].sum(axis=1)
    plot_df = agg_df[['Normal Traffic'] + attack_cols].copy()

    plt.figure(figsize=(12, 5))
    colors = ['#cccccc', '#d62728', '#ff7f0e', '#ffbb78', '#9467bd'] 
    
    plot_df.plot(kind='area', stacked=True, color=colors[:len(plot_df.columns)], alpha=0.8, ax=plt.gca())
    
    plt.title('Temporal Distribution of API Sessions (Normal vs. Attack Variants)')
    plt.xlabel('Time (Minutes)')
    plt.ylabel('Number of Active Windows per Minute')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('temporal_distribution.png', dpi=300)
    print("[*] Saved temporal_distribution.png")

if __name__ == "__main__":
    plot_distribution()