
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import hashlib
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2_feature_extraction')))

from spp_experiments import (
    ExperimentConfig, WindowConfig,
    load_and_prepare_df, build_windows_df, select_feature_columns
)
from window_relabel_credit_only import relabel_windows_credit_only

TARGET_FPR = 0.01

def hash_api_key(key: str) -> str:
    if pd.isna(key): return "unknown"
    return hashlib.sha256(str(key).encode('utf-8')).hexdigest()[:16]

class GRUAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        attn_weights = torch.softmax(self.attention(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        return self.fc(context).squeeze(-1)

def create_sequences_entity_aware(X_raw, df, lookback=20, group_col="api_key"):
    N, D = X_raw.shape
    X_seq = np.zeros((N, lookback, D), dtype=np.float32)
    df = df.copy()
    df["_row_id"] = np.arange(N)
    for _, g in df.groupby(group_col):
        idx = g.index.to_numpy()
        Xg = X_raw[idx]
        for local_t in range(len(g)):
            seq = Xg[max(0, local_t - lookback + 1): local_t + 1]
            pad = lookback - len(seq)
            if pad > 0:
                seq = np.vstack([np.zeros((pad, D)), seq])
            X_seq[df.loc[idx[local_t], "_row_id"]] = seq
    return X_seq

def main():
    print("=" * 80)
    print(" GRU+Attention Baseline: Strict Entity-Disjoint Split & Grid Search")
    print("=" * 80)

    cfg = ExperimentConfig(csv_file="traffic_logs.csv", group_col="api_key")
    win_cfg = WindowConfig(window_s=60, attack_ratio_thr=0.1)
    
    df_raw = load_and_prepare_df(cfg)
    
    original_keys = df_raw['api_key'].unique()
    atk_keys_original = df_raw[df_raw['label'].str.lower() == 'attack']['api_key'].unique()
    key_mapping = {k: hash_api_key(k) for k in original_keys}
    atk_keys_hashed = set([key_mapping[k] for k in atk_keys_original])
    df_raw['api_key'] = df_raw['api_key'].map(key_mapping)

    windows_df = build_windows_df(df_raw, cfg, win_cfg, True, True)
    windows_df = relabel_windows_credit_only(
        df_raw=df_raw, windows_df=windows_df, window_s=win_cfg.window_s,
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

    grid_space = {
        "hidden_dim": [32, 64, 128],
        "num_layers": [1, 2],
        "dropout": [0.0, 0.2, 0.5],
        "lookback": [10, 20, 30]
    }

    feature_sets = ["baseline23", "ours_all"]

    for feat_name in feature_sets:
        print(f"\n>>> Running Grid Search for Feature Set: {feat_name}")
        feat_cols = select_feature_columns(windows_df, feat_name)
        X_raw = windows_df[feat_cols].to_numpy(dtype=float)
        y_all = windows_df["y_binary"].to_numpy(dtype=int)

        scaler = StandardScaler()
        X_raw[tr] = scaler.fit_transform(X_raw[tr])
        X_raw[val] = scaler.transform(X_raw[val])
        X_raw[te] = scaler.transform(X_raw[te])

        best_val_tpr = -1.0
        best_config = None
        best_model_state = None
        best_lookback = 20
        best_thr = 0.5

        for hd, nl, dp, lb in itertools.product(grid_space["hidden_dim"], grid_space["num_layers"], grid_space["dropout"], grid_space["lookback"]):
            if nl == 1 and dp != 0.0: continue 

            X_seq = create_sequences_entity_aware(X_raw, windows_df, lookback=lb, group_col=cfg.group_col)
            
            X_tr_t = torch.tensor(X_seq[tr], dtype=torch.float32)
            y_tr_t = torch.tensor(y_all[tr], dtype=torch.float32)
            X_val_t = torch.tensor(X_seq[val], dtype=torch.float32)
            y_val_t = y_all[val]

            model = GRUAttention(input_dim=len(feat_cols), hidden_dim=hd, num_layers=nl, dropout=dp)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            pos = max(float(y_tr_t.sum().item()), 1.0)
            neg = float(len(y_tr_t) - pos)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
            loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)

            model.train()
            for _ in range(15): 
                for bx, by in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                p_val = torch.sigmoid(model(X_val_t)).cpu().numpy()
                
                neg_mask_val = (y_val_t == 0) & (~is_atk_all[val])
                if neg_mask_val.sum() > 0:
                    try: thr = float(np.quantile(p_val[neg_mask_val], 1 - TARGET_FPR, method="higher")) + 1e-6
                    except TypeError: thr = float(np.quantile(p_val[neg_mask_val], 1 - TARGET_FPR, interpolation="higher")) + 1e-6
                else: thr = 0.5

                y_pred_val = (p_val > thr).astype(int)
                tp = ((y_val_t == 1) & (y_pred_val == 1)).sum()
                fn = ((y_val_t == 1) & (y_pred_val == 0)).sum()
                val_tpr = tp / max(tp + fn, 1)

                if val_tpr > best_val_tpr:
                    best_val_tpr = val_tpr
                    best_config = {"hidden_dim": hd, "num_layers": nl, "dropout": dp, "lookback": lb}
                    best_model_state = model.state_dict()
                    best_lookback = lb
                    best_thr = thr

        print(f"[*] Best Config for {feat_name}: {best_config} (Val TPR: {best_val_tpr*100:.1f}%)")

        X_seq_best = create_sequences_entity_aware(X_raw, windows_df, lookback=best_lookback, group_col=cfg.group_col)
        X_te_t = torch.tensor(X_seq_best[te], dtype=torch.float32)
        y_te_t = y_all[te]

        best_model = GRUAttention(input_dim=len(feat_cols), **{k: v for k, v in best_config.items() if k != "lookback"})
        best_model.load_state_dict(best_model_state)
        best_model.eval()

        with torch.no_grad():
            p_te = torch.sigmoid(best_model(X_te_t)).cpu().numpy()
            y_pred_te = (p_te > best_thr).astype(int)

            tp_te = ((y_te_t == 1) & (y_pred_te == 1)).sum()
            fn_te = ((y_te_t == 1) & (y_pred_te == 0)).sum()
            te_tpr = tp_te / max(tp_te + fn_te, 1)
            pr_auc = average_precision_score(y_te_t, p_te)

            print(f"--> [Test] GRU+Attention ({feat_name}) PR-AUC: {pr_auc:.4f}")
            print(f"--> [Test] GRU+Attention ({feat_name}) Overall TPR @ 1% Normal-FPR: {te_tpr*100:.1f}%\n")

if __name__ == "__main__":
    main()