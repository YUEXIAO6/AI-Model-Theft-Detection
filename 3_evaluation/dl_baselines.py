
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2_feature_extraction')))
from spp_experiments import (
    ExperimentConfig, WindowConfig,
    load_and_prepare_df, build_windows_df, select_feature_columns
)
from window_relabel_credit_only import relabel_windows_credit_only

TARGET_FPR = 0.01

class GRUAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
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

def run_dl_baseline(feature_set, windows_df, cfg):
    feat_cols = select_feature_columns(windows_df, feature_set)
    X_raw = windows_df[feat_cols].to_numpy(dtype=float)
    y_all = windows_df["y_binary"].to_numpy(dtype=int)
    is_atk_all = windows_df[cfg.group_col].astype(str).str.contains("atk", na=False).to_numpy()

    n = len(X_raw)
    tr_idx, val_idx = int(n * 0.7), int(n * 0.8)

    scaler = StandardScaler()
    X_raw[:tr_idx] = scaler.fit_transform(X_raw[:tr_idx])
    X_raw[tr_idx:] = scaler.transform(X_raw[tr_idx:])

    X_seq = create_sequences_entity_aware(X_raw, windows_df, lookback=20, group_col="api_key")

    X_tr, y_tr = torch.tensor(X_seq[:tr_idx], dtype=torch.float32), torch.tensor(y_all[:tr_idx], dtype=torch.float32)
    X_val, y_val = torch.tensor(X_seq[tr_idx:val_idx], dtype=torch.float32), y_all[tr_idx:val_idx]
    X_te, y_te = torch.tensor(X_seq[val_idx:], dtype=torch.float32), y_all[val_idx:]

    model = GRUAttention(input_dim=len(feat_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    pos = max(float(y_tr.sum().item()), 1.0)
    neg = float(len(y_tr) - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

    model.train()
    for _ in range(15):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        p_val = torch.sigmoid(model(X_val)).cpu().numpy()
        is_atk_val = is_atk_all[tr_idx:val_idx]
        neg_val = (y_val == 0) & (~is_atk_val)
        
        if neg_val.sum() > 0:
            try:
                thr = float(np.quantile(p_val[neg_val], 1 - TARGET_FPR, method="higher")) + 1e-6
            except TypeError:
                thr = float(np.quantile(p_val[neg_val], 1 - TARGET_FPR, interpolation="higher")) + 1e-6
        else:
            thr = 0.5

        p_te = torch.sigmoid(model(X_te)).cpu().numpy()
        y_pred = (p_te > thr).astype(int)

        tp = ((y_te == 1) & (y_pred == 1)).sum()
        fn = ((y_te == 1) & (y_pred == 0)).sum()
        tpr = tp / max(tp + fn, 1)

        print(f"[{feature_set:<12}] PR-AUC: {average_precision_score(y_te, p_te):.4f} | TPR @ 1% Normal-FPR: {tpr*100:.1f}%")

def main():
    print("="*70)
    print(" DL Baseline Comparison (Proving Semantic Features Matter) ")
    print("="*70)
    cfg = ExperimentConfig(csv_file="traffic_logs.csv")
    win_cfg = WindowConfig(window_s=60, attack_ratio_thr=0.1)
    
    df_raw = load_and_prepare_df(cfg)
    windows_df = build_windows_df(df_raw, cfg, win_cfg, True, True)
    windows_df = relabel_windows_credit_only(
        df_raw=df_raw, windows_df=windows_df, window_s=win_cfg.window_s,
        group_col=cfg.group_col, credit_endpoint="/api/v1/credit_score"
    ).sort_values("win_start").reset_index(drop=True)

    run_dl_baseline("baseline23", windows_df, cfg)
    run_dl_baseline("ours_all", windows_df, cfg)

if __name__ == "__main__":
    main()