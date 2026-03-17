# window_relabel_credit_only.py
import numpy as np
import pandas as pd


def _compute_win_start(df_raw: pd.DataFrame, window_s: int, scheme: str) -> pd.Series:
    """
    scheme:
      - "epoch_floor": floor(ts / window_s) * window_s
      - "t0_floor": floor((ts - t0) / window_s) * window_s + floor(t0 / window_s) * window_s
      - "t0_exact": floor((ts - t0) / window_s) * window_s + t0  
    """
    ts = df_raw["timestamp"].astype(float).to_numpy()
    w = float(window_s)
    t0 = float(np.min(ts))

    if scheme == "epoch_floor":
        ws = np.floor(ts / w) * w
    elif scheme == "t0_floor":
        base = np.floor(t0 / w) * w
        ws = np.floor((ts - t0) / w) * w + base
    elif scheme == "t0_exact":
        ws = np.floor((ts - t0) / w) * w + t0
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    return pd.Series(ws, index=df_raw.index, name="win_start")


def _pick_best_alignment(df_raw: pd.DataFrame, windows_df: pd.DataFrame, window_s: int, group_col: str) -> str:
   
    win_keys = windows_df[[group_col, "win_start"]].copy()
    win_keys["win_start"] = win_keys["win_start"].astype(float)
    win_keys = win_keys.drop_duplicates()

    best_scheme, best_rate = None, -1.0
    for scheme in ["epoch_floor", "t0_floor", "t0_exact"]:
        tmp = df_raw[[group_col]].copy()
        tmp["win_start"] = _compute_win_start(df_raw, window_s, scheme).astype(float)

        if len(tmp) > 200000:
            tmp = tmp.sample(200000, random_state=42)

        merged = tmp.merge(win_keys, on=[group_col, "win_start"], how="left", indicator=True)
        hit_rate = (merged["_merge"] == "both").mean()

        if hit_rate > best_rate:
            best_rate = hit_rate
            best_scheme = scheme

    if best_scheme is None:
        best_scheme = "epoch_floor"

    print(f"[Relabel] alignment scheme={best_scheme}, estimated key hit-rate={best_rate*100:.2f}%")
    return best_scheme


def relabel_windows_credit_only(
    df_raw: pd.DataFrame,
    windows_df: pd.DataFrame,
    window_s: int,
    group_col: str = "api_key",
    credit_endpoint: str = "/api/v1/credit_score",
) -> pd.DataFrame:

    df = df_raw.copy()

    df["timestamp"] = df["timestamp"].astype(float)
    df["endpoint"] = df["endpoint"].astype(str)
    df["label"] = df["label"].astype(str)
    df["type"] = df["type"].astype(str)
    df[group_col] = df[group_col].astype(str)

    windows_df2 = windows_df.copy()
    windows_df2["win_start"] = windows_df2["win_start"].astype(float)
    windows_df2[group_col] = windows_df2[group_col].astype(str)

    scheme = _pick_best_alignment(df, windows_df2, window_s, group_col)
    df["win_start"] = _compute_win_start(df, window_s, scheme).astype(float)

    credit = df[df["endpoint"] == credit_endpoint].copy()
    credit["is_attack_credit"] = (credit["label"] == "Attack").astype(int)

    g_all = credit.groupby([group_col, "win_start"], as_index=False).agg(
        credit_cnt=("endpoint", "size"),
        attack_credit_cnt=("is_attack_credit", "sum"),
    )
    g_all["y_credit_anyattack"] = (g_all["attack_credit_cnt"] > 0).astype(int)
    g_all["attack_credit_ratio"] = g_all["attack_credit_cnt"] / g_all["credit_cnt"].clip(lower=1)

    attack_credit = credit[credit["is_attack_credit"] == 1].copy()
    if len(attack_credit) > 0:
        def _mode_stable(s: pd.Series) -> str:
            vc = s.value_counts()
            top = vc[vc == vc.max()].index.astype(str)
            return sorted(top)[0]

        g_type = attack_credit.groupby([group_col, "win_start"], as_index=False).agg(
            type_mode_attack_credit=("type", _mode_stable),
            attack_credit_rows=("type", "size"),
        )
    else:
        g_type = pd.DataFrame(columns=[group_col, "win_start", "type_mode_attack_credit", "attack_credit_rows"])

    out = windows_df2.merge(g_all, on=[group_col, "win_start"], how="left")
    out = out.merge(g_type, on=[group_col, "win_start"], how="left")

    out["y_binary"] = out["y_credit_anyattack"].fillna(0).astype(int)

    if "type_mode" in out.columns:
        out.loc[out["y_binary"] == 1, "type_mode"] = out.loc[out["y_binary"] == 1, "type_mode_attack_credit"].values
    else:
        out["type_mode"] = np.where(out["y_binary"] == 1, out["type_mode_attack_credit"], "NA")

    print("[Relabel] positive windows by attack type_mode (credit-only):")
    if "type_mode" in out.columns:
        print(out[out["y_binary"] == 1]["type_mode"].value_counts(dropna=False).head(20))

    return out