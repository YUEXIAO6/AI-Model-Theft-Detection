
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def target_model(age, income):
    score = 0.4 * np.sin((age / 100) * np.pi) + 0.6 * (1 - np.exp(-(income / 150000) * 3))
    return np.round(np.clip(score, 0, 1), 4)

def fit_surrogate(X_train, y_train, random_state=42):
    n = len(X_train)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    if n < 25:
        model = MLPRegressor(hidden_layer_sizes=(32, 16), solver="lbfgs", alpha=0.01, max_iter=800, random_state=random_state)
    else:
        model = MLPRegressor(hidden_layer_sizes=(64, 32), solver="adam", alpha=0.01, max_iter=2000, early_stopping=True, validation_fraction=0.2, n_iter_no_change=30, random_state=random_state)

    model.fit(Xs, y_train)
    return scaler, model

def main():
    print("[*] Learning Curve Analysis of Surrogate Modeling Using Deterministic Response Replay...")

    df = pd.read_csv("traffic_logs.csv")
    
    extract_df = df[
        (df["endpoint"] == "/api/v1/credit_score") &
        (df["api_key"].astype(str).str.contains("atk"))
    ].copy()

    with open("detection_records.json", "r", encoding="utf-8") as f:
        detections = json.load(f)

    rng = np.random.RandomState(123)
    X_test = np.column_stack((rng.uniform(18, 90, 1000), rng.uniform(10000, 120000, 1000)))
    y_test = np.array([target_model(a, i) for a, i in X_test], dtype=float)

    baseline_mse = mean_squared_error(y_test, np.full_like(y_test, 0.5))
    print(f"[*] Baseline MSE (predict 0.5): {baseline_mse:.4f}")

    plt.figure(figsize=(10, 6))
    
    k_points = [10, 20, 50, 100, 200, 300, 400, 600, 800]
    
    for api_key in extract_df["api_key"].unique():
        entity_queries = extract_df[extract_df["api_key"] == api_key].sort_values("timestamp")
        
        is_detected = api_key in detections
        det_time = float(detections[api_key]) if is_detected else float('inf')
        
        mses = []
        valid_ks = []
        
        for k in k_points:
            if k > len(entity_queries): break
            
            sub_queries = entity_queries.iloc[:k]

            if sub_queries.iloc[-1]["timestamp"] > det_time:
                break
                
            X_train = sub_queries[["age", "income"]].to_numpy(dtype=float)
            y_train = np.array([target_model(a, inc) for a, inc in X_train], dtype=float)
            
            scaler, surrogate = fit_surrogate(X_train, y_train)
            X_test_scaled = scaler.transform(X_test)
            preds = np.clip(surrogate.predict(X_test_scaled), 0, 1)
            mses.append(mean_squared_error(y_test, preds))
            valid_ks.append(k)
            
        if len(valid_ks) > 0:
            color = "red" if is_detected else "gray"
            alpha = 0.8 if is_detected else 0.3
            label = "Blocked by SPP" if (is_detected and api_key == extract_df["api_key"].unique()[0]) else None
            
            plt.plot(valid_ks, mses, marker='o', markersize=4, color=color, alpha=alpha, label=label)
            
            if is_detected:
                plt.scatter(valid_ks[-1], mses[-1], color='black', marker='X', s=100, zorder=5)

    plt.scatter([], [], color='black', marker='X', s=100, label='Detection & Block Point')
    plt.axhline(y=baseline_mse, color='blue', linestyle='--', alpha=0.6, label='Baseline MSE (No Knowledge)')
    
    plt.axhline(y=0.005, color='green', linestyle=':', linewidth=2, label='High-Fidelity Threshold (MSE=0.005)')
    
    plt.title("Model Extraction Learning Curve (Deterministic Replay)")
    plt.xlabel("Number of Successful Queries (k)")
    plt.ylabel("Surrogate Model MSE")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.savefig("log_driven_learning_curve.png", dpi=300)
    print("[*] saved log_driven_learning_curve.png")

if __name__ == "__main__":
    main()