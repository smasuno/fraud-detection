"""
Generate dummy model artifacts for testing the API locally.

Run this before starting the server if you don't have real model files:
    python create_test_artifacts.py
    uvicorn app:app --reload
"""

import json
import numpy as np
import xgboost as xgb
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import make_classification
import os

os.makedirs("artifacts", exist_ok=True)

# ── Feature names ────────────────────────────────────────────────────────────

feature_names = [
    "txn_amount",
    "txn_amount_log",
    "merchant_risk_score",
    "hours_since_last_txn",
    "txn_count_1h",
    "txn_count_24h",
    "avg_amount_7d",
    "std_amount_7d",
    "distance_from_home",
    "is_international",
]

with open("artifacts/feature_names.json", "w") as f:
    json.dump(feature_names, f)

# ── Synthetic training data ──────────────────────────────────────────────────

X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    weights=[0.97, 0.03],  # 3% fraud rate
    random_state=42,
)

# ── Preprocessor ─────────────────────────────────────────────────────────────

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "artifacts/preprocessor.pkl")

# ── Train a simple XGBoost model ─────────────────────────────────────────────

dtrain = xgb.DMatrix(X_scaled, label=y, feature_names=feature_names)

params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "max_depth": 4,
    "learning_rate": 0.1,
    "scale_pos_weight": 30,
    "seed": 42,
}

model = xgb.train(params, dtrain, num_boost_round=50)
model.save_model("artifacts/model.bst")

print("Artifacts created in ./artifacts/")
print("  - model.bst")
print("  - preprocessor.pkl")
print("  - feature_names.json")

# ── Print a sample request for testing ───────────────────────────────────────

sample = X[0].tolist()
print(f"\nSample request body:")
print(json.dumps({
    "transactions": [{"features": [round(v, 4) for v in sample]}],
    "explain": True,
    "top_k_features": 5,
}, indent=2))
