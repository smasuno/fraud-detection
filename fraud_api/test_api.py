"""
Test the fraud detection API.

Start the server first:
    cd fraud_api
    python create_test_artifacts.py
    uvicorn app:app --port 8080

Then run this script:
    python test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8080"


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    print("=== Health Check ===")
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2))
    print()


def test_single_prediction():
    payload = {
        "features": [0.5, -1.2, 3.4, 0.0, 1.1, -0.3, 2.8, 0.7, -1.5, 0.2]
    }
    r = requests.post(f"{BASE_URL}/predict/single", json=payload)
    print("=== Single Prediction ===")
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Fraud probability: {data['fraud_probability']}")
    print(f"Is fraud: {data['is_fraud']}")
    print(f"Threshold: {data['threshold']}")
    print(f"SHAP base value: {data['shap_base_value']}")
    print("\nTop contributing features:")
    for feat in data["top_features"]:
        arrow = "↑" if feat["direction"] == "increases_risk" else "↓"
        print(f"  {arrow} {feat['feature']:>25s}  value={feat['value']:+.4f}  shap={feat['shap_value']:+.4f}")
    print()


def test_batch_prediction():
    payload = {
        "transactions": [
            {"features": [0.5, -1.2, 3.4, 0.0, 1.1, -0.3, 2.8, 0.7, -1.5, 0.2]},
            {"features": [-0.3, 0.8, -1.1, 2.5, 0.0, 1.7, -0.5, 0.1, 0.9, 0.0]},
            {"features": [3.2, 2.1, 4.5, -0.8, 5.3, -1.2, 6.1, 3.4, -2.1, 1.0]},
        ],
        "explain": True,
        "top_k_features": 3,
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    print("=== Batch Prediction (3 transactions) ===")
    print(f"Status: {r.status_code}")
    data = r.json()
    for i, pred in enumerate(data["predictions"]):
        flag = "FRAUD" if pred["is_fraud"] else "LEGIT"
        print(f"\n  Transaction {i+1}: {pred['fraud_probability']:.4f} [{flag}]")
        for feat in pred["top_features"]:
            arrow = "↑" if feat["direction"] == "increases_risk" else "↓"
            print(f"    {arrow} {feat['feature']:>25s}  shap={feat['shap_value']:+.4f}")
    print()


def test_no_explanation():
    payload = {
        "transactions": [
            {"features": [0.5, -1.2, 3.4, 0.0, 1.1, -0.3, 2.8, 0.7, -1.5, 0.2]},
        ],
        "explain": False,
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    print("=== Prediction Without Explanation ===")
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2))
    print()


if __name__ == "__main__":
    test_health()
    test_single_prediction()
    test_batch_prediction()
    test_no_explanation()
