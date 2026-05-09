"""
Fraud Detection API
====================
FastAPI service that returns fraud probability scores
and SHAP-based feature explanations for each prediction.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="Returns fraud probability scores with SHAP explanations.",
    version="1.0.0",
)

# ── Global Model & Explainer ────────────────────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.bst")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.pkl")
FEATURE_NAMES_PATH = os.getenv("FEATURE_NAMES_PATH", "artifacts/feature_names.json")
THRESHOLD = float(os.getenv("THRESHOLD", "0.236"))

model = None
preprocessor = None
explainer = None
feature_names = None


def load_artifacts():
    """Load model, preprocessor, and SHAP explainer at startup."""
    global model, preprocessor, explainer, feature_names

    # MODEL_PATH may be a pickled sklearn-API XGBClassifier (legacy artifact
    # from the training notebook) or a native XGBoost model file. Detect by
    # the pickle protocol magic byte and unwrap to a Booster either way.
    with open(MODEL_PATH, "rb") as f:
        magic = f.read(1)
    if magic == b"\x80":
        with open(MODEL_PATH, "rb") as f:
            obj = pickle.load(f)
        model = obj.get_booster() if hasattr(obj, "get_booster") else obj
    else:
        model = xgb.Booster()
        model.load_model(MODEL_PATH)

    # Load preprocessor (optional — may not exist for simple setups)
    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Load feature names
    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH, "r") as f:
            file = json.load(f)
            feature_names = file['features']

    # Initialize SHAP TreeExplainer (fast for tree-based models)
    explainer = shap.TreeExplainer(model)


@app.on_event("startup")
async def startup():
    load_artifacts()


# ── Request / Response Schemas ───────────────────────────────────────────────


class Transaction(BaseModel):
    """Single transaction to score."""
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class BatchRequest(BaseModel):
    """One or more transactions to score."""
    instances: list[Transaction]
    parameters: Optional[dict] = None


class FeatureContribution(BaseModel):
    """A single feature's contribution to the prediction."""
    feature: str
    value: float
    shap_value: float
    direction: str  # "increases_risk" or "decreases_risk"


class PredictionResult(BaseModel):
    """Prediction output for a single transaction."""
    fraud_probability: float
    is_fraud: bool
    threshold: float
    top_features: Optional[list[FeatureContribution]] = None
    shap_base_value: Optional[float] = None


class BatchResponse(BaseModel):
    """Batch prediction response."""
    predictions: list[PredictionResult]


# ── Helper Functions ─────────────────────────────────────────────────────────


def get_feature_name(idx: int) -> str:
    """Return feature name by index, falling back to f_0, f_1, etc."""
    if feature_names and idx < len(feature_names):
        return feature_names[idx]
    return f"f_{idx}"


def compute_shap_explanation(
    features: np.ndarray, top_k: int
) -> tuple[list[FeatureContribution], float]:
    """
    Compute SHAP values and return the top-k contributing features
    sorted by absolute impact.
    """
    dmatrix = xgb.DMatrix(features, feature_names=feature_names)
    shap_values = explainer.shap_values(dmatrix)

    # shap_values shape: (1, n_features) for a single instance
    sv = shap_values[0]
    base_value = float(np.ravel(explainer.expected_value)[0])

    # Pair each feature with its SHAP value
    contributions = []
    for i, shap_val in enumerate(sv):
        contributions.append(
            {
                "idx": i,
                "feature": get_feature_name(i),
                "value": float(features[0, i]),
                "shap_value": float(shap_val),
                "abs_shap": abs(float(shap_val)),
            }
        )

    # Sort by absolute SHAP value, take top-k
    contributions.sort(key=lambda x: x["abs_shap"], reverse=True)
    top = contributions[:top_k]

    result = [
        FeatureContribution(
            feature=c["feature"],
            value=c["value"],
            shap_value=c["shap_value"],
            direction="increases_risk" if c["shap_value"] > 0 else "decreases_risk",
        )
        for c in top
    ]

    return result, base_value


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Health check endpoint (required for Vertex AI compatibility)."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=BatchResponse)
async def predict(request: BatchRequest):
    """
    Score one or more transactions for fraud probability.

    Returns fraud scores and optionally SHAP-based feature explanations
    showing which features contributed most to each prediction.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    explain = request.parameters.get("explain", True) if request.parameters else True
    top_k = request.parameters.get("top_k_features", 5) if request.parameters else 5

    results = []

    for txn in request.instances:
        txn_dict = txn.model_dump()
        features = np.array([[txn_dict[f] for f in feature_names]])

        # Apply preprocessor if available
        if preprocessor is not None:
            features = preprocessor.transform(features)

        # Predict
        dmatrix = xgb.DMatrix(features, feature_names=feature_names)
        prob = float(model.predict(dmatrix)[0])

        # SHAP explanation
        top_features = None
        base_value = None
        if explain:
            top_features, base_value = compute_shap_explanation(
                features, top_k
            )

        results.append(
            PredictionResult(
                fraud_probability=round(prob, 6),
                is_fraud=prob >= THRESHOLD,
                threshold=THRESHOLD,
                top_features=top_features,
                shap_base_value=base_value,
            )
        )

    return BatchResponse(predictions=results)


@app.post("/predict/single", response_model=PredictionResult)
async def predict_single(txn: Transaction):
    """Convenience endpoint for scoring a single transaction."""
    batch = BatchRequest(instances=[txn], parameters={"explain": True, "top_k_features": 5})
    response = await predict(batch)
    return response.predictions[0]
