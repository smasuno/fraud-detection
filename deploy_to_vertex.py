#!/usr/bin/env python3
"""
Deploy the fraud detection XGBoost model to Vertex AI.

Prerequisites:
    pip install google-cloud-aiplatform
    gcloud auth login
    gcloud auth application-default login

Usage:
    python deploy_to_vertex.py --project YOUR_PROJECT_ID
    python deploy_to_vertex.py --project YOUR_PROJECT_ID --region us-west1
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: str, **kwargs) -> subprocess.CompletedProcess:
    print(f"\n$ {cmd}")
    return subprocess.run(cmd, shell=True, check=True, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy fraud model to Vertex AI")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region (default: us-central1)")
    parser.add_argument("--model-name", default="fraud-detector-xgb", help="Vertex AI model display name")
    parser.add_argument("--endpoint-name", default="fraud-detector-endpoint", help="Vertex AI endpoint display name")
    parser.add_argument("--repo", default="fraud-detection", help="Artifact Registry repository name")
    parser.add_argument("--machine-type", default="n1-standard-2", help="VM type for the serving replica")
    args = parser.parse_args()

    PROJECT = args.project
    REGION = args.region
    IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT}/{args.repo}/fraud-api:latest"

    root = Path(__file__).parent
    fraud_api_dir = root / "fraud_api"
    artifacts_dir = fraud_api_dir / "artifacts"
    model_src = root / "models" / "model.bst"

    if not model_src.exists():
        sys.exit(f"ERROR: model not found at {model_src}")

    # ── 1. Stage model artifact into the Docker build context ─────────────────
    print("\n=== Staging model artifacts ===")
    artifacts_dir.mkdir(exist_ok=True)
    shutil.copy2(model_src, artifacts_dir / "model.bst")
    print(f"  Copied model.bst → {artifacts_dir / 'model.bst'}")

    # ── 2. Create Artifact Registry repo (idempotent) ─────────────────────────
    print("\n=== Artifact Registry ===")
    run(
        f"gcloud artifacts repositories create {args.repo} "
        f"--repository-format=docker --location={REGION} "
        f"--project={PROJECT} --quiet 2>/dev/null || true"
    )
    run(f"gcloud auth configure-docker {REGION}-docker.pkg.dev --quiet")

    # ── 3. Build & push Docker image ──────────────────────────────────────────
    print("\n=== Building Docker image ===")
    run(f"docker build --platform=linux/amd64 -t {IMAGE} {fraud_api_dir}")

    print("\n=== Pushing image to Artifact Registry ===")
    run(f"docker push {IMAGE}")

    # ── 4. Upload model resource & deploy to endpoint ─────────────────────────
    print("\n=== Uploading model to Vertex AI ===")
    try:
        from google.cloud import aiplatform
    except ImportError:
        sys.exit("ERROR: run 'pip install google-cloud-aiplatform' first")

    aiplatform.init(project=PROJECT, location=REGION)

    model = aiplatform.Model.upload(
        display_name=args.model_name,
        serving_container_image_uri=IMAGE,
        serving_container_ports=[8080],
        serving_container_health_route="/health",
        serving_container_predict_route="/predict",
        serving_container_environment_variables={
            "MODEL_PATH": "artifacts/model.bst",
            "THRESHOLD": "0.236",
        },
    )
    print(f"  Model: {model.resource_name}")

    print("\n=== Creating endpoint ===")
    endpoint = aiplatform.Endpoint.create(display_name=args.endpoint_name)

    print("\n=== Deploying model (takes ~10 minutes) ===")
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=args.model_name,
        machine_type=args.machine_type,
        min_replica_count=1,
        max_replica_count=3,
        traffic_percentage=100,
    )

    endpoint_id = endpoint.name  # short numeric ID

    # ── 5. Save connection config for the notebook ────────────────────────────
    config = {
        "project_id": PROJECT,
        "region": REGION,
        "endpoint_id": endpoint_id,
        "endpoint_resource_name": endpoint.resource_name,
    }
    config_path = root / "vertex_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    print("\n" + "=" * 60)
    print("Deployment complete!")
    print(f"  Endpoint ID:   {endpoint_id}")
    print(f"  Config saved → {config_path}")
    print("\nOpen vertex_inference.ipynb to start scoring transactions.")


if __name__ == "__main__":
    main()
