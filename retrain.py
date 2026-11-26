"""Retrain and conditionally promote model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


FEATURE_COLUMNS = [
    "worker_exp_yr",
    "rain_mm_week",
    "slope_deg",
    "temp_c",
    "bush_age_yr",
    "hours_worked",
    "flush",
    "section",
]
TARGET_COLUMN = "kg_per_worker"
DATE_COLUMN = "date"
PROMOTION_IMPROVEMENT = 0.08


def chronological_split(later: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    required = {DATE_COLUMN}
    missing = required.difference(later.columns)
    if missing:
        raise ValueError(f"later data is missing required columns: {sorted(missing)}")
    if len(later) < 2:
        raise ValueError("later data must contain at least two rows")

    ordered = later.copy()
    ordered[DATE_COLUMN] = pd.to_datetime(ordered[DATE_COLUMN], errors="raise")
    ordered = ordered.sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)
    split_at = int(len(ordered) * 0.70)
    split_at = min(max(split_at, 1), len(ordered) - 1)
    return ordered.iloc[:split_at].copy(), ordered.iloc[split_at:].copy()


def _validate_frame(frame: pd.DataFrame, label: str) -> None:
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError(f"{label} contains missing {TARGET_COLUMN} values")


def _registered_version(client: Any, model_name: str, run_id: str) -> str:

    versions = client.search_model_versions(f"name='{model_name}'")
    matches = [version for version in versions if version.run_id == run_id]
    if not matches:
        raise RuntimeError(
            f"MLflow registered the model but no version was found for run {run_id}"
        )
    return str(max(matches, key=lambda item: int(item.version)).version)


def run_retraining(
    *,
    clean_train_path: Path,
    clean_later_path: Path,
    original_train_path: Path,
    tracking_uri: str = "sqlite:///mlflow.db",
    registered_model_name: str,
    experiment_name: str = "MP272-Retraining",
) -> dict[str, Any]:


    try:
        import mlflow
        import mlflow.sklearn
        from mlflow import MlflowClient
        from sklearn.base import clone
        from sklearn.metrics import mean_absolute_error
    except ImportError as exc:
        raise RuntimeError(
        ) from exc

    for path in (clean_train_path, clean_later_path, original_train_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required data file does not exist: {path}")

    clean_train = pd.read_csv(clean_train_path)
    clean_later = pd.read_csv(clean_later_path)
    _validate_frame(clean_train, "cleaned train data")
    _validate_frame(clean_later, "cleaned later data")

    later_train, later_check = chronological_split(clean_later)
    train_all = pd.concat([clean_train, later_train], ignore_index=True)
    _validate_frame(train_all, "combined v2 training data")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)
    v1 = mlflow.sklearn.load_model(f"models:/{registered_model_name}@champion")

    x_check = later_check[FEATURE_COLUMNS]
    y_check = later_check[TARGET_COLUMN]
    mae_v1 = float(mean_absolute_error(y_check, v1.predict(x_check)))

    v2 = clone(v1)
    v2.fit(train_all[FEATURE_COLUMNS], train_all[TARGET_COLUMN])
    mae_v2 = float(mean_absolute_error(y_check, v2.predict(x_check)))
    promote = mae_v2 <= mae_v1 * (1.0 - PROMOTION_IMPROVEMENT)

    data_sha256 = hashlib.sha256(original_train_path.read_bytes()).hexdigest()
    with mlflow.start_run(run_name="v2_chronological_retrain") as active_run:
        mlflow.log_params(
            {
                "later_train_fraction": 0.70,
                "promotion_improvement": PROMOTION_IMPROVEMENT,
                "training_rows_clean_train": len(clean_train),
                "training_rows_later_first_70_percent": len(later_train),
                "evaluation_rows_later_last_30_percent": len(later_check),
                "data_sha256": data_sha256,
            }
        )
        mlflow.log_metrics({"mae_v1_on_check": mae_v1, "mae_v2_on_check": mae_v2})
        model_info = mlflow.sklearn.log_model(
            sk_model=v2,
            name="model",
            registered_model_name=registered_model_name,
            serialization_format="cloudpickle",
        )

    run_id = active_run.info.run_id
    version = getattr(model_info, "registered_model_version", None)
    if version is None:
        version = _registered_version(client, registered_model_name, run_id)
    else:
        version = str(version)

    if promote:
        client.set_registered_model_alias(
            registered_model_name, "champion", version
        )

    return {
        "registered_model_name": registered_model_name,
        "v2_version": version,
        "mae_v1_on_check": mae_v1,
        "mae_v2_on_check": mae_v2,
        "required_v2_mae_max": mae_v1 * (1.0 - PROMOTION_IMPROVEMENT),
        "decision": "promote" if promote else "keep",
        "champion_updated": promote,
        "mlflow_run_id": run_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-train", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--clean-later", type=Path, default=Path("data/processed/later.csv"))
    parser.add_argument("--original-train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument(
        "--registered-model-name",
        required=True,
        help="The Stage 2 MLflow registered model name containing the v1 champion",
    )
    parser.add_argument("--experiment-name", default="MP272-Retraining")
    args = parser.parse_args()

    result = run_retraining(
        clean_train_path=args.clean_train,
        clean_later_path=args.clean_later,
        original_train_path=args.original_train,
        tracking_uri=args.tracking_uri,
        registered_model_name=args.registered_model_name,
        experiment_name=args.experiment_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
