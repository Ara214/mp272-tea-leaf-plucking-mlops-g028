"""Train the baseline regression model (v1) for MP272 tea-leaf plucking.

Steps
-----
1. Load ``data/raw/train.csv``.
2. Clean it, applying the issues recorded in ``reports/findings.yaml``:
   * hours_worked recorded in minutes for ``Upper Garden``  -> divide by 60
   * worker_exp_yr sentinel value 9999                     -> missing
   * rain_mm_week stuck sensor 2026-03-25 .. 2026-04-15    -> missing
3. Define features (X) and target (y).
4. Train / validation split (80 / 20, seeded).
5. Fit a baseline pipeline: median imputation + scaling + one-hot(section)
   followed by ordinary least-squares ``LinearRegression``.
6. Report validation MAE (next to a predict-the-mean baseline for context).
7. Save the fitted pipeline to ``models/model_v1.joblib`` and the metrics to
   ``reports/metrics_v1.json``. If MLflow is installed the run is also logged
   and the run id is printed (needed for ``tracking.best_run_id``).

The saved model expects *cleaned* input. When scoring later.csv or
holdout_features.csv, run the rows through the same function first:

    from mp272.train import clean        # with PYTHONPATH=src
    model.predict(clean(df))

Usage (from the repo root)
--------------------------
    dvc pull                      # fetch data/raw/*.csv
    python src/mp272/train.py
    python src/mp272/train.py --split time --target <target_column>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
TRAIN_CSV = RAW_DIR / "train.csv"
HOLDOUT_CSV = RAW_DIR / "holdout_features.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model_v1.joblib"
METRICS_PATH = ROOT / "reports" / "metrics_v1.json"

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "worker_exp_yr",
    "rain_mm_week",
    "slope_deg",
    "temp_c",
    "bush_age_yr",
    "hours_worked",
]
CATEGORICAL_FEATURES = ["section"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Set this to the target column name to skip auto-detection.
TARGET: str | None = None

RANDOM_STATE = 42
VALID_SIZE = 0.20

# --------------------------------------------------------------------------
# Cleaning rules (keep in sync with reports/findings.yaml)
# --------------------------------------------------------------------------
UNIT_MIXUP = {"column": "hours_worked", "group_col": "section",
              "group_value": "Upper Garden", "factor": 60}
SENTINEL = {"column": "worker_exp_yr", "value": 9999}
STUCK_SENSOR = {"column": "rain_mm_week",
                "start_date": "2026-03-25", "end_date": "2026-04-15"}


def find_date_column(df: pd.DataFrame) -> str | None:
    """Return the name of the date column, if there is one."""
    candidates = [c for c in df.columns if "date" in c.lower()]
    return candidates[0] if candidates else None


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the data issues listed in findings.yaml. Returns a copy."""
    df = df.copy()

    # 1. Unit mix-up: Upper Garden logged hours_worked in minutes.
    mask = df[UNIT_MIXUP["group_col"]] == UNIT_MIXUP["group_value"]
    df.loc[mask, UNIT_MIXUP["column"]] = (
        df.loc[mask, UNIT_MIXUP["column"]] / UNIT_MIXUP["factor"]
    )

    # 2. Sentinel value -> missing (imputed later inside the pipeline).
    df[SENTINEL["column"]] = df[SENTINEL["column"]].replace(
        SENTINEL["value"], np.nan
    )

    # 3. Stuck sensor: readings in this date window are not trustworthy.
    date_col = find_date_column(df)
    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        stuck = dates.between(pd.Timestamp(STUCK_SENSOR["start_date"]),
                              pd.Timestamp(STUCK_SENSOR["end_date"]))
        df.loc[stuck, STUCK_SENSOR["column"]] = np.nan

    return df


def infer_target(train: pd.DataFrame) -> str:
    """The target is the column in train.csv that holdout_features.csv lacks."""
    if TARGET is not None:
        return TARGET

    if HOLDOUT_CSV.exists():
        holdout_cols = set(pd.read_csv(HOLDOUT_CSV, nrows=0).columns)
        missing = [c for c in train.columns if c not in holdout_cols]
        if len(missing) == 1:
            return missing[0]

    # Fallback: the only numeric column that is not a feature or an id.
    numeric = train.select_dtypes("number").columns
    leftover = [c for c in numeric
                if c not in FEATURES and not c.lower().endswith("id")]
    if len(leftover) == 1:
        return leftover[0]

    raise ValueError(
        "Could not infer the target column; pass --target <name>. "
        f"Columns in train.csv: {list(train.columns)}"
    )


def build_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", LinearRegression())])


def split(df: pd.DataFrame, target: str, how: str):
    """Return X_train, X_valid, y_train, y_valid."""
    if how == "time":
        date_col = find_date_column(df)
        if date_col is None:
            raise ValueError("--split time needs a date column in train.csv")
        df = df.assign(_d=pd.to_datetime(df[date_col])).sort_values("_d")
        cut = int(len(df) * (1 - VALID_SIZE))
        tr, va = df.iloc[:cut], df.iloc[cut:]
        return tr[FEATURES], va[FEATURES], tr[target], va[target]

    return train_test_split(df[FEATURES], df[target],
                            test_size=VALID_SIZE, random_state=RANDOM_STATE)


def log_to_mlflow(pipe, params: dict, metrics: dict,
                  experiment: str) -> str | None:
    """Log the run to MLflow if it is installed; return the run id."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("MLflow not installed - skipping experiment tracking.")
        return None

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="v1_baseline_linear") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        try:
            # MLflow 3.x saves sklearn models with skops, which rejects
            # numpy.dtype (used by OneHotEncoder) unless it is trusted.
            mlflow.sklearn.log_model(pipe, name="model",
                                     skops_trusted_types=["numpy.dtype"])
        except TypeError:
            # Older MLflow: no skops / no `name` argument.
            mlflow.sklearn.log_model(pipe, artifact_path="model")
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=TRAIN_CSV)
    parser.add_argument("--target", default=None,
                        help="target column (auto-detected if omitted)")
    parser.add_argument("--split", choices=["random", "time"], default="random",
                        help="random 80/20 (default) or last 20%% by date")
    parser.add_argument("--experiment", default="mp272-tea-leaf-G028")
    args = parser.parse_args()

    global TARGET
    if args.target:
        TARGET = args.target

    # 1-2. Load and clean
    raw = pd.read_csv(args.data)
    df = clean(raw)

    # 3. Features and target
    target = infer_target(raw)
    missing_cols = [c for c in FEATURES + [target] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in {args.data.name}: {missing_cols}")
    df = df.dropna(subset=[target])
    print(f"Rows: {len(df)}  |  target: {target}  |  features: {FEATURES}")

    # 4. Split
    X_train, X_valid, y_train, y_valid = split(df, target, args.split)
    print(f"Train: {len(X_train)} rows  |  Validation: {len(X_valid)} rows "
          f"({args.split} split)")

    # 5. Fit
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # 6. Evaluate
    mae = mean_absolute_error(y_valid, pipe.predict(X_valid))
    mae_mean = mean_absolute_error(y_valid,
                                   np.full(len(y_valid), y_train.mean()))
    print(f"Validation MAE (v1 linear):      {mae:.4f}")
    print(f"Validation MAE (predict mean):   {mae_mean:.4f}")

    # 7. Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    params = {
        "model": "LinearRegression",
        "target": target,
        "features": ",".join(FEATURES),
        "split": args.split,
        "valid_size": VALID_SIZE,
        "random_state": RANDOM_STATE,
    }
    metrics = {"mae_valid": float(mae), "mae_valid_mean_baseline": float(mae_mean)}
    run_id = log_to_mlflow(pipe, params, metrics, args.experiment)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(
        {**params, **metrics, "n_train": len(X_train),
         "n_valid": len(X_valid), "mlflow_run_id": run_id},
        indent=2))

    print(f"Saved model   -> {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved metrics -> {METRICS_PATH.relative_to(ROOT)}")
    if run_id:
        print(f"MLflow run id: {run_id}  (use for tracking.best_run_id)")


if __name__ == "__main__":
    main()
