
import hashlib
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.mp272.config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    MLFLOW_CONFIG,
)
from src.mp272.train import clean, build_pipeline, FEATURES
from src.mp272.tracking import log_experiment


def main():
    # 1. Read the original training dataset
    train_path = DATA_CONFIG["train_path"]
    df = pd.read_csv(train_path)

    # 2. Calculate SHA-256 of the original, unmodified CSV
    with open(train_path, "rb") as file:
        data_sha256 = hashlib.sha256(file.read()).hexdigest()

    # 3. Clean the dataset using the original project function
    df = clean(df)

    target = "kg_per_worker"

    if target not in df.columns:
        raise ValueError(f"Target column {target} not found.")

    # 4. Define three experiment configurations
    experiments = [
        {"name": "random_seed_42", "split": "random", "seed": 42},
        {"name": "random_seed_7", "split": "random", "seed": 7},
        {"name": "time_split", "split": "time", "seed": 42},
    ]

    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])

    results = []

    for experiment in experiments:
        # Use the configured validation size
        valid_size = MODEL_CONFIG["validation_size"]

        # Sort chronologically for time-based validation
        if experiment["split"] == "time":
            date_col = next(
                (c for c in df.columns if "date" in c.lower()),
                None,
            )
            if date_col is None:
                raise ValueError("No date column found.")

            ordered = df.assign(
                _date=pd.to_datetime(df[date_col])
            ).sort_values("_date")

            cut = int(len(ordered) * (1 - valid_size))
            train_df = ordered.iloc[:cut]
            valid_df = ordered.iloc[cut:]

            X_train = train_df[FEATURES]
            X_valid = valid_df[FEATURES]
            y_train = train_df[target]
            y_valid = valid_df[target]

        else:
            from sklearn.model_selection import train_test_split

            X_train, X_valid, y_train, y_valid = train_test_split(
                df[FEATURES],
                df[target],
                test_size=valid_size,
                random_state=experiment["seed"],
            )

        # Train the original Linear Regression pipeline
        model = build_pipeline()
        model.fit(X_train, y_train)

        # Calculate validation MAE
        predictions = model.predict(X_valid)
        mae = mean_absolute_error(y_valid, predictions)

        params = {
            "model": "LinearRegression",
            "target": target,
            "split": experiment["split"],
            "random_state": experiment["seed"],
            "validation_size": valid_size,
            "features": ",".join(FEATURES),
        }

        metrics = {"mae_valid": mae}

        # Log the experiment without registering every model
        result = log_experiment(
            model=model,
            params=params,
            metrics=metrics,
            data_sha256=data_sha256,
        )

        results.append({
            "experiment": experiment["name"],
            "mae": mae,
            "run_id": result["run_id"],
            "model_uri": result["model_uri"],
        })

        print(f"\nExperiment: {experiment['name']}")
        print(f"Validation MAE: {mae:.4f}")
        print(f"Run ID: {result['run_id']}")

    print("\n========== EXPERIMENT SUMMARY ==========")

    for result in results:
        print(
            f"{result['experiment']} | "
            f"MAE: {result['mae']:.4f} | "
            f"Run ID: {result['run_id']}"
        )

    best = min(results, key=lambda item: item["mae"])

    print("\nBest experiment:", best["experiment"])
    print("Best MAE:", best["mae"])
    print("Best Run ID:", best["run_id"])


if __name__ == "__main__":
    main()
