
import hashlib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import mean_absolute_error

from src.mp272.config import DATA_CONFIG, MLFLOW_CONFIG
from src.mp272.train import clean, build_pipeline, FEATURES
from src.mp272.tracking import log_experiment, register_model


def main():
    # 1. Load the original training data and later data
    train_path = DATA_CONFIG["train_path"]
    later_path = DATA_CONFIG["later_path"]

    original_train = pd.read_csv(train_path)
    later = pd.read_csv(later_path)

    # 2. Calculate the original training dataset hash
    with open(train_path, "rb") as file:
        data_sha256 = hashlib.sha256(file.read()).hexdigest()

    # 3. Clean both datasets using the original cleaning function
    original_train = clean(original_train)
    later = clean(later)

    target = "kg_per_worker"

    # 4. Sort later data chronologically and split 70:30
    date_col = next(
        (c for c in later.columns if "date" in c.lower()),
        None
    )

    if date_col is None:
        raise ValueError("Date column not found in later.csv")

    later = later.assign(
        _date=pd.to_datetime(later[date_col])
    ).sort_values("_date")

    cut = int(len(later) * 0.70)

    later_train = later.iloc[:cut]
    later_test = later.iloc[cut:]

    print("Original training rows:", len(original_train))
    print("Later training rows:", len(later_train))
    print("Final evaluation rows:", len(later_test))

    # 5. Combine original training data and first 70% of later data
    combined_train = pd.concat(
        [original_train, later_train],
        ignore_index=True
    )

    X_train = combined_train[FEATURES]
    y_train = combined_train[target]

    X_test = later_test[FEATURES]
    y_test = later_test[target]

    # 6. Load the existing V1 champion
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])

    model_name = MLFLOW_CONFIG["registered_model_name"]

    v1_uri = f"models:/{model_name}@champion"
    v1_model = mlflow.sklearn.load_model(v1_uri)

    # 7. Evaluate V1 on the same final 30%
    v1_predictions = v1_model.predict(X_test)
    v1_mae = mean_absolute_error(y_test, v1_predictions)

    print("\nV1 MAE on final 30%:", round(v1_mae, 6))

    # 8. Train V2 using combined training data
    v2_model = build_pipeline()
    v2_model.fit(X_train, y_train)

    # 9. Evaluate V2 on exactly the same final 30%
    v2_predictions = v2_model.predict(X_test)
    v2_mae = mean_absolute_error(y_test, v2_predictions)

    print("V2 MAE on final 30%:", round(v2_mae, 6))

    # 10. Calculate improvement and promotion eligibility
    improvement = (v1_mae - v2_mae) / v1_mae

    promote = improvement >= 0.08

    print("\nImprovement:", round(improvement * 100, 2), "%")
    print("Promotion eligible:", promote)

    # 11. Log the V2 experiment
    params = {
        "model": "LinearRegression",
        "model_stage": "V2",
        "target": target,
        "original_train_rows": len(original_train),
        "later_training_rows": len(later_train),
        "evaluation_rows": len(later_test),
        "later_split": "70:30 chronological",
        "features": ",".join(FEATURES),
    }

    metrics = {
        "mae_v1_later_holdout": v1_mae,
        "mae_v2_later_holdout": v2_mae,
        "mae_improvement_fraction": improvement,
    }

    result = log_experiment(
        model=v2_model,
        params=params,
        metrics=metrics,
        data_sha256=data_sha256,
    )

    print("\nV2 MLflow run:", result["run_id"])

    # 12. Register V2 as Version 2
    registration = register_model(
        model_uri=result["model_uri"],
        run_id=result["run_id"],
        promote=promote,
    )

    print("\nV2 registration:", registration)

    if promote:
        print("V2 promoted to champion!")
    else:
        print("V1 remains champion because V2 did not meet the 8% rule.")


if __name__ == "__main__":
    main()
