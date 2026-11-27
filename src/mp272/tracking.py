
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.mp272.config import MLFLOW_CONFIG

# MLflow configuration loaded from params.yaml
TRACKING_URI = MLFLOW_CONFIG["tracking_uri"]
EXPERIMENT_NAME = MLFLOW_CONFIG["experiment_name"]
REGISTERED_MODEL_NAME = MLFLOW_CONFIG["registered_model_name"]


def log_experiment(model, params, metrics, data_sha256):
    """
    Log model, parameters, metrics and dataset hash.
    Does not register the model.
    """

    if not data_sha256 or data_sha256 == "dummy-test-hash":
        raise ValueError("A valid dataset SHA-256 hash is required.")

    if not any("mae" in key.lower() for key in metrics):
        raise ValueError("At least one MAE metric is required.")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("data_sha256", data_sha256)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle"
        )

        run_id = run.info.run_id

    return {
        "run_id": run_id,
        "model_uri": model_info.model_uri
    }


def register_model(model_uri, run_id, promote=False):
    """
    Register a selected model and optionally assign champion alias.
    """

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    if promote:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="champion",
            version=model_version.version
        )

    return {
        "run_id": run_id,
        "model_version": model_version.version,
        "promoted": promote
    }
