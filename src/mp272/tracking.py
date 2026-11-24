
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# MLflow configuration
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "MP272_Tea_Leaf_Plucking"
REGISTERED_MODEL_NAME = "MP272_TeaLeaf_Model"


def log_experiment(model, params, metrics, data_sha256):
    """
    Log a trained model, its parameters, metrics,
    and dataset hash to MLflow.
    """

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:

        # Log model parameters
        mlflow.log_params(params)

        # Log evaluation metrics
        mlflow.log_metrics(metrics)

        # Log training dataset hash
        mlflow.log_param("data_sha256", data_sha256)

        # Log and register the trained sklearn model
        model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name=REGISTERED_MODEL_NAME,
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

        run_id = run.info.run_id

    # Retrieve the registered model version
    client = MlflowClient()

    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    registered_version = next(
        version for version in versions
        if version.run_id == run_id
    )

    # Assign champion alias
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "champion",
        registered_version.version
    )

    return {
        "run_id": run_id,
        "model_version": registered_version.version,
        "model_uri": model_info.model_uri
    }
