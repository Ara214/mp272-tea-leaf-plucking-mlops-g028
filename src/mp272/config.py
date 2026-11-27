from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "params.yaml"


def load_config():
    with open(PARAMS_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


CONFIG = load_config()

DATA_CONFIG = CONFIG["data"]
MODEL_CONFIG = CONFIG["model"]
MLFLOW_CONFIG = CONFIG["mlflow"]
