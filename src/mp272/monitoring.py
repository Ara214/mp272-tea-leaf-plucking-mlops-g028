
import pandas as pd
import numpy as np
from pathlib import Path

from src.mp272.cleaning import clean_data


# Six numeric input features
NUMERIC_FEATURES = [
    "rain_mm_week",
    "hours_worked",
    "temp_c",
    "worker_exp_yr",
    "slope_deg",
    "bush_age_yr"
]


def calculate_psi(train, later, bins=10):
    """
    Calculate Population Stability Index (PSI)
    using 10 equal-width bins based on training data.
    """

    train = pd.to_numeric(train, errors="coerce").dropna()
    later = pd.to_numeric(later, errors="coerce").dropna()

    if len(train) == 0 or len(later) == 0:
        raise ValueError("Feature contains no valid numeric values.")

    # Create 10 bins using training data range
    minimum = train.min()
    maximum = train.max()

    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5

    edges = np.linspace(minimum, maximum, bins + 1)

    # Include values outside the training range
    edges[0] = -np.inf
    edges[-1] = np.inf

    train_counts = np.histogram(train, bins=edges)[0]
    later_counts = np.histogram(later, bins=edges)[0]

    # Convert counts into proportions
    train_pct = train_counts / len(train)
    later_pct = later_counts / len(later)

    # Avoid division by zero and log(0)
    epsilon = 0.0001

    train_pct = np.clip(train_pct, epsilon, None)
    later_pct = np.clip(later_pct, epsilon, None)

    # PSI formula
    psi = np.sum(
        (later_pct - train_pct) *
        np.log(later_pct / train_pct)
    )

    return float(psi)


def calculate_drift(train_df, later_df):
    """Calculate PSI for all six numeric features."""

    results = []

    for feature in NUMERIC_FEATURES:

        if feature not in train_df.columns:
            raise ValueError(f"{feature} missing from train data")

        if feature not in later_df.columns:
            raise ValueError(f"{feature} missing from later data")

        psi_value = calculate_psi(
            train_df[feature],
            later_df[feature],
            bins=10
        )

        results.append({
            "feature": feature,
            "PSI": psi_value
        })

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        by="PSI",
        ascending=False
    ).reset_index(drop=True)

    most_drifted = result_df.iloc[0]["feature"]

    return result_df, most_drifted


def run_monitoring():

    root = Path(__file__).resolve().parents[2]

    # Read original datasets
    train = pd.read_csv(root / "data/raw/train.csv")
    later = pd.read_csv(root / "data/raw/later.csv")

    # Apply existing Stage 2 cleaning logic
    train_cleaned = clean_data(train.copy())
    later_cleaned = clean_data(later.copy())

    # Calculate PSI
    results, most_drifted = calculate_drift(
        train_cleaned,
        later_cleaned
    )

    print("\n===== DATA DRIFT MONITORING =====")
    print(results.to_string(index=False))

    print("\nMost drifted feature:", most_drifted)


if __name__ == "__main__":
    run_monitoring()
