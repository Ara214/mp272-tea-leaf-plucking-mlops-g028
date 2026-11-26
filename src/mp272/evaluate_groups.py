"""Model performance monitoring: MAE of the v1 model per section on later.csv.

Steps
-----
1. Load the existing v1 model (``models/model_v1.joblib`` by default).
2. Load ``data/raw/later.csv`` and clean it with the same ``clean()`` used
   in training, so the model sees inputs in the same units.
3. Predict and calculate MAE separately for each value of ``section``
   (Upper Garden, Lower Garden, River Side), plus overall.
4. Identify the worst-served group (highest MAE).
5. Save the results to ``reports/group_mae.json`` and print the lines to
   copy into ``reports/findings.yaml`` (monitoring.mae_by_group and
   monitoring.worst_served_group).

Usage (from the repo root)
--------------------------
    python3 src/mp272/evaluate_groups.py
    python3 src/mp272/evaluate_groups.py --model models/model.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

try:  # run as part of the package (pytest, python -m mp272.evaluate_groups)
    from mp272.train import FEATURES, clean
except ImportError:  # run as a script: python3 src/mp272/evaluate_groups.py
    from train import FEATURES, clean

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "model_v1.joblib"
LATER_CSV = ROOT / "data" / "raw" / "later.csv"
REPORT_PATH = ROOT / "reports" / "group_mae.json"

TARGET = "kg_per_worker"
GROUP_COL = "section"
GROUPS = ["Upper Garden", "Lower Garden", "River Side"]


def mae_by_group(model, df: pd.DataFrame, target: str = TARGET,
                 group_col: str = GROUP_COL) -> dict[str, float]:
    """Return {group: MAE} for each group, in the order of GROUPS."""
    df = df.dropna(subset=[target]).copy()
    df["_pred"] = model.predict(df[FEATURES])

    results = {}
    for group in GROUPS:
        rows = df[df[group_col] == group]
        if rows.empty:
            print(f"Warning: no rows for group '{group}' in the data")
            continue
        results[group] = round(
            float(mean_absolute_error(rows[target], rows["_pred"])), 4)

    others = sorted(set(df[group_col].dropna()) - set(GROUPS))
    if others:
        print(f"Warning: unexpected {group_col} values ignored: {others}")
    return results


def worst_group(maes: dict[str, float]) -> str:
    """The group with the highest MAE."""
    return max(maes, key=maes.get)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--data", type=Path, default=LATER_CSV)
    parser.add_argument("--target", default=TARGET)
    args = parser.parse_args()

    # 1. Load model
    model = joblib.load(args.model)

    # 2. Load and clean later.csv
    df = clean(pd.read_csv(args.data))
    missing = [c for c in FEATURES + [args.target, GROUP_COL]
               if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in {args.data.name}: {missing}")

    # 3. MAE per group, plus overall
    maes = mae_by_group(model, df, args.target)
    scored = df.dropna(subset=[args.target])
    overall = float(mean_absolute_error(scored[args.target],
                                        model.predict(scored[FEATURES])))
    counts = scored[GROUP_COL].value_counts().to_dict()

    # 4. Worst-served group
    worst = worst_group(maes)

    print(f"Model: {args.model.name}  |  data: {args.data.name}  "
          f"({len(scored)} rows)\n")
    print(f"{'Group':<15}{'Rows':>6}{'MAE':>10}")
    for group, mae in maes.items():
        flag = "  <- worst" if group == worst else ""
        print(f"{group:<15}{counts.get(group, 0):>6}{mae:>10.4f}{flag}")
    print(f"{'Overall':<15}{len(scored):>6}{overall:>10.4f}\n")

    # 5. Save
    report = {
        "model": args.model.name,
        "data": args.data.name,
        "mae_by_group": maes,
        "rows_by_group": {g: int(counts.get(g, 0)) for g in maes},
        "mae_overall": round(overall, 4),
        "worst_served_group": worst,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Saved -> {REPORT_PATH.relative_to(ROOT)}\n")

    print("Copy into reports/findings.yaml under monitoring:")
    print("  mae_by_group:")
    for group, mae in maes.items():
        print(f"    '{group}': {mae}")
    print(f"  worst_served_group: {worst}")


if __name__ == "__main__":
    main()
