
import pandas as pd


def clean_data(df):
    """
    Clean MP272 dataset:
    1. Replace sentinel value 9999 with NaN.
    2. Convert Upper Garden hours from minutes to hours.
    3. Correct stuck rain sensor readings using interpolation.
    """

    # Create a copy so original data remains unchanged
    df = df.copy()

    # Convert date column into datetime
    df["date"] = pd.to_datetime(df["date"])

    # 1. Handle sentinel values
    df["worker_exp_yr"] = df["worker_exp_yr"].replace(9999, float("nan"))

    # 2. Fix Upper Garden hours unit mix-up
    upper_garden = df["section"].astype(str).str.strip().str.lower() == "upper garden"

    df.loc[upper_garden, "hours_worked"] = (
        df.loc[upper_garden, "hours_worked"] / 60
    )

    # 3. Fix stuck rain sensor
    start_date = pd.Timestamp("2026-03-25")
    end_date = pd.Timestamp("2026-04-15")

    # Get one rain reading per date
    daily_rain = (
        df.groupby("date")["rain_mm_week"]
        .first()
        .sort_index()
    )

    # Mark stuck sensor readings as missing
    stuck_period = (
        (daily_rain.index >= start_date) &
        (daily_rain.index <= end_date)
    )

    daily_rain.loc[stuck_period] = float("nan")

    # Interpolate missing readings using surrounding dates
    daily_rain = daily_rain.interpolate(
        method="time",
        limit_area="inside"
    )

    # Update rain values in original dataframe
    df["rain_mm_week"] = df["date"].map(daily_rain)

    return df
