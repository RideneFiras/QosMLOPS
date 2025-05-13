import pandas as pd
import numpy as np
import joblib

# Load expected features order from saved training set
processed_data_path = "Models/processedx_data.pkl"
expected_features = joblib.load(processed_data_path)[0].columns.tolist()


def preprocess_single_row(row: dict) -> pd.DataFrame:
    """
    Preprocess a single row (from raw test.csv or user input) to match model requirements.
    Must return a DataFrame with the exact same columns as during training.
    """
    df = pd.DataFrame([row])

    # Feature: SCell_Active
    scell_cols = [
        "SCell_Cell_Identity",
        "SCell_RSRP_max",
        "SCell_RSRQ_max",
        "SCell_RSSI_max",
        "SCell_SNR_max",
        "SCell_Downlink_Num_RBs",
        "SCell_Downlink_Average_MCS",
        "SCell_Downlink_bandwidth_MHz",
    ]
    absence_activite_scell = df[scell_cols].isnull().any(axis=1)
    df["SCell_Active"] = np.where(absence_activite_scell, 0, 1)

    # Fill values based on SCell_Active
    mask_active = df["SCell_Active"] == 1
    for col in scell_cols:
        if col in df.columns:
            df.loc[mask_active, col] = df[col].fillna(df[col].mean())

    mask_inactive = df["SCell_Active"] == 0
    zero_cols = scell_cols + ["SCell_freq_MHz"]
    for col in zero_cols:
        if col in df.columns:
            df.loc[mask_inactive & df[col].isnull(), col] = 0

    # Drop irrelevant columns
    cols_to_drop = [
        "visibility",
        "windSpeed",
        "SCell_freq_MHz",
        "PCell_freq_MHz",
        "uvIndex",
        "COG",
        "precipIntensity",
        "Pressure",
        "id",
        "PCell_Cell_Identity",
        "SCell_Cell_Identity",
    ]
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    # Handle timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["hour"] = df["timestamp"].dt.hour
        df["day"] = df["timestamp"].dt.day
        df["weekday"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Feature Engineering
    if "PCell_SNR_max" in df.columns and "PCell_RSRP_max" in df.columns:
        df["RSRP_SNR_ratio"] = df["PCell_SNR_max"] / (df["PCell_RSRP_max"].abs() + 1e-3)
    if "PCell_SNR_max" in df.columns and "PCell_RSRQ_max" in df.columns:
        df["RSRQ_SNR_ratio"] = df["PCell_SNR_max"] / (df["PCell_RSRQ_max"].abs() + 1e-3)
    if (
        "PCell_Downlink_Num_RBs" in df.columns
        and "PCell_Downlink_Average_MCS" in df.columns
    ):
        df["estimated_utilization"] = (
            df["PCell_Downlink_Num_RBs"] * df["PCell_Downlink_Average_MCS"]
        )

    # Keep only numeric features and reorder
    df = df.select_dtypes(include=[np.number])
    df = df.reindex(columns=expected_features)

    return df
