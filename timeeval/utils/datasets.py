from pathlib import Path

import numpy as np
import pandas as pd


def extract_labels(df: pd.DataFrame) -> np.ndarray:
    labels: np.ndarray = df.values[:, -1].astype(np.float64)
    return labels


def extract_features(df: pd.DataFrame) -> np.ndarray:
    features: np.ndarray = df.values[:, 1:-1]
    return features


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def load_labels_only(path: Path) -> np.ndarray:
    labels: np.ndarray = pd.read_csv(path, usecols=["is_anomaly"])[
        "is_anomaly"
    ].values.astype(np.float64)
    return labels

def load_dimensional_labels_only(path: Path) -> np.ndarray:
    full_df = pd.read_csv(path)
    total_columns = full_df.columns.tolist()
    aggregated_is_anomaly_column = 'is_anomaly' if 'is_anomaly' in total_columns else None
    assert aggregated_is_anomaly_column is not None
    dimensional_is_anomaly_columns = [f for f in total_columns if f.startsWith('is_anomaly_')]
    # if len(dimensional_is_anomaly_columns) > 0:
    dimensional_labels: np.ndarray = full_df[dimensional_is_anomaly_columns].values.astype(np.float64)
    return dimensional_labels
