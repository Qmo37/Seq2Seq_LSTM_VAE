"""
Data preprocessing module for OULAD dataset.
Handles loading, cleaning, feature engineering, and sequence creation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split


def load_oulad_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load OULAD dataset CSV files.

    Args:
        data_path: Path to directory containing CSV files

    Returns:
        Tuple of (studentInfo, studentVle, studentAssessment) DataFrames
    """
    student_info = pd.read_csv(f"{data_path}/studentInfo.csv")
    student_vle = pd.read_csv(f"{data_path}/studentVle.csv")
    student_assessment = pd.read_csv(f"{data_path}/studentAssessment.csv")

    return student_info, student_vle, student_assessment


def aggregate_weekly_features(
    student_info: pd.DataFrame,
    student_vle: pd.DataFrame,
    student_assessment: pd.DataFrame,
    max_weeks: int = 30,
) -> pd.DataFrame:
    """
    Aggregate student activity data into weekly features.

    Features per week:
    - clicks: Total weekly click count
    - submit_cnt: Number of submissions
    - has_submit: Binary indicator of submission
    - avg_score_sofar: Cumulative average score
    - clicks_diff1: First-order difference of clicks

    Args:
        student_info: Student demographic information
        student_vle: Virtual Learning Environment interaction logs
        student_assessment: Assessment submission and scores
        max_weeks: Maximum number of weeks to consider

    Returns:
        DataFrame with student_id, week, and all features
    """
    # Convert date to week number
    student_vle["week"] = student_vle["date"] // 7
    student_assessment["week"] = student_assessment["date_submitted"] // 7

    # Aggregate clicks per student per week
    clicks_df = (
        student_vle.groupby(["id_student", "week"])["sum_click"].sum().reset_index()
    )
    clicks_df.columns = ["id_student", "week", "clicks"]

    # Aggregate submissions per student per week
    submit_df = (
        student_assessment.groupby(["id_student", "week"])
        .agg({"score": ["count", "mean"]})
        .reset_index()
    )
    submit_df.columns = ["id_student", "week", "submit_cnt", "avg_score"]

    # Get unique student IDs
    student_ids = student_info["id_student"].unique()

    # Create complete week grid for each student
    all_weeks = []
    for student_id in student_ids:
        for week in range(max_weeks):
            all_weeks.append({"id_student": student_id, "week": week})

    df = pd.DataFrame(all_weeks)

    # Merge features
    df = df.merge(clicks_df, on=["id_student", "week"], how="left")
    df = df.merge(submit_df, on=["id_student", "week"], how="left")

    # Fill missing values
    df["clicks"] = df["clicks"].fillna(0)
    df["submit_cnt"] = df["submit_cnt"].fillna(0)
    df["avg_score"] = df["avg_score"].fillna(0)

    # Sort by student and week
    df = df.sort_values(["id_student", "week"]).reset_index(drop=True)

    # Create derived features
    df["has_submit"] = (df["submit_cnt"] > 0).astype(int)

    # Calculate cumulative average score per student
    df["avg_score_sofar"] = df.groupby("id_student")["avg_score"].transform(
        lambda x: x.expanding().mean()
    )

    # Calculate first-order difference of clicks
    df["clicks_diff1"] = df.groupby("id_student")["clicks"].diff().fillna(0)

    return df


def create_sequences(
    df: pd.DataFrame,
    input_weeks: int = 4,
    output_weeks: int = 2,
    feature_cols: list = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create input-output sequences for sequence-to-sequence learning.

    Args:
        df: DataFrame with weekly features per student
        input_weeks: Number of past weeks to use as input
        output_weeks: Number of future weeks to predict
        feature_cols: List of feature column names to use

    Returns:
        Tuple of (X, y, student_ids) arrays
        - X: shape (n_samples, input_weeks, n_features)
        - y: shape (n_samples, output_weeks, 1) - predicting clicks only
        - student_ids: shape (n_samples,) - for train/test split
    """
    if feature_cols is None:
        feature_cols = ["clicks", "has_submit", "avg_score_sofar", "clicks_diff1"]

    X_list = []
    y_list = []
    student_id_list = []

    # Group by student
    for student_id, group in df.groupby("id_student"):
        group = group.sort_values("week").reset_index(drop=True)

        # Skip students with insufficient data
        if len(group) < input_weeks + output_weeks:
            continue

        # Create sliding windows
        for i in range(len(group) - input_weeks - output_weeks + 1):
            # Input: past input_weeks
            X_window = group.iloc[i : i + input_weeks][feature_cols].values

            # Output: future output_weeks (clicks only)
            y_window = group.iloc[i + input_weeks : i + input_weeks + output_weeks][
                ["clicks"]
            ].values

            X_list.append(X_window)
            y_list.append(y_window)
            student_id_list.append(student_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    student_ids = np.array(student_id_list)

    return X, y, student_ids


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, Tuple[float, float]],
]:
    """
    Normalize features using training set statistics.

    Args:
        X_train, X_val, X_test: Input sequences
        y_train, y_val, y_test: Output sequences

    Returns:
        Normalized arrays and normalization statistics dictionary
    """
    # Calculate statistics from training set
    X_mean = X_train.mean(axis=(0, 1), keepdims=True)
    X_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8

    # Normalize
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std

    # Store normalization stats
    norm_stats = {"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std}

    return (
        X_train_norm,
        X_val_norm,
        X_test_norm,
        y_train_norm,
        y_val_norm,
        y_test_norm,
        norm_stats,
    )


def load_and_preprocess_data(
    data_path: str,
    input_weeks: int = 4,
    output_weeks: int = 2,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> Dict:
    """
    Complete data loading and preprocessing pipeline.

    Args:
        data_path: Path to directory containing OULAD CSV files
        input_weeks: Number of input weeks
        output_weeks: Number of output weeks
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing train/val/test splits and normalization stats
    """
    # Load data
    print("Loading OULAD data...")
    student_info, student_vle, student_assessment = load_oulad_data(data_path)

    # Aggregate weekly features
    print("Aggregating weekly features...")
    df = aggregate_weekly_features(student_info, student_vle, student_assessment)

    # Create sequences
    print("Creating sequences...")
    X, y, student_ids = create_sequences(df, input_weeks, output_weeks)

    print(f"Total sequences: {len(X)}")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    # Split by student ID to avoid data leakage
    unique_students = np.unique(student_ids)
    train_students, test_students = train_test_split(
        unique_students, test_size=test_size, random_state=random_seed
    )
    train_students, val_students = train_test_split(
        train_students, test_size=val_size, random_state=random_seed
    )

    # Create masks
    train_mask = np.isin(student_ids, train_students)
    val_mask = np.isin(student_ids, val_students)
    test_mask = np.isin(student_ids, test_students)

    # Split data
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Normalize
    print("Normalizing features...")
    X_train, X_val, X_test, y_train, y_val, y_test, norm_stats = normalize_features(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "norm_stats": norm_stats,
    }
