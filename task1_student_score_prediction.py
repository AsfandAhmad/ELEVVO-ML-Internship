"""
Task 1 – Student Score Prediction (Elevvo ML Internship)
Author: <your name>
Usage:
  python task1_student_score_prediction.py --data data/student_performance_factors.csv

What this script does
1) Loads the dataset (CSV)
2) Basic cleaning + EDA plots
3) Baseline Linear Regression using only study hours
4) Polynomial Regression (degree 2 & 3) comparison
5) Optional: Multi‑feature Linear Regression with preprocessing (scaler + one‑hot encoder)
6) Saves figures and a trained model under ./artifacts

Notes:
- Adjust DATA_PATH or pass via --data if your CSV has a different location.
- The script tries to auto‑detect the target (exam score) and the study‑hours column.
- If auto‑detection fails, edit TARGET_CANDIDATES and HOURS_CANDIDATES below.
"""

import argparse
import os
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

# ----------------------
# Configuration
# ----------------------
ARTIFACT_DIR = Path("artifacts")
FIG_DIR = ARTIFACT_DIR / "figures"
MODEL_DIR = ARTIFACT_DIR / "models"
ARTIFACT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Common column name guesses (case/underscore insensitive)
TARGET_CANDIDATES = [
    "exam_score", "score", "final_score", "g3", "marks", "performanceindex", "mathscore",
]
HOURS_CANDIDATES = [
    "hours_studied", "study_hours", "studyhours", "hours", "reading_hours",
]

RANDOM_STATE = 42


# ----------------------
# Helpers
# ----------------------

def std_name(s: str) -> str:
    """Standardize column names: lowercase and remove non-alphanumerics."""
    return re.sub(r"[^a-z0-9]", "_", s.strip().lower())


def find_first_match(columns, candidates):
    """Return the first column from `columns` that matches any name in `candidates` after standardization."""
    std_cols = {std_name(c): c for c in columns}
    for cand in candidates:
        if cand in std_cols:
            return std_cols[cand]
    # Also try fuzzy contains (e.g., 'hours' in 'hours_studied')
    for cand in candidates:
        for sc, original in std_cols.items():
            if cand in sc:
                return original
    return None


def metric_dict(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


# ----------------------
# Core
# ----------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standardize column names for internal checks, but keep original for display
    df.columns = [c.strip() for c in df.columns]
    return df


def basic_eda(df: pd.DataFrame, target_col: str, hours_col: str):
    print("\n=== Basic Info ===")
    print(df.info())
    print("\n=== Describe (numeric) ===")
    print(df.select_dtypes(include=[np.number]).describe())


    # Histogram of target
    plt.figure(figsize=(6, 4))
    df[target_col].hist(bins=30)
    plt.title(f"Distribution of {target_col}")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"hist_{std_name(target_col)}.png", dpi=150)
    plt.close()

    # Histogram of hours studied
    plt.figure(figsize=(6, 4))
    df[hours_col].hist(bins=30)
    plt.title(f"Distribution of {hours_col}")
    plt.xlabel(hours_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"hist_{std_name(hours_col)}.png", dpi=150)
    plt.close()

    # Scatter: hours vs target
    plt.figure(figsize=(6, 4))
    plt.scatter(df[hours_col], df[target_col], alpha=0.6)
    plt.title(f"{hours_col} vs {target_col}")
    plt.xlabel(hours_col)
    plt.ylabel(target_col)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"scatter_{std_name(hours_col)}_vs_{std_name(target_col)}.png", dpi=150)
    plt.close()


def train_baseline_hours(df: pd.DataFrame, target_col: str, hours_col: str):
    data = df[[hours_col, target_col]].dropna()
    X = data[[hours_col]]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = metric_dict(y_test, y_pred)

    # Plot predictions vs. actuals
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.7)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, linestyle="--")
    plt.title("Baseline Linear Regression: Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "baseline_actual_vs_pred.png", dpi=150)
    plt.close()

    # Residuals plot
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.title("Baseline Residuals vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "baseline_residuals.png", dpi=150)
    plt.close()

    print("\n=== Baseline (Hours -> Score) Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Save model
    joblib.dump(model, MODEL_DIR / "baseline_linear_hours.joblib")

    return model, metrics


def train_polynomial(df: pd.DataFrame, target_col: str, hours_col: str, degree: int = 2):
    data = df[[hours_col, target_col]].dropna()
    X = data[[hours_col]]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("linreg", LinearRegression()),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = metric_dict(y_test, y_pred)

    print(f"\n=== Polynomial Regression (degree={degree}) Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    joblib.dump(pipe, MODEL_DIR / f"poly_deg{degree}_hours.joblib")
    return pipe, metrics


def train_multifeature(df: pd.DataFrame, target_col: str, hours_col: str):
    # Identify features
    feature_df = df.drop(columns=[target_col])

    # Separate columns by dtype
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Basic cleaning: drop rows with missing target; impute features in pipeline
    data = df.dropna(subset=[target_col]).copy()

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
            (
                "cat",
                Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline([
        ("pre", preprocessor),
        ("linreg", LinearRegression()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = metric_dict(y_test, y_pred)

    print("\n=== Multi‑feature Linear Regression Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Plot predictions vs. actuals
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.7)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, linestyle="--")
    plt.title("Multi‑feature Linear Regression: Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "multifeature_actual_vs_pred.png", dpi=150)
    plt.close()

    joblib.dump(model, MODEL_DIR / "multifeature_linear.joblib")

    # Try to extract feature importances (coefficients) after one‑hot
    try:
        # Get feature names from preprocessor
        pre = model.named_steps["pre"]
        num_features = numeric_cols
        cat_features = list(pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols))
        all_features = num_features + cat_features

        lin = model.named_steps["linreg"]
        coefs = lin.coef_.ravel()
        coef_df = pd.DataFrame({"feature": all_features, "coef": coefs})
        coef_df["abs_coef"] = coef_df["coef"].abs()
        coef_df.sort_values("abs_coef", ascending=False).head(15).to_csv(
            ARTIFACT_DIR / "top_coefficients.csv", index=False
        )
        print("Saved top coefficients to artifacts/top_coefficients.csv")
    except Exception as e:
        print(f"Could not extract coefficients: {e}")

    return model, metrics


def main(data_path: str):
    df = load_data(data_path)

    # Try to detect columns
    std_cols_map = {std_name(c): c for c in df.columns}

    target_col = find_first_match(df.columns, TARGET_CANDIDATES)
    hours_col = find_first_match(df.columns, HOURS_CANDIDATES)

    if target_col is None or hours_col is None:
        print("\n[!] Auto‑detect failed.")
        print("Columns in your CSV:")
        print(list(df.columns))
        raise SystemExit(
            "Please edit TARGET_CANDIDATES and HOURS_CANDIDATES at the top of the script to match your column names."
        )

    print(f"Detected target column: {target_col}")
    print(f"Detected study-hours column: {hours_col}")

    # Basic EDA
    basic_eda(df, target_col, hours_col)

    # Baseline
    _, base_metrics = train_baseline_hours(df, target_col, hours_col)

    # Polynomial (degree 2 & 3)
    _, poly2_metrics = train_polynomial(df, target_col, hours_col, degree=2)
    _, poly3_metrics = train_polynomial(df, target_col, hours_col, degree=3)

    # Multi‑feature (bonus)
    _, multi_metrics = train_multifeature(df, target_col, hours_col)

    # Compare results
    results = pd.DataFrame([
        {"Model": "Baseline Linear (Hours)", **base_metrics},
        {"Model": "Polynomial (deg=2)", **poly2_metrics},
        {"Model": "Polynomial (deg=3)", **poly3_metrics},
        {"Model": "Multi‑feature Linear", **multi_metrics},
    ])
    print("\n=== Model Comparison ===")
    print(results)
    results.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/student_performance_factors.csv", help="Path to CSV dataset")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"CSV not found at: {args.data}. Place your dataset there or pass --data path.")

    main(args.data)
