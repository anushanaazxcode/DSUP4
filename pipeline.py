"""
End-to-end machine learning pipeline:
raw CSV -> train/test split -> preprocessing/model pipeline -> evaluation
-> misclassification analysis -> saved predictions and metrics.
"""

from pathlib import Path
import json
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "breast_cancer_raw.csv"
DEFAULT_OUTPUT = ROOT / "outputs"
TARGET_COLUMN = "target"
TARGET_NAMES = {0: "malignant", 1: "benign"}


def run_pipeline(data_path=DEFAULT_DATA, output_dir=DEFAULT_OUTPUT):
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read raw data.
    df = pd.read_csv(data_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' is missing.")
    if df.empty:
        raise ValueError("The input dataset is empty.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Split before fitting preprocessing or model components.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # All learned preprocessing is inside the pipeline to prevent leakage.
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix_labels_order": ["malignant (0)", "benign (1)"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "positive_class": "benign (1)",
    }

    # Save all incorrect predictions for manual error analysis.
    errors = X_test.copy()
    errors.insert(0, "row_id", X_test.index)
    errors["actual_label"] = y_test
    errors["predicted_label"] = predictions
    errors["actual_name"] = y_test.map(TARGET_NAMES)
    errors["predicted_name"] = pd.Series(predictions, index=y_test.index).map(TARGET_NAMES)
    errors["predicted_benign_probability"] = probabilities
    errors = errors.loc[errors["actual_label"] != errors["predicted_label"]]
    errors.to_csv(output_dir / "misclassified_cases.csv", index=False)

    # Save predictions for every test row to support review.
    test_predictions = pd.DataFrame({
        "row_id": X_test.index,
        "actual_label": y_test.to_numpy(),
        "predicted_label": predictions,
        "predicted_benign_probability": probabilities,
    })
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "classification_report.txt").write_text(
        classification_report(
            y_test, predictions,
            labels=[0, 1],
            target_names=["malignant", "benign"],
            zero_division=0,
        ),
        encoding="utf-8",
    )

    print("Pipeline completed.")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test precision: {metrics['precision']:.4f}")
    print(f"Test recall: {metrics['recall']:.4f}")
    print(f"Test F1-score: {metrics['f1_score']:.4f}")
    print(f"Misclassified cases saved: {len(errors)}")
    print(f"Output folder: {output_dir}")
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the end-to-end breast cancer ML pipeline.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to input CSV file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Directory for generated outputs.")
    args = parser.parse_args()
    run_pipeline(args.data, args.output)
