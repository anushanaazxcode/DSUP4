# Task 04 — End-to-End Machine Learning Capstone

## Project: Breast Cancer Classification

### Overview
This capstone takes a prediction problem from a raw CSV dataset to a trained model, test predictions, evaluation metrics, and an error-analysis file. It predicts whether a tumour record is **malignant** or **benign** using the public Breast Cancer Wisconsin Diagnostic dataset.

The project uses Logistic Regression as a simple, interpretable starting classifier. It is an educational machine-learning project, not a medical diagnostic tool.

## What this project includes
- A runnable pipeline from raw CSV data to predictions.
- An 80/20 stratified train-test split performed before fitting preprocessing or model steps.
- Missing-value imputation and feature scaling inside a scikit-learn Pipeline to avoid test-data leakage.
- Logistic Regression model training.
- Accuracy, precision, recall, F1-score, and confusion matrix.
- Error analysis: incorrect test predictions are saved to `outputs/misclassified_cases.csv`.
- A short discussion of assumptions and limitations.

## Dataset
- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository; distributed in scikit-learn.
- **Local file:** `data/breast_cancer_raw.csv`
- **Rows:** 569
- **Feature columns:** 30
- **Target:** `target` (0 = malignant, 1 = benign)

The CSV is included, so the pipeline does not need to download data when it runs.

## Project structure
```text
End_to_End_ML_Capstone/
├── data/
│   └── breast_cancer_raw.csv
├── outputs/
│   ├── classification_report.txt
│   ├── metrics.json
│   └── misclassified_cases.csv
├── pipeline.py
├── requirements.txt
└── README.md
```

## How to run
1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the pipeline:
   ```bash
   python pipeline.py
   ```
5. Review the generated files in the `outputs/` folder.

You can also provide another compatible CSV:
```bash
python pipeline.py --data data/your_data.csv --output outputs
```
The input CSV must contain numeric feature columns and a `target` column encoded as 0 and 1.

## Evaluation results
The saved results in `outputs/metrics.json` were generated using:
- Test size: 20%
- Random state: 42
- Stratified split
- Logistic Regression with median imputation and standard scaling

| Metric | Held-out test result |
|---|---:|
| Accuracy | 0.9825 |
| Precision (benign class) | 0.9861 |
| Recall (benign class) | 0.9861 |
| F1-score (benign class) | 0.9861 |

The confusion matrix label order is malignant (0), then benign (1). See `outputs/classification_report.txt` for class-wise results.

## Error analysis
The pipeline writes every incorrectly classified test row to `outputs/misclassified_cases.csv`. The file includes:
- Original row ID
- Actual and predicted labels
- Predicted probability for the benign class
- Original feature values

Reviewing these rows helps identify which observations the model struggles to classify. The number and nature of errors should be considered alongside recall and the confusion matrix; accuracy alone can hide important error types.

## Assumptions
- The target is correctly encoded: 0 = malignant and 1 = benign.
- Each row represents one independent observation.
- The numeric input features are measured consistently with the training data.
- A stratified random hold-out split is a reasonable estimate of performance on similar unseen records.
- Median imputation and standard scaling are suitable baseline preprocessing choices.

## Limitations
- This dataset is relatively small and may not represent all populations, equipment, or clinical settings.
- A single train-test split can produce a variable estimate; cross-validation and external validation would provide stronger evidence.
- Logistic Regression may not capture complex non-linear relationships.
- The model is not calibrated or approved for clinical use.
- False negatives and false positives have different real-world consequences; model use would require domain-expert review and a carefully chosen decision threshold.
- Test performance only describes this dataset and split; it does not guarantee performance on future data.

## Reproducibility
The split and model use `random_state=42`. The pipeline saves metrics, test predictions, a classification report, and misclassified cases each time it runs.
