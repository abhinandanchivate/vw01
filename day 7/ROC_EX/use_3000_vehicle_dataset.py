"""
Use the 3,000-row vehicle-failure dataset for:
1. Data loading and inspection
2. Train/test split
3. L1 and L2 Logistic Regression
4. Confusion matrix, classification metrics and ROC-AUC
5. ROC curve
6. Five-fold stratified cross-validation
7. Optional threshold adjustment

Keep this script in the same directory as:
    vehicle_failure_ml_dataset.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------
# 1. LOAD THE DATASET
# -------------------------------------------------------------------
DATA_FILE = Path(__file__).with_name("vehicle_failure_ml_dataset.csv")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Place vehicle_failure_ml_dataset.csv in the same folder as this script."
    )

data = pd.read_csv(DATA_FILE)

print("\nDATASET INFORMATION")
print("-" * 60)
print("Shape:", data.shape)
print("\nFirst five rows:")
print(data.head())
print("\nTarget distribution:")
print(data["failure_next_30_days"].value_counts())
print("\nTarget percentages:")
print(data["failure_next_30_days"].value_counts(normalize=True).mul(100).round(2))
print("\nMissing values:")
print(data.isna().sum())


# -------------------------------------------------------------------
# 2. SELECT FEATURES AND TARGET
# -------------------------------------------------------------------
# Do not use:
# - vehicle_id: identifier, not a useful failure signal
# - inspection_date: excluded in the basic random-split example
# - failure_probability_hidden: used to generate the synthetic target;
#   including it would cause target leakage
columns_to_exclude = [
    "vehicle_id",
    "inspection_date",
    "failure_probability_hidden",
    "failure_next_30_days",
]

X = data.drop(columns=columns_to_exclude, errors="ignore")
y = data["failure_next_30_days"]

print("\nMODEL FEATURES")
print("-" * 60)
print(X.columns.tolist())
print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)


# -------------------------------------------------------------------
# 3. TRAIN/TEST SPLIT
# -------------------------------------------------------------------
# stratify=y preserves the failure ratio in training and testing data.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("\nTRAIN/TEST SPLIT")
print("-" * 60)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training target counts:")
print(y_train.value_counts())
print("Testing target counts:")
print(y_test.value_counts())


# -------------------------------------------------------------------
# 4. CREATE L1 AND L2 MODELS
# -------------------------------------------------------------------
# class_weight='balanced' gives more importance to the minority failure class.
models = {
    "L1 Logistic Regression": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    l1_ratio=1.0,
                    solver="liblinear",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    ),
    "L2 Logistic Regression": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    l1_ratio=0.0,
                    solver="liblinear",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    ),
}


# -------------------------------------------------------------------
# 5. TRAIN AND EVALUATE THE MODELS
# -------------------------------------------------------------------
model_results = []
roc_results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    failure_probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, failure_probabilities)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    model_results.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        }
    )

    fpr, tpr, thresholds = roc_curve(y_test, failure_probabilities)
    roc_results[model_name] = (fpr, tpr, auc)

    print(f"\n{model_name.upper()}")
    print("-" * 60)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["Normal", "Failure"]))
    print(f"ROC-AUC: {auc:.4f}")

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Normal", "Failure"],
        values_format="d",
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(DATA_FILE.with_name(f"{safe_name}_confusion_matrix.png"), dpi=150)
    plt.close()

results_df = pd.DataFrame(model_results)
print("\nMODEL COMPARISON")
print("-" * 60)
print(results_df.round(4).to_string(index=False))
results_df.to_csv(DATA_FILE.with_name("model_comparison_results.csv"), index=False)


# -------------------------------------------------------------------
# 6. PLOT ROC CURVES
# -------------------------------------------------------------------
plt.figure(figsize=(8, 6))
for model_name, (fpr, tpr, auc) in roc_results.items():
    plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", label="Random model (AUC=0.500)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Recall")
plt.title("ROC Curve - Vehicle Failure Prediction")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(DATA_FILE.with_name("roc_curve_3000_rows.png"), dpi=150)
plt.close()


# -------------------------------------------------------------------
# 7. STRATIFIED FIVE-FOLD CROSS-VALIDATION
# -------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

cv_summary = []

for model_name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    row = {"model": model_name}
    for metric in scoring:
        metric_values = scores[f"test_{metric}"]
        row[f"mean_{metric}"] = metric_values.mean()
        row[f"std_{metric}"] = metric_values.std()

    cv_summary.append(row)

cv_results_df = pd.DataFrame(cv_summary)
print("\nFIVE-FOLD STRATIFIED CROSS-VALIDATION")
print("-" * 60)
print(cv_results_df.round(4).to_string(index=False))
cv_results_df.to_csv(DATA_FILE.with_name("cross_validation_summary.csv"), index=False)


# -------------------------------------------------------------------
# 8. OPTIONAL: LOWER THE THRESHOLD TO CATCH MORE FAILURES
# -------------------------------------------------------------------
# The default threshold is 0.50. A lower threshold usually increases recall
# but may also create more false alarms.
best_model = models["L2 Logistic Regression"]
best_model.fit(X_train, y_train)
probabilities = best_model.predict_proba(X_test)[:, 1]

threshold_rows = []
for threshold in [0.50, 0.40, 0.30, 0.20]:
    threshold_predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, threshold_predictions).ravel()

    threshold_rows.append(
        {
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, threshold_predictions),
            "precision": precision_score(y_test, threshold_predictions, zero_division=0),
            "recall": recall_score(y_test, threshold_predictions, zero_division=0),
            "f1_score": f1_score(y_test, threshold_predictions, zero_division=0),
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "true_negative": tn,
        }
    )

threshold_df = pd.DataFrame(threshold_rows)
print("\nTHRESHOLD COMPARISON - L2 MODEL")
print("-" * 60)
print(threshold_df.round(4).to_string(index=False))
threshold_df.to_csv(DATA_FILE.with_name("threshold_comparison.csv"), index=False)

print("\nFiles created:")
print("- model_comparison_results.csv")
print("- cross_validation_summary.csv")
print("- threshold_comparison.csv")
print("- roc_curve_3000_rows.png")
print("- L1 and L2 confusion-matrix images")
