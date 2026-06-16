"""
Vehicle Failure Prediction - Confusion Matrix Case Study
========================================================

Goal:
Predict whether a vehicle will experience a brake-related failure
within the next 30 days.

Class labels:
0 = Normal vehicle
1 = Risky vehicle / likely failure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# -------------------------------------------------------
# STEP 1: Generate a synthetic fleet dataset
# -------------------------------------------------------
RNG = np.random.default_rng(42)
TOTAL_VEHICLES = 1000

data = pd.DataFrame({
    "vehicle_id": [f"VH-{i:04d}" for i in range(1, TOTAL_VEHICLES + 1)],
    "mileage_km": RNG.integers(5000, 150000, TOTAL_VEHICLES),
    "brake_wear_percent": np.clip(
        RNG.normal(55, 20, TOTAL_VEHICLES), 0, 100
    ),
    "brake_temperature_c": np.clip(
        RNG.normal(85, 25, TOTAL_VEHICLES), 30, 180
    ),
    "vibration_level": np.clip(
        RNG.normal(4.5, 2, TOTAL_VEHICLES), 0, 10
    ),
    "days_since_service": RNG.integers(5, 730, TOTAL_VEHICLES),
    "warning_light_count": RNG.poisson(0.7, TOTAL_VEHICLES),
})

# A synthetic risk score is used only to create the training label.
# In a real project, the label comes from historical failure records.
risk_score = (
    0.000015 * data["mileage_km"]
    + 0.06 * data["brake_wear_percent"]
    + 0.025 * data["brake_temperature_c"]
    + 0.45 * data["vibration_level"]
    + 0.002 * data["days_since_service"]
    + 0.8 * data["warning_light_count"]
    + RNG.normal(0, 1.2, TOTAL_VEHICLES)
)

# Mark the highest-risk 5% as vehicles that fail within 30 days.
failure_cutoff = np.quantile(risk_score, 0.95)
data["failure_next_30_days"] = (risk_score >= failure_cutoff).astype(int)

print("Class distribution:")
print(data["failure_next_30_days"].value_counts().sort_index())
print()


# -------------------------------------------------------
# STEP 2: Evaluate the misleading baseline
# -------------------------------------------------------
y_actual = data["failure_next_30_days"]
y_baseline = np.zeros(TOTAL_VEHICLES, dtype=int)

print("BASELINE: Predict every vehicle as normal")
print("Confusion matrix:")
print(confusion_matrix(y_actual, y_baseline))
print("Accuracy:", accuracy_score(y_actual, y_baseline))
print("Failure recall:", recall_score(y_actual, y_baseline, zero_division=0))
print()

# Baseline result:
# [[950   0]
#  [ 50   0]]
# Accuracy = 95%, but recall for failures = 0%.


# -------------------------------------------------------
# STEP 3: Prepare X and y
# -------------------------------------------------------
feature_columns = [
    "mileage_km",
    "brake_wear_percent",
    "brake_temperature_c",
    "vibration_level",
    "days_since_service",
    "warning_light_count",
]

X = data[feature_columns]
y = data["failure_next_30_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# -------------------------------------------------------
# STEP 4: Train a class-aware model
# -------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_train, y_train)


# -------------------------------------------------------
# STEP 5: Evaluate at the default threshold of 0.50
# -------------------------------------------------------
failure_probability = model.predict_proba(X_test)[:, 1]
default_predictions = (failure_probability >= 0.50).astype(int)


def evaluate(name, y_true, y_pred):
    """Print confusion-matrix metrics for a prediction set."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(name)
    print("Confusion matrix:")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1-score : {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return cm


default_cm = evaluate(
    "MODEL AT DEFAULT THRESHOLD 0.50",
    y_test,
    default_predictions,
)


# -------------------------------------------------------
# STEP 6: Use a safety-oriented threshold
# -------------------------------------------------------
# Lowering the threshold flags more vehicles for inspection.
# This usually increases recall but also increases false positives.
safety_threshold = 0.15
safety_predictions = (failure_probability >= safety_threshold).astype(int)

safety_cm = evaluate(
    f"MODEL AT SAFETY THRESHOLD {safety_threshold}",
    y_test,
    safety_predictions,
)


# -------------------------------------------------------
# STEP 7: Visualize the safety-threshold confusion matrix
# -------------------------------------------------------
display = ConfusionMatrixDisplay(
    confusion_matrix=safety_cm,
    display_labels=["Normal", "Failure"],
)
display.plot(values_format="d")
plt.title("Vehicle Failure Confusion Matrix")
plt.tight_layout()
plt.show()


# -------------------------------------------------------
# STEP 8: Predict risk for new vehicles
# -------------------------------------------------------
new_vehicles = pd.DataFrame([
    {
        "mileage_km": 22000,
        "brake_wear_percent": 25,
        "brake_temperature_c": 65,
        "vibration_level": 2.0,
        "days_since_service": 60,
        "warning_light_count": 0,
    },
    {
        "mileage_km": 132000,
        "brake_wear_percent": 94,
        "brake_temperature_c": 145,
        "vibration_level": 8.7,
        "days_since_service": 610,
        "warning_light_count": 4,
    },
])

new_probabilities = model.predict_proba(new_vehicles)[:, 1]
new_predictions = (new_probabilities >= safety_threshold).astype(int)

for index, (probability, prediction) in enumerate(
    zip(new_probabilities, new_predictions), start=1
):
    status = "Inspect immediately" if prediction == 1 else "Continue monitoring"
    print(
        f"New vehicle {index}: failure probability={probability:.3f} -> {status}"
    )
