# ============================================================
# DECISION TREE USING ENTROPY
# Use Case: Vehicle Service Risk Prediction
# ============================================================

# Install libraries if required:
# pip install pandas scikit-learn matplotlib

import math
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
    export_text
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# STEP 1: CREATE SAMPLE DATASET
# ============================================================

data = {
    "engine_alerts": [
        7, 6, 5, 8, 4,
        2, 1, 0, 2, 1,
        9, 3, 6, 1, 5,
        0, 7, 2, 8, 3
    ],

    "mileage_km": [
        85000, 78000, 69000, 92000, 65000,
        58000, 30000, 22000, 35000, 27000,
        98000, 45000, 74000, 25000, 70000,
        18000, 88000, 32000, 95000, 40000
    ],

    "brake_wear_percent": [
        90, 85, 80, 95, 75,
        65, 30, 20, 40, 25,
        96, 55, 82, 28, 78,
        18, 91, 38, 94, 50
    ],

    "service_gap_months": [
        18, 16, 14, 22, 13,
        10, 5, 3, 6, 4,
        24, 8, 15, 4, 14,
        2, 20, 6, 23, 9
    ],

    "service_risk": [
        "High Risk", "High Risk", "High Risk", "High Risk", "High Risk",
        "High Risk", "Low Risk", "Low Risk", "Low Risk", "Low Risk",
        "High Risk", "Low Risk", "High Risk", "Low Risk", "High Risk",
        "Low Risk", "High Risk", "Low Risk", "High Risk", "Low Risk"
    ]
}

df = pd.DataFrame(data)

print("=" * 60)
print("VEHICLE SERVICE RISK DATASET")
print("=" * 60)
print(df)

print("\nDataset shape:", df.shape)

print("\nTarget class distribution:")
print(df["service_risk"].value_counts())


# ============================================================
# STEP 2: MANUALLY CALCULATE PARENT ENTROPY
# ============================================================

def calculate_entropy(labels):
    """
    Calculates entropy using:

    H(S) = -sum(p * log2(p))
    """

    total_records = len(labels)
    class_counts = labels.value_counts()

    entropy = 0

    for class_name, count in class_counts.items():
        probability = count / total_records

        entropy = entropy - (
            probability * math.log2(probability)
        )

        print(
            f"Class: {class_name}, "
            f"Count: {count}, "
            f"Probability: {probability:.3f}"
        )

    return entropy


print("\n" + "=" * 60)
print("PARENT ENTROPY CALCULATION")
print("=" * 60)

parent_entropy = calculate_entropy(df["service_risk"])

print(f"\nParent Entropy: {parent_entropy:.4f}")


# ============================================================
# STEP 3: SEPARATE INPUT FEATURES AND TARGET
# ============================================================

X = df[
    [
        "engine_alerts",
        "mileage_km",
        "brake_wear_percent",
        "service_gap_months"
    ]
]

y = df["service_risk"]

print("\n" + "=" * 60)
print("INPUT FEATURES")
print("=" * 60)
print(X.head())

print("\nTarget:")
print(y.head())


# ============================================================
# STEP 4: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print("Total records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# STEP 5: CREATE DECISION TREE MODEL
# ============================================================

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    min_samples_split=2,
    random_state=42
)


# ============================================================
# STEP 6: TRAIN THE MODEL
# ============================================================

model.fit(X_train, y_train)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)


# ============================================================
# STEP 7: MAKE PREDICTIONS
# ============================================================

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)


# ============================================================
# STEP 8: CALCULATE TRAINING AND TESTING ACCURACY
# ============================================================

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print("\nTraining Accuracy:", round(train_accuracy, 4))
print("Testing Accuracy:", round(test_accuracy, 4))


# ============================================================
# STEP 9: DISPLAY ACTUAL AND PREDICTED VALUES
# ============================================================

results = X_test.copy()

results["Actual Risk"] = y_test.values
results["Predicted Risk"] = test_predictions

print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED RESULTS")
print("=" * 60)
print(results)


# ============================================================
# STEP 10: CONFUSION MATRIX
# ============================================================

labels = ["High Risk", "Low Risk"]

matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=labels
)

confusion_df = pd.DataFrame(
    matrix,
    index=[
        "Actual High Risk",
        "Actual Low Risk"
    ],
    columns=[
        "Predicted High Risk",
        "Predicted Low Risk"
    ]
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print(confusion_df)


# ============================================================
# STEP 11: CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)


# ============================================================
# STEP 12: DISPLAY DECISION TREE RULES
# ============================================================

tree_rules = export_text(
    model,
    feature_names=list(X.columns)
)

print("\n" + "=" * 60)
print("DECISION TREE RULES")
print("=" * 60)
print(tree_rules)


# ============================================================
# STEP 13: FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)
print(feature_importance)


# ============================================================
# STEP 14: PREDICT ONE NEW VEHICLE
# ============================================================

new_vehicle = pd.DataFrame({
    "engine_alerts": [6],
    "mileage_km": [72000],
    "brake_wear_percent": [82],
    "service_gap_months": [15]
})

new_prediction = model.predict(new_vehicle)

new_probability = model.predict_proba(new_vehicle)

print("\n" + "=" * 60)
print("NEW VEHICLE PREDICTION")
print("=" * 60)

print(new_vehicle)

print(
    "\nPredicted Service Risk:",
    new_prediction[0]
)

print("\nClass order:", model.classes_)
print(
    "Prediction probabilities:",
    new_probability[0]
)


# ============================================================
# STEP 15: PREDICT MULTIPLE NEW VEHICLES
# ============================================================

new_vehicles = pd.DataFrame({
    "engine_alerts": [1, 8, 3, 6],
    "mileage_km": [25000, 95000, 42000, 76000],
    "brake_wear_percent": [25, 96, 52, 84],
    "service_gap_months": [4, 24, 8, 16]
})

multiple_predictions = model.predict(new_vehicles)

new_vehicles["Predicted Service Risk"] = multiple_predictions

print("\n" + "=" * 60)
print("MULTIPLE VEHICLE PREDICTIONS")
print("=" * 60)
print(new_vehicles)


# ============================================================
# STEP 16: VISUALIZE THE DECISION TREE
# ============================================================

plt.figure(figsize=(18, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title(
    "Vehicle Service Risk Prediction\n"
    "Decision Tree Using Entropy"
)

plt.tight_layout()
plt.show()
