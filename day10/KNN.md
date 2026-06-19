# KNN-Based Machine Learning Case Study

## Vehicle Service-Priority Classification

### 1. Business problem

An automobile service company receives thousands of vehicle-service requests. Manually checking every vehicle makes it difficult to identify vehicles that require urgent attention.

The company wants a machine-learning system that classifies each vehicle into one of three service-priority categories:

| Priority   | Meaning                                     | Recommended action             |
| ---------- | ------------------------------------------- | ------------------------------ |
| **Low**    | Vehicle condition appears normal            | Schedule routine service       |
| **Medium** | Some warning indicators are present         | Schedule service within 7 days |
| **High**   | Vehicle has several serious risk indicators | Inspect immediately            |

The company decides to use **K-Nearest Neighbors classification** because vehicles with similar operating conditions often have similar service requirements.

---

# 2. Problem statement

Build a KNN classification model that predicts a vehicle’s service priority using:

* Mileage
* Months since last service
* Number of engine alerts
* Brake-wear percentage
* Battery voltage
* Customer complaints
* Vehicle age
* Driving pattern

The model should predict:

```text
Low
Medium
High
```

---

# 3. Dataset structure

| Feature               | Description               | Example |
| --------------------- | ------------------------- | ------: |
| `mileage_km`          | Total vehicle mileage     |  82,000 |
| `service_gap_months`  | Months since last service |      15 |
| `engine_alerts`       | Number of engine warnings |       5 |
| `brake_wear_percent`  | Percentage of brake wear  |      78 |
| `battery_voltage`     | Current battery voltage   |    11.3 |
| `customer_complaints` | Number of complaints      |       4 |
| `vehicle_age_years`   | Age of the vehicle        |       8 |
| `driving_pattern`     | Typical driving condition |    City |
| `service_priority`    | Target class              |    High |

---

# 4. Why KNN can be used

KNN predicts a new vehicle by finding the most similar vehicles in the training dataset.

Suppose a new vehicle has:

```text
Mileage             : 95,000 km
Service gap         : 18 months
Engine alerts       : 6
Brake wear          : 85%
Battery voltage     : 10.9 V
Customer complaints : 4
```

KNN searches for vehicles with similar values.

```text
New vehicle
     ↓
Find the K nearest historical vehicles
     ↓
Check their service-priority classes
     ↓
Select the majority class
```

For example, with `K = 5`:

| Nearest vehicle | Priority |
| --------------: | -------- |
|       Vehicle 1 | High     |
|       Vehicle 2 | High     |
|       Vehicle 3 | Medium   |
|       Vehicle 4 | High     |
|       Vehicle 5 | High     |

Result:

```text
High    = 4 neighbours
Medium  = 1 neighbour
Low     = 0 neighbours
```

Therefore:

```text
Predicted priority = High
```

---

# 5. Case-study workflow

```text
Vehicle service data
        ↓
Separate features and target
        ↓
Train-test split
        ↓
Scale numerical features
        ↓
Encode driving pattern
        ↓
Train KNN model
        ↓
Use GridSearchCV to find the best K
        ↓
Evaluate the model
        ↓
Predict service priority for new vehicles
```

---

# 6. Complete Python code

```python
# =========================================================
# KNN CASE STUDY
# Vehicle Service-Priority Classification
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ---------------------------------------------------------
# 1. CREATE A SYNTHETIC VEHICLE DATASET
# ---------------------------------------------------------

np.random.seed(42)

number_of_records = 1500

data = pd.DataFrame({
    "mileage_km": np.random.randint(
        5000,
        180000,
        number_of_records
    ),

    "service_gap_months": np.random.randint(
        1,
        30,
        number_of_records
    ),

    "engine_alerts": np.random.randint(
        0,
        10,
        number_of_records
    ),

    "brake_wear_percent": np.random.randint(
        5,
        100,
        number_of_records
    ),

    "battery_voltage": np.round(
        np.random.uniform(
            10.0,
            14.8,
            number_of_records
        ),
        2
    ),

    "customer_complaints": np.random.randint(
        0,
        7,
        number_of_records
    ),

    "vehicle_age_years": np.random.randint(
        1,
        16,
        number_of_records
    ),

    "driving_pattern": np.random.choice(
        ["City", "Highway", "Mixed"],
        size=number_of_records,
        p=[0.45, 0.25, 0.30]
    )
})


# ---------------------------------------------------------
# 2. CREATE A SERVICE-RISK SCORE
# ---------------------------------------------------------
# This section is used only to create a realistic synthetic
# target column. The risk score is not given to the model.
# ---------------------------------------------------------

risk_score = (
    data["mileage_km"] / 35000
    + data["service_gap_months"] * 0.28
    + data["engine_alerts"] * 0.90
    + data["brake_wear_percent"] * 0.06
    + (14 - data["battery_voltage"]).clip(lower=0) * 1.80
    + data["customer_complaints"] * 0.70
    + data["vehicle_age_years"] * 0.22
    + data["driving_pattern"].map({
        "City": 1.5,
        "Mixed": 0.8,
        "Highway": 0.2
    })
    + np.random.normal(
        loc=0,
        scale=1.5,
        size=number_of_records
    )
)


# ---------------------------------------------------------
# 3. CREATE TARGET CLASSES
# ---------------------------------------------------------

data["service_priority"] = pd.qcut(
    risk_score,
    q=[0, 0.40, 0.75, 1.00],
    labels=["Low", "Medium", "High"]
).astype(str)


# ---------------------------------------------------------
# 4. SAVE THE DATASET
# ---------------------------------------------------------

data.to_csv(
    "vehicle_service_knn_dataset.csv",
    index=False
)

print("Dataset shape:")
print(data.shape)

print("\nFirst five records:")
print(data.head())

print("\nTarget distribution:")
print(data["service_priority"].value_counts())


# ---------------------------------------------------------
# 5. SEPARATE FEATURES AND TARGET
# ---------------------------------------------------------

X = data.drop(
    columns="service_priority"
)

y = data["service_priority"]


# ---------------------------------------------------------
# 6. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ---------------------------------------------------------

numerical_features = [
    "mileage_km",
    "service_gap_months",
    "engine_alerts",
    "brake_wear_percent",
    "battery_voltage",
    "customer_complaints",
    "vehicle_age_years"
]

categorical_features = [
    "driving_pattern"
]


# ---------------------------------------------------------
# 7. TRAIN-TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:")
print(X_train.shape[0])

print("Testing records:")
print(X_test.shape[0])


# ---------------------------------------------------------
# 8. CREATE PREPROCESSOR
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ---------------------------------------------------------
# 9. CREATE KNN PIPELINE
# ---------------------------------------------------------

knn_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "knn",
            KNeighborsClassifier()
        )
    ]
)


# ---------------------------------------------------------
# 10. DEFINE GRID-SEARCH PARAMETERS
# ---------------------------------------------------------

parameter_grid = {
    "knn__n_neighbors": [
        3, 5, 7, 9, 11, 15
    ],

    "knn__weights": [
        "uniform",
        "distance"
    ],

    "knn__metric": [
        "euclidean",
        "manhattan"
    ]
}


# ---------------------------------------------------------
# 11. CREATE GRID SEARCH
# ---------------------------------------------------------

grid_search = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=parameter_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)


# ---------------------------------------------------------
# 12. TRAIN THE MODEL
# ---------------------------------------------------------

grid_search.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# 13. DISPLAY BEST PARAMETERS
# ---------------------------------------------------------

print("\nBest KNN parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1 score:")
print(
    round(
        grid_search.best_score_,
        4
    )
)


# ---------------------------------------------------------
# 14. GET THE BEST MODEL
# ---------------------------------------------------------

best_knn_model = grid_search.best_estimator_


# ---------------------------------------------------------
# 15. MAKE TEST PREDICTIONS
# ---------------------------------------------------------

test_predictions = best_knn_model.predict(
    X_test
)


# ---------------------------------------------------------
# 16. EVALUATE THE MODEL
# ---------------------------------------------------------

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print("\nTest accuracy:")
print(
    round(
        test_accuracy,
        4
    )
)

print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_test,
        test_predictions,
        labels=["Low", "Medium", "High"]
    )
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions
    )
)


# ---------------------------------------------------------
# 17. DISPLAY CONFUSION MATRIX
# ---------------------------------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_predictions,
    labels=["Low", "Medium", "High"],
    cmap="Blues"
)

plt.title(
    "KNN Vehicle Service-Priority Confusion Matrix"
)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 18. CREATE NEW VEHICLE RECORDS
# ---------------------------------------------------------

new_vehicles = pd.DataFrame({
    "mileage_km": [
        25000,
        72000,
        145000
    ],

    "service_gap_months": [
        4,
        13,
        25
    ],

    "engine_alerts": [
        0,
        3,
        8
    ],

    "brake_wear_percent": [
        22,
        62,
        94
    ],

    "battery_voltage": [
        14.2,
        12.5,
        10.4
    ],

    "customer_complaints": [
        0,
        2,
        6
    ],

    "vehicle_age_years": [
        2,
        7,
        14
    ],

    "driving_pattern": [
        "Highway",
        "Mixed",
        "City"
    ]
})


# ---------------------------------------------------------
# 19. PREDICT NEW VEHICLE PRIORITIES
# ---------------------------------------------------------

new_predictions = best_knn_model.predict(
    new_vehicles
)

new_probabilities = best_knn_model.predict_proba(
    new_vehicles
)


# ---------------------------------------------------------
# 20. CREATE PROBABILITY DATAFRAME
# ---------------------------------------------------------

class_names = (
    best_knn_model
    .named_steps["knn"]
    .classes_
)

probability_data = pd.DataFrame(
    new_probabilities,
    columns=[
        f"probability_{class_name}"
        for class_name in class_names
    ]
)


# ---------------------------------------------------------
# 21. CREATE FINAL PREDICTION REPORT
# ---------------------------------------------------------

prediction_report = new_vehicles.copy()

prediction_report["predicted_service_priority"] = (
    new_predictions
)

prediction_report["prediction_confidence"] = (
    new_probabilities.max(axis=1)
).round(4)

prediction_report = pd.concat(
    [
        prediction_report.reset_index(drop=True),
        probability_data.reset_index(drop=True)
    ],
    axis=1
)

print("\nNew vehicle predictions:")
print(
    prediction_report.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 22. SAVE PREDICTIONS
# ---------------------------------------------------------

prediction_report.to_csv(
    "vehicle_service_knn_predictions.csv",
    index=False
)

print(
    "\nPredictions saved to "
    "vehicle_service_knn_predictions.csv"
)
```

---

# 7. Why feature scaling is compulsory for KNN

KNN calculates the distance between records.

Suppose two features are:

| Feature       | Example value |
| ------------- | ------------: |
| Mileage       |       120,000 |
| Engine alerts |             5 |

Mileage has a much larger numerical value than engine alerts.

Without scaling:

```text
Mileage difference = 50,000
Engine-alert difference = 3
```

Mileage will dominate the distance calculation. KNN may practically ignore engine alerts.

`StandardScaler` transforms the features into comparable ranges:

[
z=\frac{x-\mu}{\sigma}
]

After scaling:

| Feature       | Scaled value |
| ------------- | -----------: |
| Mileage       |         1.25 |
| Engine alerts |         1.10 |

Both features can now contribute fairly to the distance.

---

# 8. Important KNN hyperparameters

## `n_neighbors`

```python
KNeighborsClassifier(
    n_neighbors=5
)
```

It controls the number of nearby records considered.

### Small K

```text
K = 1 or 3
```

* Sensitive to individual records
* Sensitive to noise
* Complicated decision boundary
* Higher chance of overfitting

### Large K

```text
K = 15 or 25
```

* Considers more records
* Produces a smoother boundary
* May ignore local patterns
* Higher chance of underfitting

---

## `weights`

### Uniform weights

```python
weights="uniform"
```

Every neighbour gets the same voting power.

```text
Neighbour 1 → one vote
Neighbour 2 → one vote
Neighbour 3 → one vote
```

### Distance weights

```python
weights="distance"
```

Closer neighbours receive more importance.

```text
Very close neighbour → high importance
Distant neighbour    → low importance
```

This is useful when the nearest vehicle should influence the prediction more strongly.

---

## `metric`

### Euclidean distance

```python
metric="euclidean"
```

[
d=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
]

It calculates straight-line distance.

### Manhattan distance

```python
metric="manhattan"
```

[
d=\sum_{i=1}^{n}|x_i-y_i|
]

It calculates the total absolute difference across features.

Grid Search tests both metrics and selects the better one.

---

# 9. Why use `f1_macro`?

The target contains three classes:

```text
Low
Medium
High
```

Accuracy only measures the overall percentage of correct predictions.

A model could perform well for `Low` priority but poorly for `High` priority.

```python
scoring="f1_macro"
```

Macro F1:

1. Calculates the F1 score separately for every class.
2. Gives equal importance to every class.
3. Calculates their average.

This is useful because identifying `High`-priority vehicles is important even when that class has fewer records.

---

# 10. Grid Search combinations

The parameter grid contains:

```text
6 values of K
2 weight options
2 distance metrics
```

Number of combinations:

[
6 \times 2 \times 2 = 24
]

With five-fold cross-validation:

[
24 \times 5 = 120
]

Therefore, Grid Search performs **120 model fits** before selecting the best configuration.

---

# 11. Understanding the prediction report

A possible report structure is:

| Vehicle | Predicted priority | Confidence |
| ------: | ------------------ | ---------: |
|       1 | Low                |       0.91 |
|       2 | Medium             |       0.68 |
|       3 | High               |       0.95 |

Interpretation:

### Vehicle 1

```text
Low mileage
Recently serviced
No engine warnings
Low brake wear
Healthy battery
```

Prediction:

```text
Low priority
```

### Vehicle 2

```text
Moderate mileage
Some engine warnings
Moderate service delay
```

Prediction:

```text
Medium priority
```

### Vehicle 3

```text
Very high mileage
Long service gap
Many engine alerts
Severe brake wear
Low battery voltage
```

Prediction:

```text
High priority
```

---

# 12. Business actions

```python
action_mapping = {
    "Low": "Schedule routine maintenance",
    "Medium": "Schedule service within 7 days",
    "High": "Contact customer and inspect immediately"
}

prediction_report["recommended_action"] = (
    prediction_report[
        "predicted_service_priority"
    ].map(action_mapping)
)

print(
    prediction_report[
        [
            "predicted_service_priority",
            "prediction_confidence",
            "recommended_action"
        ]
    ]
)
```

---

# 13. Advantages of KNN in this case study

* Easy to understand
* No complicated mathematical training process
* Works well when similar vehicles have similar risks
* Supports multiclass classification
* Can capture nonlinear patterns
* Distance-weighted voting can prioritize highly similar vehicles

---

# 14. Limitations

* Prediction becomes slower when the dataset grows
* Feature scaling is compulsory
* Sensitive to irrelevant features
* Sensitive to the selected value of K
* Requires storing the training dataset
* May perform poorly with many features due to the curse of dimensionality

---

# 15. Final interpretation

KNN classifies a vehicle by comparing it with similar historical vehicles.

```text
New vehicle
      ↓
Scale and encode its information
      ↓
Calculate distance from training vehicles
      ↓
Find the K nearest vehicles
      ↓
Collect their classes
      ↓
Use majority or distance-weighted voting
      ↓
Predict Low, Medium or High service priority
```

The model does not create rules such as:

```text
If mileage > 100,000, predict High
```

Instead, it asks:

> Which historical vehicles are most similar to this vehicle, and what service priorities did those vehicles receive?
