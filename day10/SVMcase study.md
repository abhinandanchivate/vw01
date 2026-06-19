# SVM in Machine Learning — Case Study

## Case Study: Vehicle Service-Risk Classification

### Business Problem

An automobile service company wants to identify vehicles that are likely to experience a serious breakdown within the next 30 days.

Currently, service advisors depend on manual inspection. This can result in:

* High-risk vehicles being missed
* Unexpected vehicle breakdowns
* Emergency repair costs
* Customer dissatisfaction
* Poor spare-parts planning

The company wants to build a **Support Vector Machine classification model** that classifies every vehicle as:

* **High Risk**
* **Low Risk**

---

## 1. Input Features

| Feature               | Description                     |   Example |
| --------------------- | ------------------------------- | --------: |
| `mileage_km`          | Total vehicle mileage           |    85,000 |
| `service_gap_months`  | Months since last service       |        14 |
| `engine_alerts`       | Number of engine-warning alerts |         5 |
| `brake_wear_percent`  | Percentage of brake wear        |        82 |
| `battery_voltage`     | Current battery voltage         |      10.8 |
| `customer_complaints` | Number of recent complaints     |         4 |
| `vehicle_age_years`   | Age of the vehicle              |         8 |
| `service_risk`        | Target variable                 | High Risk |

---

## 2. Why SVM Is Suitable

SVM attempts to find a decision boundary called a **hyperplane** that separates the two classes.

For this problem, it finds a boundary between:

* Vehicles with normal service conditions
* Vehicles showing dangerous maintenance conditions

SVM selects the boundary with the maximum possible distance from both classes. This distance is called the **margin**.

```text
Low-Risk Vehicles          High-Risk Vehicles
      ○  ○  ○                   ●  ●  ●
   ○  ○  ○         |         ●  ●  ●
                    |
             Decision Boundary
                Hyperplane
```

The closest observations on both sides of the boundary are called **support vectors**.

---

# 3. Business Rules Used to Generate Sample Data

For demonstration purposes, a vehicle is more likely to be high risk when:

* Mileage is high
* Service gap is long
* Engine alerts are frequent
* Brake wear is high
* Battery voltage is low
* Customer complaints are high
* Vehicle age is high

These rules are only used to create the sample dataset. In a real project, the target value would come from historical breakdown records.

---

# 4. Complete Python Implementation

```python
# =========================================================
# SVM CASE STUDY
# Vehicle Service-Risk Classification
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)


# ---------------------------------------------------------
# 1. CREATE SAMPLE DATASET
# ---------------------------------------------------------

np.random.seed(42)

number_of_records = 1000

data = pd.DataFrame({
    "mileage_km": np.random.randint(
        10000, 150000, number_of_records
    ),

    "service_gap_months": np.random.randint(
        1, 25, number_of_records
    ),

    "engine_alerts": np.random.randint(
        0, 9, number_of_records
    ),

    "brake_wear_percent": np.random.randint(
        10, 100, number_of_records
    ),

    "battery_voltage": np.round(
        np.random.uniform(9.5, 14.5, number_of_records),
        1
    ),

    "customer_complaints": np.random.randint(
        0, 8, number_of_records
    ),

    "vehicle_age_years": np.random.randint(
        1, 16, number_of_records
    )
})


# ---------------------------------------------------------
# 2. CREATE RISK SCORE FOR SAMPLE TARGET
# ---------------------------------------------------------

risk_score = (
    (data["mileage_km"] > 80000).astype(int) * 2
    + (data["service_gap_months"] > 12).astype(int) * 2
    + (data["engine_alerts"] >= 4).astype(int) * 2
    + (data["brake_wear_percent"] > 70).astype(int) * 2
    + (data["battery_voltage"] < 11.5).astype(int) * 2
    + (data["customer_complaints"] >= 3).astype(int)
    + (data["vehicle_age_years"] > 8).astype(int)
)


# Add a small amount of random variation
risk_score = risk_score + np.random.normal(
    0,
    1,
    number_of_records
)


data["service_risk"] = np.where(
    risk_score >= 7,
    "High Risk",
    "Low Risk"
)


print("First five records:")
print(data.head())

print("\nTarget distribution:")
print(data["service_risk"].value_counts())


# ---------------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# ---------------------------------------------------------

X = data.drop("service_risk", axis=1)

y = data["service_risk"]


# ---------------------------------------------------------
# 4. SPLIT TRAINING AND TEST DATA
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ---------------------------------------------------------
# 5. CREATE SVM PIPELINE
# ---------------------------------------------------------

svm_pipeline = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# ---------------------------------------------------------
# 6. TRAIN THE MODEL
# ---------------------------------------------------------

svm_pipeline.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# 7. MAKE PREDICTIONS
# ---------------------------------------------------------

predictions = svm_pipeline.predict(
    X_test
)

probabilities = svm_pipeline.predict_proba(
    X_test
)


# ---------------------------------------------------------
# 8. MODEL EVALUATION
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

confusion = confusion_matrix(
    y_test,
    predictions,
    labels=["Low Risk", "High Risk"]
)

report = classification_report(
    y_test,
    predictions
)


print("\nModel Accuracy:")
print(round(accuracy, 4))

print("\nConfusion Matrix:")
print(confusion)

print("\nClassification Report:")
print(report)


# ---------------------------------------------------------
# 9. CALCULATE ROC-AUC
# ---------------------------------------------------------

class_order = list(
    svm_pipeline.named_steps["model"].classes_
)

high_risk_index = class_order.index(
    "High Risk"
)

high_risk_probabilities = probabilities[
    :,
    high_risk_index
]

y_test_binary = y_test.map({
    "Low Risk": 0,
    "High Risk": 1
})

roc_auc = roc_auc_score(
    y_test_binary,
    high_risk_probabilities
)

print("\nROC-AUC Score:")
print(round(roc_auc, 4))


# ---------------------------------------------------------
# 10. PREDICT NEW VEHICLES
# ---------------------------------------------------------

new_vehicles = pd.DataFrame({
    "mileage_km": [
        35000,
        92000,
        125000
    ],

    "service_gap_months": [
        4,
        15,
        21
    ],

    "engine_alerts": [
        0,
        5,
        7
    ],

    "brake_wear_percent": [
        30,
        78,
        92
    ],

    "battery_voltage": [
        13.2,
        11.0,
        10.2
    ],

    "customer_complaints": [
        0,
        3,
        6
    ],

    "vehicle_age_years": [
        2,
        9,
        13
    ]
})


new_predictions = svm_pipeline.predict(
    new_vehicles
)

new_probabilities = svm_pipeline.predict_proba(
    new_vehicles
)


prediction_result = new_vehicles.copy()

prediction_result["predicted_service_risk"] = (
    new_predictions
)

prediction_result["high_risk_probability"] = (
    new_probabilities[:, high_risk_index]
    .round(4)
)


print("\nNew Vehicle Predictions:")
print(
    prediction_result.to_string(index=False)
)


# ---------------------------------------------------------
# 11. SAVE OUTPUT
# ---------------------------------------------------------

prediction_result.to_csv(
    "vehicle_service_risk_predictions.csv",
    index=False
)

print(
    "\nPredictions saved to "
    "vehicle_service_risk_predictions.csv"
)


# ---------------------------------------------------------
# 12. DISPLAY ROC CURVE
# ---------------------------------------------------------

RocCurveDisplay.from_predictions(
    y_test_binary,
    high_risk_probabilities
)

plt.title("SVM ROC Curve — Vehicle Service Risk")
plt.show()
```

---

# 5. Workflow Architecture

```text
Historical Vehicle Data
          ↓
Data Validation
          ↓
Separate Features and Target
          ↓
Training and Testing Split
          ↓
StandardScaler
          ↓
SVM Classification Model
          ↓
Model Evaluation
          ↓
New Vehicle Data
          ↓
Risk Prediction
          ↓
Service Recommendation
```

---

# 6. Why StandardScaler Is Required

SVM is sensitive to the scale of input features.

Consider two columns:

| Feature       | Example value |
| ------------- | ------------: |
| Mileage       |        90,000 |
| Engine alerts |             5 |

Without scaling, mileage has a much larger numerical value than engine alerts. The model may incorrectly give mileage more importance simply because of its scale.

`StandardScaler` transforms every feature using:

[
z = \frac{x-\mu}{\sigma}
]

Where:

* (x) is the original value
* (\mu) is the feature mean
* (\sigma) is the feature standard deviation
* (z) is the scaled value

After scaling, the features are represented on a comparable scale.

---

# 7. Understanding the SVM Parameters

## Kernel

```python
kernel="rbf"
```

The kernel controls the type of decision boundary.

| Kernel    | Usage                                            |
| --------- | ------------------------------------------------ |
| `linear`  | Data can be separated using a straight boundary  |
| `rbf`     | Data requires a curved or nonlinear boundary     |
| `poly`    | Polynomial relationships exist                   |
| `sigmoid` | Boundary similar to a neural activation function |

For vehicle risk, the relationship may be nonlinear. For example, high mileage alone may not indicate risk, but high mileage combined with high brake wear and long service gaps may indicate risk.

Therefore, the RBF kernel is appropriate.

---

## C Parameter

```python
C=1.0
```

`C` controls the penalty for classification errors.

|     C value | Behaviour                                                    |
| ----------: | ------------------------------------------------------------ |
|       Low C | Allows more classification errors and creates a wider margin |
|      High C | Tries harder to classify training records correctly          |
| Very high C | May cause overfitting                                        |

### Interpretation

```text
Low C
Wider margin
More training errors allowed
Better generalization may occur
```

```text
High C
Narrower margin
Fewer training errors
Greater overfitting risk
```

---

## Gamma Parameter

```python
gamma="scale"
```

Gamma controls how far the influence of one training record reaches.

| Gamma           | Behaviour                                    |
| --------------- | -------------------------------------------- |
| Low gamma       | Each record has a wider area of influence    |
| High gamma      | Each record affects only nearby observations |
| Very high gamma | Complex boundary and possible overfitting    |

---

## Probability

```python
probability=True
```

This enables:

```python
predict_proba()
```

The model can then provide outputs such as:

| Vehicle   | Low-risk probability | High-risk probability |
| --------- | -------------------: | --------------------: |
| Vehicle 1 |                 0.93 |                  0.07 |
| Vehicle 2 |                 0.22 |                  0.78 |
| Vehicle 3 |                 0.03 |                  0.97 |

---

## Class Weight

```python
class_weight="balanced"
```

This is useful when the dataset has fewer high-risk vehicles than low-risk vehicles.

The model gives additional importance to the minority class rather than focusing mostly on the majority class.

---

# 8. Expected Prediction Interpretation

The exact values may vary, but the result could look like:

| Mileage | Service gap | Engine alerts | Brake wear | Prediction | High-risk probability |
| ------: | ----------: | ------------: | ---------: | ---------- | --------------------: |
|  35,000 |           4 |             0 |        30% | Low Risk   |                  0.04 |
|  92,000 |          15 |             5 |        78% | High Risk  |                  0.84 |
| 125,000 |          21 |             7 |        92% | High Risk  |                  0.98 |

### Vehicle 1

The vehicle has:

* Low mileage
* Recent service
* No engine alerts
* Normal brake wear
* Healthy battery voltage

Therefore, it is predicted as **Low Risk**.

### Vehicle 2

The vehicle has:

* High mileage
* Long service gap
* Several engine alerts
* High brake wear
* Low battery voltage

Therefore, it is predicted as **High Risk**.

### Vehicle 3

The vehicle has extremely high values across several risk indicators. It receives the highest probability of breakdown risk.

---

# 9. Confusion Matrix Interpretation

Suppose the confusion matrix is:

```text
[[130, 10],
 [  8, 52]]
```

| Actual / Predicted | Predicted Low Risk | Predicted High Risk |
| ------------------ | -----------------: | ------------------: |
| Actual Low Risk    |                130 |                  10 |
| Actual High Risk   |                  8 |                  52 |

### Meaning

* **130 true negatives:** Low-risk vehicles correctly classified
* **52 true positives:** High-risk vehicles correctly classified
* **10 false positives:** Safe vehicles incorrectly classified as high risk
* **8 false negatives:** High-risk vehicles incorrectly classified as low risk

For this case study, false negatives are particularly dangerous because a risky vehicle may not receive timely service.

---

# 10. Business Actions

| Predicted risk                   | Recommended action                                        |
| -------------------------------- | --------------------------------------------------------- |
| Low Risk                         | Continue normal maintenance schedule                      |
| High Risk, probability 50–70%    | Contact customer and recommend inspection                 |
| High Risk, probability 70–90%    | Schedule service within seven days                        |
| High Risk, probability above 90% | Immediate inspection and emergency service recommendation |

---

# 11. Hyperparameter Tuning

The model can be improved using `GridSearchCV`.

```python
from sklearn.model_selection import GridSearchCV

parameter_grid = {
    "model__kernel": [
        "linear",
        "rbf"
    ],

    "model__C": [
        0.1,
        1,
        10,
        100
    ],

    "model__gamma": [
        "scale",
        0.01,
        0.1,
        1
    ]
}


grid_search = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=parameter_grid,
    scoring="f1_weighted",
    cv=5,
    n_jobs=-1
)


grid_search.fit(
    X_train,
    y_train
)


print("Best parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation score:")
print(grid_search.best_score_)


best_model = grid_search.best_estimator_

best_predictions = best_model.predict(
    X_test
)

print("\nBest model classification report:")
print(
    classification_report(
        y_test,
        best_predictions
    )
)
```

---

# 12. SVM Advantages and Limitations

| Advantages                                        | Limitations                                     |
| ------------------------------------------------- | ----------------------------------------------- |
| Effective for complex classification boundaries   | Training can be slow on very large datasets     |
| Works well with high-dimensional data             | Feature scaling is necessary                    |
| Uses only important support vectors               | Parameter tuning can be difficult               |
| Supports nonlinear classification through kernels | Less interpretable than decision trees          |
| Can control overfitting using C and gamma         | Probability calculation increases training time |

---

# 13. Case Study Conclusion

The SVM model helps the automobile service company identify vehicles with a high probability of breakdown.

The complete solution includes:

```text
Vehicle measurements
        ↓
Feature scaling
        ↓
SVM model
        ↓
High-risk or low-risk classification
        ↓
Probability calculation
        ↓
Service recommendation
```

The model should be evaluated mainly using:

* Recall for the High Risk class
* F1-score
* Confusion matrix
* ROC-AUC

Accuracy alone should not be used because missing a high-risk vehicle may be more expensive and dangerous than incorrectly recommending an inspection for a low-risk vehicle.
