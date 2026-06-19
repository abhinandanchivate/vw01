# Complex Neural Network Regression Case Study

## Predicting 90-Day Vehicle Maintenance Cost

## 1. Clear Problem Statement

A fleet-management company operates thousands of passenger and commercial vehicles. It wants to predict the **total maintenance cost for each vehicle over the next 90 days**.

At present, maintenance budgets are based mainly on vehicle age and mileage. This approach ignores important factors such as:

* Engine-warning frequency
* Battery health
* Brake wear
* Vehicle usage pattern
* Previous breakdowns
* Service history
* Commercial load
* Harsh-driving events
* Regional operating conditions

The company wants to build a **Neural Network Regression model** that predicts:

```text
Expected maintenance cost during the next 90 days, in INR
```

The prediction will help the company:

* Prepare maintenance budgets
* Reserve expensive spare parts
* Identify high-cost vehicles
* Schedule preventive maintenance
* Decide whether to repair or replace a vehicle
* Allocate workshop capacity

---

# 2. Why This Is a Regression Problem

The output is a continuous numerical value.

Examples:

```text
₹12,450
₹38,700
₹91,250
```

It is not predicting categories such as High Risk or Low Risk.

---

# 3. Dataset Features

Each row represents one vehicle-condition snapshot.

| Feature                  | Type        | Description                           |
| ------------------------ | ----------- | ------------------------------------- |
| `mileage_km`             | Numerical   | Total vehicle mileage                 |
| `vehicle_age_years`      | Numerical   | Age of the vehicle                    |
| `service_gap_months`     | Numerical   | Months since last service             |
| `engine_alerts_30d`      | Numerical   | Engine alerts in the last 30 days     |
| `brake_wear_percent`     | Numerical   | Current brake wear                    |
| `battery_health_percent` | Numerical   | Estimated battery health              |
| `vibration_level`        | Numerical   | Engine vibration measurement          |
| `previous_breakdowns`    | Numerical   | Previous breakdown count              |
| `harsh_braking_events`   | Numerical   | Harsh braking events                  |
| `average_load_percent`   | Numerical   | Average carried load                  |
| `fuel_efficiency_kmpl`   | Numerical   | Current fuel efficiency               |
| `fuel_type`              | Categorical | Petrol, Diesel, Hybrid or Electric    |
| `vehicle_segment`        | Categorical | Hatchback, Sedan, SUV or Commercial   |
| `driving_pattern`        | Categorical | City, Highway or Mixed                |
| `region`                 | Categorical | North, South, East or West            |
| `maintenance_cost_90d`   | Target      | Maintenance cost for the next 90 days |

---

# 4. Why the Problem Is Complex

The maintenance cost does not depend on a single feature.

For example:

```text
High mileage alone
→ may not create high maintenance cost
```

But:

```text
High mileage
+ long service gap
+ high brake wear
+ low battery health
+ multiple engine alerts
→ may create very high maintenance cost
```

The relationship is nonlinear because combinations of conditions increase cost more than individual conditions.

A neural network can learn these interactions.

---

# 5. Proposed Neural Network Architecture

```text
Numerical features
        ↓
Missing-value imputation
        ↓
StandardScaler
        │
        ├─────────────────────┐
        │                     │
Categorical features          │
        ↓                     │
Missing-value imputation      │
        ↓                     │
One-Hot Encoding              │
        │                     │
        └──────────┬──────────┘
                   ↓
            Combined features
                   ↓
            Hidden layer 1
              128 neurons
                   ↓
            Hidden layer 2
               64 neurons
                   ↓
            Hidden layer 3
               32 neurons
                   ↓
            Hidden layer 4
               16 neurons
                   ↓
             Output neuron
                   ↓
     Predicted 90-day maintenance cost
```

---

# 6. Complete Python Solution

```python
# ============================================================
# COMPLEX NEURAL NETWORK REGRESSION CASE STUDY
# Predicting 90-Day Vehicle Maintenance Cost
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import (
    GridSearchCV,
    train_test_split
)

from sklearn.neural_network import MLPRegressor

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


# ============================================================
# 1. GENERATE A REALISTIC SAMPLE DATASET
# ============================================================

np.random.seed(42)

number_of_records = 5000


data = pd.DataFrame({

    "mileage_km": np.random.randint(
        5000,
        250000,
        number_of_records
    ),

    "vehicle_age_years": np.random.randint(
        1,
        16,
        number_of_records
    ),

    "service_gap_months": np.random.randint(
        1,
        25,
        number_of_records
    ),

    "engine_alerts_30d": np.random.poisson(
        2.5,
        number_of_records
    ),

    "brake_wear_percent": np.random.randint(
        10,
        100,
        number_of_records
    ),

    "battery_health_percent": np.random.randint(
        30,
        101,
        number_of_records
    ),

    "vibration_level": np.round(
        np.random.uniform(
            0.5,
            10.0,
            number_of_records
        ),
        2
    ),

    "previous_breakdowns": np.random.poisson(
        1.2,
        number_of_records
    ),

    "harsh_braking_events": np.random.randint(
        0,
        60,
        number_of_records
    ),

    "average_load_percent": np.random.randint(
        20,
        101,
        number_of_records
    ),

    "fuel_efficiency_kmpl": np.round(
        np.random.uniform(
            6,
            25,
            number_of_records
        ),
        2
    ),

    "fuel_type": np.random.choice(
        [
            "Petrol",
            "Diesel",
            "Hybrid",
            "Electric"
        ],
        number_of_records,
        p=[
            0.30,
            0.40,
            0.15,
            0.15
        ]
    ),

    "vehicle_segment": np.random.choice(
        [
            "Hatchback",
            "Sedan",
            "SUV",
            "Commercial"
        ],
        number_of_records,
        p=[
            0.25,
            0.25,
            0.30,
            0.20
        ]
    ),

    "driving_pattern": np.random.choice(
        [
            "City",
            "Highway",
            "Mixed"
        ],
        number_of_records,
        p=[
            0.45,
            0.25,
            0.30
        ]
    ),

    "region": np.random.choice(
        [
            "North",
            "South",
            "East",
            "West"
        ],
        number_of_records
    )
})


# ============================================================
# 2. CREATE REALISTIC TARGET VARIABLE
# ============================================================

base_cost = 3000


# Basic numerical impact
cost = (

    base_cost

    + data["mileage_km"] * 0.045

    + data["vehicle_age_years"] * 850

    + data["service_gap_months"] * 500

    + data["engine_alerts_30d"] * 2200

    + data["brake_wear_percent"] * 95

    + (100 - data["battery_health_percent"]) * 170

    + data["vibration_level"] * 900

    + data["previous_breakdowns"] * 2800

    + data["harsh_braking_events"] * 120

    + data["average_load_percent"] * 75

    + (25 - data["fuel_efficiency_kmpl"]) * 280
)


# ------------------------------------------------------------
# Categorical effects
# ------------------------------------------------------------

fuel_cost_effect = data["fuel_type"].map({

    "Petrol": 1500,

    "Diesel": 2300,

    "Hybrid": 3500,

    "Electric": 4200
})


segment_cost_effect = data["vehicle_segment"].map({

    "Hatchback": 0,

    "Sedan": 2500,

    "SUV": 5000,

    "Commercial": 9000
})


driving_cost_effect = data["driving_pattern"].map({

    "Highway": 0,

    "Mixed": 2000,

    "City": 4500
})


region_cost_effect = data["region"].map({

    "North": 1800,

    "South": 1000,

    "East": 2200,

    "West": 1500
})


cost = (

    cost

    + fuel_cost_effect

    + segment_cost_effect

    + driving_cost_effect

    + region_cost_effect
)


# ------------------------------------------------------------
# Nonlinear interaction effects
# ------------------------------------------------------------

# High mileage and old vehicle together
old_high_mileage_penalty = np.where(

    (
        data["mileage_km"] > 150000
    )
    &
    (
        data["vehicle_age_years"] > 8
    ),

    15000,

    0
)


# High brake wear combined with harsh braking
brake_risk_penalty = np.where(

    (
        data["brake_wear_percent"] > 75
    )
    &
    (
        data["harsh_braking_events"] > 35
    ),

    11000,

    0
)


# Low battery health with many engine alerts
electrical_risk_penalty = np.where(

    (
        data["battery_health_percent"] < 50
    )
    &
    (
        data["engine_alerts_30d"] >= 4
    ),

    13000,

    0
)


# Commercial vehicle carrying high load
commercial_load_penalty = np.where(

    (
        data["vehicle_segment"] == "Commercial"
    )
    &
    (
        data["average_load_percent"] > 80
    ),

    17000,

    0
)


# Long service gap and high vibration
maintenance_delay_penalty = np.where(

    (
        data["service_gap_months"] > 15
    )
    &
    (
        data["vibration_level"] > 7
    ),

    12500,

    0
)


cost = (

    cost

    + old_high_mileage_penalty

    + brake_risk_penalty

    + electrical_risk_penalty

    + commercial_load_penalty

    + maintenance_delay_penalty
)


# ------------------------------------------------------------
# Add random real-world variation
# ------------------------------------------------------------

noise = np.random.normal(

    loc=0,

    scale=6000,

    size=number_of_records
)


data["maintenance_cost_90d"] = (

    cost + noise

).clip(

    lower=2500

).round(2)


print("Dataset preview:")

print(
    data.head().to_string(
        index=False
    )
)


print(
    "\nDataset shape:",
    data.shape
)


print(
    "\nTarget summary:"
)

print(
    data[
        "maintenance_cost_90d"
    ].describe().round(2)
)


# ============================================================
# 3. INTRODUCE SOME MISSING VALUES
# ============================================================

columns_with_missing_values = [

    "battery_health_percent",

    "vibration_level",

    "fuel_efficiency_kmpl",

    "fuel_type",

    "region"
]


for column in columns_with_missing_values:

    missing_indexes = np.random.choice(

        data.index,

        size=int(
            len(data) * 0.02
        ),

        replace=False
    )

    data.loc[
        missing_indexes,
        column
    ] = np.nan


print(
    "\nMissing values:"
)

print(
    data.isnull().sum()
)


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop(

    columns=[
        "maintenance_cost_90d"
    ]
)


y = data[
    "maintenance_cost_90d"
]


# ============================================================
# 5. DEFINE FEATURE GROUPS
# ============================================================

numerical_features = [

    "mileage_km",

    "vehicle_age_years",

    "service_gap_months",

    "engine_alerts_30d",

    "brake_wear_percent",

    "battery_health_percent",

    "vibration_level",

    "previous_breakdowns",

    "harsh_braking_events",

    "average_load_percent",

    "fuel_efficiency_kmpl"
]


categorical_features = [

    "fuel_type",

    "vehicle_segment",

    "driving_pattern",

    "region"
]


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42
)


print(
    "\nTraining records:",
    len(X_train)
)


print(
    "Testing records:",
    len(X_test)
)


# ============================================================
# 7. NUMERICAL PREPROCESSING PIPELINE
# ============================================================

numerical_pipeline = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",

            StandardScaler()
        )
    ]
)


# ============================================================
# 8. CATEGORICAL PREPROCESSING PIPELINE
# ============================================================

categorical_pipeline = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",

            OneHotEncoder(

                handle_unknown="ignore",

                sparse_output=False
            )
        )
    ]
)


# For older Scikit-learn versions, replace:
#
# sparse_output=False
#
# with:
#
# sparse=False


# ============================================================
# 9. COLUMN TRANSFORMER
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numerical",

            numerical_pipeline,

            numerical_features
        ),

        (
            "categorical",

            categorical_pipeline,

            categorical_features
        )
    ]
)


# ============================================================
# 10. CREATE NEURAL NETWORK
# ============================================================

neural_network = MLPRegressor(

    hidden_layer_sizes=(
        128,
        64,
        32,
        16
    ),

    activation="relu",

    solver="adam",

    alpha=0.0005,

    learning_rate_init=0.001,

    batch_size=64,

    max_iter=600,

    early_stopping=True,

    validation_fraction=0.15,

    n_iter_no_change=25,

    random_state=42
)


# ============================================================
# 11. CREATE COMPLETE PIPELINE
# ============================================================

regression_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",

            preprocessor
        ),

        (
            "model",

            neural_network
        )
    ]
)


# ============================================================
# 12. APPLY LOG TRANSFORMATION TO TARGET
# ============================================================

# Maintenance cost is positively skewed.
#
# The model learns log(cost) rather than raw cost.
# Predictions are automatically converted back to INR.

target_transformed_model = TransformedTargetRegressor(

    regressor=regression_pipeline,

    func=np.log1p,

    inverse_func=np.expm1
)


# ============================================================
# 13. HYPERPARAMETER TUNING
# ============================================================

parameter_grid = {

    "regressor__model__hidden_layer_sizes": [

        (64, 32),

        (128, 64, 32),

        (128, 64, 32, 16)
    ],

    "regressor__model__alpha": [

        0.0001,

        0.0005,

        0.001
    ],

    "regressor__model__learning_rate_init": [

        0.001,

        0.0005
    ]
}


grid_search = GridSearchCV(

    estimator=target_transformed_model,

    param_grid=parameter_grid,

    scoring="neg_mean_absolute_error",

    cv=3,

    n_jobs=-1,

    verbose=1
)


print(
    "\nTraining and tuning "
    "the neural network..."
)


grid_search.fit(

    X_train,

    y_train
)


best_model = grid_search.best_estimator_


print(
    "\nBest parameters:"
)

print(
    grid_search.best_params_
)


print(
    "\nBest cross-validation MAE:",
    round(
        -grid_search.best_score_,
        2
    )
)


# ============================================================
# 14. MAKE TEST PREDICTIONS
# ============================================================

test_predictions = best_model.predict(

    X_test
)


# Maintenance cost should not be negative
test_predictions = np.maximum(

    test_predictions,

    0
)


# ============================================================
# 15. EVALUATE THE MODEL
# ============================================================

mae = mean_absolute_error(

    y_test,

    test_predictions
)


mse = mean_squared_error(

    y_test,

    test_predictions
)


rmse = np.sqrt(mse)


r2 = r2_score(

    y_test,

    test_predictions
)


mape = mean_absolute_percentage_error(

    y_test,

    test_predictions
) * 100


print(
    "\nModel Evaluation"
)


print(
    "Mean Absolute Error:",
    f"₹{mae:,.2f}"
)


print(
    "Root Mean Squared Error:",
    f"₹{rmse:,.2f}"
)


print(
    "R-squared Score:",
    round(
        r2,
        4
    )
)


print(
    "Mean Absolute Percentage Error:",
    f"{mape:.2f}%"
)


# ============================================================
# 16. ACTUAL VS PREDICTED RESULTS
# ============================================================

results = X_test.copy()


results[
    "actual_cost"
] = y_test.values


results[
    "predicted_cost"
] = test_predictions.round(2)


results[
    "absolute_error"
] = (

    results["actual_cost"]

    -

    results["predicted_cost"]

).abs().round(2)


results[
    "percentage_error"
] = (

    results["absolute_error"]

    /

    results["actual_cost"]

    * 100

).round(2)


results = results.sort_values(

    "absolute_error",

    ascending=False
)


print(
    "\nLargest prediction errors:"
)


print(

    results[
        [
            "mileage_km",
            "vehicle_segment",
            "service_gap_months",
            "actual_cost",
            "predicted_cost",
            "absolute_error",
            "percentage_error"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 17. PREDICT FOR NEW VEHICLES
# ============================================================

new_vehicles = pd.DataFrame({

    "mileage_km": [

        32000,

        98000,

        185000,

        220000
    ],

    "vehicle_age_years": [

        2,

        7,

        11,

        13
    ],

    "service_gap_months": [

        4,

        12,

        19,

        22
    ],

    "engine_alerts_30d": [

        0,

        3,

        7,

        9
    ],

    "brake_wear_percent": [

        28,

        65,

        86,

        94
    ],

    "battery_health_percent": [

        91,

        67,

        43,

        35
    ],

    "vibration_level": [

        1.8,

        5.4,

        8.1,

        9.3
    ],

    "previous_breakdowns": [

        0,

        2,

        5,

        7
    ],

    "harsh_braking_events": [

        5,

        28,

        46,

        55
    ],

    "average_load_percent": [

        35,

        60,

        84,

        95
    ],

    "fuel_efficiency_kmpl": [

        19.5,

        14.2,

        9.4,

        7.1
    ],

    "fuel_type": [

        "Petrol",

        "Diesel",

        "Diesel",

        "Diesel"
    ],

    "vehicle_segment": [

        "Hatchback",

        "SUV",

        "Commercial",

        "Commercial"
    ],

    "driving_pattern": [

        "Highway",

        "Mixed",

        "City",

        "City"
    ],

    "region": [

        "West",

        "South",

        "North",

        "East"
    ]
})


new_predictions = best_model.predict(

    new_vehicles
)


new_predictions = np.maximum(

    new_predictions,

    0
)


prediction_results = new_vehicles.copy()


prediction_results[
    "predicted_maintenance_cost_90d"
] = new_predictions.round(2)


# ============================================================
# 18. CREATE BUSINESS COST CATEGORY
# ============================================================

prediction_results[
    "cost_category"
] = pd.cut(

    prediction_results[
        "predicted_maintenance_cost_90d"
    ],

    bins=[

        -np.inf,

        30000,

        60000,

        100000,

        np.inf
    ],

    labels=[

        "Low Cost",

        "Moderate Cost",

        "High Cost",

        "Very High Cost"
    ]
)


def generate_action(row):

    predicted_cost = row[
        "predicted_maintenance_cost_90d"
    ]

    if predicted_cost >= 100000:

        return (
            "Immediate detailed inspection, reserve parts "
            "and perform repair-versus-replacement analysis"
        )

    elif predicted_cost >= 60000:

        return (
            "Schedule priority maintenance and reserve "
            "major mechanical components"
        )

    elif predicted_cost >= 30000:

        return (
            "Schedule preventive maintenance within "
            "the next two weeks"
        )

    else:

        return (
            "Continue normal maintenance schedule"
        )


prediction_results[
    "recommended_action"
] = prediction_results.apply(

    generate_action,

    axis=1
)


print(
    "\nNew Vehicle Predictions:"
)


print(

    prediction_results[
        [
            "mileage_km",
            "vehicle_segment",
            "service_gap_months",
            "engine_alerts_30d",
            "predicted_maintenance_cost_90d",
            "cost_category",
            "recommended_action"
        ]
    ].to_string(index=False)
)


# ============================================================
# 19. SAVE RESULTS
# ============================================================

results.to_csv(

    "neural_network_test_predictions.csv",

    index=False
)


prediction_results.to_csv(

    "new_vehicle_maintenance_predictions.csv",

    index=False
)


print(
    "\nPrediction files saved successfully."
)


# ============================================================
# 20. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(

    figsize=(8, 6)
)


plt.scatter(

    y_test,

    test_predictions,

    alpha=0.5
)


minimum_value = min(

    y_test.min(),

    test_predictions.min()
)


maximum_value = max(

    y_test.max(),

    test_predictions.max()
)


plt.plot(

    [
        minimum_value,
        maximum_value
    ],

    [
        minimum_value,
        maximum_value
    ],

    linestyle="--"
)


plt.xlabel(

    "Actual Maintenance Cost"
)


plt.ylabel(

    "Predicted Maintenance Cost"
)


plt.title(

    "Actual vs Predicted 90-Day Maintenance Cost"
)


plt.tight_layout()

plt.show()


# ============================================================
# 21. RESIDUAL PLOT
# ============================================================

residuals = (

    y_test.values

    -

    test_predictions
)


plt.figure(

    figsize=(8, 6)
)


plt.scatter(

    test_predictions,

    residuals,

    alpha=0.5
)


plt.axhline(

    y=0,

    linestyle="--"
)


plt.xlabel(

    "Predicted Maintenance Cost"
)


plt.ylabel(

    "Residual: Actual - Predicted"
)


plt.title(

    "Neural Network Residual Analysis"
)


plt.tight_layout()

plt.show()


# ============================================================
# 22. TRAINING LOSS CURVE
# ============================================================

trained_network = (

    best_model
    .regressor_
    .named_steps["model"]
)


plt.figure(

    figsize=(8, 5)
)


plt.plot(

    trained_network.loss_curve_
)


plt.xlabel(

    "Training Iteration"
)


plt.ylabel(

    "Loss"
)


plt.title(

    "Neural Network Training Loss"
)


plt.tight_layout()

plt.show()
```

---

# 7. Main Complex Concepts Used

## Mixed Data Processing

The model handles both:

```text
Numerical data
+
Categorical data
```

Numerical features are processed using:

```text
Median imputation
→ StandardScaler
```

Categorical features are processed using:

```text
Most-frequent imputation
→ One-Hot Encoding
```

---

## Nonlinear Interactions

The dataset contains complex relationships such as:

```text
High brake wear
+ frequent harsh braking
→ major braking-system repair cost
```

Another example:

```text
Commercial vehicle
+ average load above 80%
→ suspension and drivetrain cost increase
```

These interactions make the problem suitable for a neural network.

---

## Target Transformation

Maintenance cost is often positively skewed.

For example:

```text
Most vehicles: ₹15,000–₹60,000

A small number: above ₹1,00,000
```

The code uses:

```python
TransformedTargetRegressor(
    func=np.log1p,
    inverse_func=np.expm1
)
```

The neural network learns:

```text
log(maintenance cost)
```

The result is then automatically converted back to INR.

This reduces the influence of extreme cost values.

---

## Hyperparameter Tuning

The solution tests:

### Different architectures

```python
(64, 32)

(128, 64, 32)

(128, 64, 32, 16)
```

### Different regularization values

```python
alpha = 0.0001
alpha = 0.0005
alpha = 0.001
```

### Different learning rates

```python
0.001
0.0005
```

The model with the lowest cross-validation MAE is selected.

---

# 8. Business Output Example

| Vehicle   | Predicted cost | Category       | Recommended action                     |
| --------- | -------------: | -------------- | -------------------------------------- |
| Vehicle 1 |        ₹18,500 | Low Cost       | Continue normal maintenance            |
| Vehicle 2 |        ₹47,200 | Moderate Cost  | Schedule preventive maintenance        |
| Vehicle 3 |        ₹83,700 | High Cost      | Priority service and parts reservation |
| Vehicle 4 |      ₹1,24,500 | Very High Cost | Repair-versus-replacement analysis     |

---

# 9. Evaluation Interpretation

Suppose the model gives:

```text
MAE  = ₹5,200
RMSE = ₹7,800
R²   = 0.91
MAPE = 8.6%
```

Interpretation:

* Predictions differ from actual cost by approximately ₹5,200 on average.
* Larger errors produce an RMSE of ₹7,800.
* The model explains approximately 91% of maintenance-cost variation.
* Predictions are approximately 8.6% away from actual cost on average.

---

# 10. Assignment Tasks

1. Generate or load at least 5,000 vehicle records.
2. Identify numerical and categorical columns.
3. Check missing values and outliers.
4. Create separate preprocessing pipelines.
5. Apply one-hot encoding.
6. Scale numerical features.
7. Train a baseline neural-network regressor.
8. tune architecture, learning rate and regularization.
9. Apply early stopping.
10. Calculate MAE, RMSE, R² and MAPE.
11. Plot actual versus predicted values.
12. Perform residual analysis.
13. Identify the ten largest prediction errors.
14. Predict cost for new vehicles.
15. Generate cost categories and recommended actions.
16. Save the trained model and prediction reports.

---

# Final Case Study Question

> Develop an end-to-end neural-network regression solution that predicts the total maintenance cost of fleet vehicles over the next 90 days. The solution must handle numerical and categorical data, missing values, nonlinear feature interactions, target skewness, feature scaling, neural-network architecture tuning, early stopping and business recommendation generation.
