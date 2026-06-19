# Complex SVM Case Study

## Multi-Class Vehicle Service Priority Classification

## 1. Case Study Title

**Predicting Vehicle Service Priority Using Support Vector Machine**

---

# 2. Clear Problem Statement

A large automobile manufacturer operates service centres across multiple cities. Thousands of connected vehicles continuously send diagnostic and usage information such as mileage, engine temperature, battery voltage, brake wear, vibration level, service history and fault-code frequency.

Currently, service advisers manually examine this information to decide whether a vehicle requires routine service, priority inspection or emergency attention.

This manual process creates several problems:

* High-risk vehicles may not be identified early.
* Service advisers may assign different priorities to similar vehicles.
* Emergency breakdowns increase towing and warranty costs.
* Service centres may not have the required spare parts available.
* Low-risk vehicles may be unnecessarily scheduled for urgent inspection.
* Customers may experience unexpected vehicle failures.

The company wants to develop an **SVM-based multi-class classification system** that examines vehicle-condition data and assigns every vehicle to one of the following service-priority categories:

| Class      | Meaning                                                |
| ---------- | ------------------------------------------------------ |
| `Normal`   | Vehicle can continue with its regular service schedule |
| `Priority` | Vehicle should be inspected within the next seven days |
| `Critical` | Vehicle requires immediate inspection within 24 hours  |

The primary objective is to correctly identify **Critical vehicles**, because incorrectly classifying a critical vehicle as Normal or Priority may result in breakdowns, safety incidents and high repair costs.

---

# 3. Business Objective

The organization wants to use historical vehicle and service data to answer:

> Based on the current condition, usage pattern and maintenance history of a vehicle, what service priority should be assigned?

The prediction will help the company:

* Contact high-risk customers early
* Schedule service appointments automatically
* Reserve spare parts before vehicle arrival
* Reduce unexpected breakdowns
* Reduce emergency towing expenses
* Improve workshop-capacity planning
* Improve customer satisfaction
* Reduce warranty claims

---

# 4. Machine-Learning Problem Definition

This is a **supervised multi-class classification problem**.

The target variable is:

```text
service_priority
```

It contains three classes:

```text
Normal
Priority
Critical
```

The machine-learning model must learn the relationship between vehicle-condition features and historical service-priority decisions.

---

# 5. Prediction Unit

Each row in the dataset represents:

> One vehicle diagnostic snapshot recorded at a specific date and time.

The same vehicle may appear multiple times if diagnostic information was collected on different dates.

Example:

| vehicle_id | diagnostic_date | mileage_km | engine_alerts | service_priority |
| ---------- | --------------- | ---------: | ------------: | ---------------- |
| VH1001     | 2026-01-10      |     52,000 |             1 | Normal           |
| VH1001     | 2026-05-15      |     68,500 |             5 | Priority         |

---

# 6. Dataset Description

The historical dataset contains approximately **20,000 vehicle diagnostic records** collected over the previous three years.

## Input Features

| Feature                   | Data type   | Description                                 |
| ------------------------- | ----------- | ------------------------------------------- |
| `vehicle_id`              | Identifier  | Unique vehicle number                       |
| `vehicle_age_years`       | Numerical   | Age of the vehicle                          |
| `mileage_km`              | Numerical   | Total distance travelled                    |
| `monthly_distance_km`     | Numerical   | Average monthly distance                    |
| `service_gap_months`      | Numerical   | Months since last service                   |
| `engine_alerts_30d`       | Numerical   | Engine warnings in the last 30 days         |
| `fault_code_count`        | Numerical   | Number of active diagnostic fault codes     |
| `brake_wear_percent`      | Numerical   | Percentage of brake wear                    |
| `battery_voltage`         | Numerical   | Current battery voltage                     |
| `engine_temperature_c`    | Numerical   | Average engine temperature                  |
| `vibration_level`         | Numerical   | Engine vibration measurement                |
| `oil_quality_percent`     | Numerical   | Estimated remaining oil quality             |
| `tyre_pressure_deviation` | Numerical   | Average deviation from recommended pressure |
| `fuel_efficiency_kmpl`    | Numerical   | Average fuel efficiency                     |
| `previous_breakdowns`     | Numerical   | Number of historical breakdowns             |
| `warranty_claims`         | Numerical   | Number of previous warranty claims          |
| `customer_complaints_90d` | Numerical   | Complaints during the last 90 days          |
| `harsh_braking_events`    | Numerical   | Harsh braking incidents                     |
| `overspeed_events`        | Numerical   | Overspeed incidents                         |
| `driving_pattern`         | Categorical | City, Highway or Mixed                      |
| `fuel_type`               | Categorical | Petrol, Diesel, CNG, Hybrid or Electric     |
| `vehicle_segment`         | Categorical | Hatchback, Sedan, SUV or Commercial         |
| `region`                  | Categorical | Operating region                            |
| `service_centre_type`     | Categorical | Company-owned or Authorized                 |
| `service_priority`        | Target      | Normal, Priority or Critical                |

---

# 7. Example Dataset

| Mileage | Service gap | Alerts | Brake wear | Battery | Temperature | Fault codes | Priority |
| ------: | ----------: | -----: | ---------: | ------: | ----------: | ----------: | -------- |
|  28,000 |           4 |      0 |         28 |    13.4 |          88 |           0 | Normal   |
|  74,000 |          13 |      4 |         69 |    11.8 |         101 |           3 | Priority |
| 126,000 |          22 |      9 |         91 |    10.2 |         119 |           8 | Critical |
|  58,000 |           8 |      2 |         52 |    12.7 |          94 |           1 | Normal   |
|  96,000 |          16 |      6 |         81 |    10.9 |         108 |           5 | Critical |

---

# 8. Important Complexity in the Problem

This is not a simple classification problem because several challenges exist.

## 8.1 Nonlinear Relationships

A single feature may not determine vehicle risk.

For example:

* High mileage alone may not indicate a critical condition.
* High mileage combined with high brake wear may indicate Priority.
* High mileage, low battery voltage and frequent fault codes may indicate Critical.

Therefore, a straight-line decision boundary may not be sufficient.

An SVM with an **RBF kernel** can model these nonlinear combinations.

---

## 8.2 Class Imbalance

Most vehicles are likely to be Normal.

Example target distribution:

| Class    | Records | Percentage |
| -------- | ------: | ---------: |
| Normal   |  13,500 |      67.5% |
| Priority |   5,000 |        25% |
| Critical |   1,500 |       7.5% |

The Critical class is relatively small but is the most important class.

A model that predicts every vehicle as Normal may show high accuracy but would be unacceptable.

The model should therefore use:

```python
class_weight="balanced"
```

---

## 8.3 Different Feature Scales

The dataset contains values with very different ranges.

| Feature            | Example range |
| ------------------ | ------------: |
| Mileage            | 5,000–250,000 |
| Battery voltage    |          9–15 |
| Engine alerts      |          0–20 |
| Brake wear         |         0–100 |
| Engine temperature |        70–140 |

SVM calculates distances between observations. Therefore, features must be scaled using `StandardScaler`.

---

## 8.4 Categorical Variables

SVM cannot directly process text categories such as:

```text
City
Highway
Mixed
```

These columns must be converted using `OneHotEncoder`.

---

## 8.5 Missing Values

Some sensor readings may be unavailable due to communication or hardware problems.

Examples:

* Missing battery voltage
* Missing vibration readings
* Missing tyre-pressure information

Numerical missing values can be handled using median imputation.

Categorical missing values can be handled using the most frequent category.

---

## 8.6 Outliers

Connected-vehicle sensors may generate abnormal values.

Examples:

* Battery voltage of 50 volts
* Brake wear of 180%
* Negative mileage
* Engine temperature of 400°C

Such values must be validated and treated before model training.

---

## 8.7 Duplicate Vehicle Records

The dataset may contain multiple diagnostic snapshots for the same vehicle. A random train-test split could place records of the same vehicle in both training and testing data.

This causes data leakage.

A better approach is to split the data using `vehicle_id` groups through:

```python
GroupShuffleSplit
```

or:

```python
GroupKFold
```

---

# 9. Why SVM Is Selected

SVM is suitable for this problem because:

* The relationships between sensor features and service priority can be nonlinear.
* The dataset contains several interacting features.
* SVM works effectively in high-dimensional spaces.
* It focuses on important boundary observations called support vectors.
* RBF kernels can detect complex separation patterns.
* Class weights can help manage imbalanced classes.
* The `C` and `gamma` parameters provide control over model complexity.

---

# 10. SVM Decision Concept

SVM tries to create boundaries between the three classes.

```text
                     Critical
                   ● ● ● ● ●
                ● ● ● ● ●
                     )
                  Boundary 2
                   )
       Priority  ▲ ▲ ▲ ▲
             ▲ ▲ ▲ ▲ ▲
              (
           Boundary 1
             (
 Normal  ○ ○ ○ ○ ○ ○
        ○ ○ ○ ○ ○ ○
```

For a multi-class problem, SVM creates multiple binary classifiers internally.

For example:

```text
Normal vs Priority
Normal vs Critical
Priority vs Critical
```

or it may use a one-versus-rest strategy depending on the configuration.

---

# 11. Project Requirements

The data-science team must build a reusable machine-learning pipeline that performs the following steps:

```text
Raw vehicle data
        ↓
Data validation
        ↓
Remove invalid records
        ↓
Handle missing values
        ↓
Encode categorical columns
        ↓
Scale numerical features
        ↓
Train SVM classifier
        ↓
Tune C, gamma and kernel
        ↓
Evaluate the model
        ↓
Predict service priority
        ↓
Generate service recommendation
```

---

# 12. Data-Preprocessing Requirements

## Numerical Features

The numerical pipeline should contain:

```text
Median Imputation
        ↓
StandardScaler
```

Examples:

```python
mileage_km
battery_voltage
brake_wear_percent
engine_temperature_c
fault_code_count
```

## Categorical Features

The categorical pipeline should contain:

```text
Most-Frequent Imputation
        ↓
OneHotEncoder
```

Examples:

```python
driving_pattern
fuel_type
vehicle_segment
region
```

---

# 13. Proposed Pipeline Architecture

```text
                         Vehicle Dataset
                               ↓
                    Separate X and target y
                               ↓
                     Group-based train-test split
                               ↓
                  ┌────────────┴────────────┐
                  │                         │
          Numerical features       Categorical features
                  │                         │
          Median imputation        Frequent-value imputation
                  │                         │
          StandardScaler             OneHotEncoder
                  │                         │
                  └────────────┬────────────┘
                               ↓
                       ColumnTransformer
                               ↓
                         SVM Classifier
                               ↓
                       GridSearchCV Tuning
                               ↓
                       Best SVM Model
                               ↓
                 Service-Priority Prediction
```

---

# 14. SVM Model Configuration

The initial model can use:

```python
SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)
```

## Parameter Meaning

| Parameter                 | Purpose                                              |
| ------------------------- | ---------------------------------------------------- |
| `kernel="rbf"`            | Creates nonlinear decision boundaries                |
| `C=1.0`                   | Controls penalty for classification errors           |
| `gamma="scale"`           | Controls the influence range of each training record |
| `class_weight="balanced"` | Gives more importance to minority classes            |
| `probability=True`        | Provides class probabilities                         |

---

# 15. Hyperparameter-Tuning Requirement

The team must compare different SVM configurations.

```python
parameter_grid = {
    "model__kernel": ["linear", "rbf"],
    "model__C": [0.1, 1, 10, 50, 100],
    "model__gamma": ["scale", 0.001, 0.01, 0.1, 1]
}
```

The model should be tuned using cross-validation.

Recommended scoring metric:

```python
scoring="f1_macro"
```

Macro F1 is preferred because it gives equal importance to:

* Normal
* Priority
* Critical

Weighted F1 may otherwise be dominated by the larger Normal class.

---

# 16. Model Evaluation Requirements

The model must not be evaluated only using accuracy.

The following metrics are required:

| Metric                | Purpose                                             |
| --------------------- | --------------------------------------------------- |
| Accuracy              | Overall percentage of correct predictions           |
| Precision             | Correctness of predicted classes                    |
| Recall                | Ability to identify all vehicles in a class         |
| F1-score              | Balance between precision and recall                |
| Macro F1              | Gives equal importance to every class               |
| Confusion matrix      | Shows exact classification errors                   |
| ROC-AUC               | Measures class-separation capability                |
| Critical-class recall | Measures how many critical vehicles were identified |

---

# 17. Most Important Evaluation Metric

The most important metric is:

```text
Recall for the Critical class
```

It is calculated as:

[
Critical\ Recall =
\frac{Correctly\ predicted\ Critical}
{Total\ actual\ Critical}
]

Suppose there are 200 actual Critical vehicles:

```text
Correctly identified Critical vehicles = 176
Missed Critical vehicles = 24
```

Then:

[
Critical\ Recall =
\frac{176}{200}
= 0.88
]

The model identifies 88% of critical vehicles.

---

# 18. Error-Cost Matrix

Different errors have different business impacts.

| Actual class | Predicted class | Business impact                 | Cost level     |
| ------------ | --------------- | ------------------------------- | -------------- |
| Critical     | Normal          | Breakdown or safety incident    | Extremely high |
| Critical     | Priority        | Required service may be delayed | High           |
| Priority     | Normal          | Possible maintenance delay      | Medium         |
| Normal       | Critical        | Unnecessary urgent inspection   | Low to medium  |
| Normal       | Priority        | Early but unnecessary service   | Low            |

Therefore, the project should prioritize reducing:

```text
Critical → Normal errors
```

---

# 19. Example Confusion Matrix

Suppose the model produces:

| Actual / Predicted | Normal | Priority | Critical |
| ------------------ | -----: | -------: | -------: |
| Normal             |  1,270 |       70 |       10 |
| Priority           |     95 |      380 |       25 |
| Critical           |      8 |       22 |      120 |

Interpretation:

* 1,270 Normal vehicles were correctly classified.
* 380 Priority vehicles were correctly classified.
* 120 Critical vehicles were correctly classified.
* 8 Critical vehicles were incorrectly classified as Normal.
* 22 Critical vehicles were incorrectly classified as Priority.

The eight Critical-to-Normal errors require detailed investigation.

---

# 20. Prediction Output Requirement

For every new vehicle, the system should generate:

| Output field           | Description                  |
| ---------------------- | ---------------------------- |
| `vehicle_id`           | Vehicle identifier           |
| `predicted_priority`   | Normal, Priority or Critical |
| `normal_probability`   | Probability of Normal        |
| `priority_probability` | Probability of Priority      |
| `critical_probability` | Probability of Critical      |
| `recommended_action`   | Suggested service action     |
| `prediction_date`      | Date of prediction           |

Example:

| Vehicle | Prediction | Normal | Priority | Critical | Action                     |
| ------- | ---------- | -----: | -------: | -------: | -------------------------- |
| VH5011  | Normal     |   0.91 |     0.08 |     0.01 | Regular service            |
| VH5012  | Priority   |   0.12 |     0.76 |     0.12 | Service within 7 days      |
| VH5013  | Critical   |   0.02 |     0.10 |     0.88 | Inspection within 24 hours |

---

# 21. Business Recommendation Rules

The model output can be converted into service actions.

| Prediction | Probability condition      | Recommended action                                          |
| ---------- | -------------------------- | ----------------------------------------------------------- |
| Normal     | Normal probability ≥ 70%   | Continue normal service schedule                            |
| Priority   | Priority probability ≥ 60% | Schedule inspection within seven days                       |
| Critical   | Critical probability ≥ 70% | Schedule inspection within 24 hours                         |
| Critical   | Critical probability ≥ 90% | Contact customer immediately and recommend vehicle stoppage |
| Uncertain  | Highest probability < 55%  | Send case for manual review                                 |

---

# 22. Example New Vehicle

A vehicle sends the following information:

| Feature             |      Value |
| ------------------- | ---------: |
| Mileage             | 118,000 km |
| Service gap         |  18 months |
| Engine alerts       |          8 |
| Brake wear          |        88% |
| Battery voltage     | 10.4 volts |
| Engine temperature  |      116°C |
| Fault codes         |          7 |
| Previous breakdowns |          3 |
| Customer complaints |          5 |
| Driving pattern     |       City |
| Vehicle age         |   10 years |

Possible model output:

```text
Predicted priority: Critical

Normal probability:   0.01
Priority probability: 0.08
Critical probability: 0.91
```

Recommended action:

```text
Contact the customer immediately.
Advise the customer not to continue long-distance driving.
Schedule a service appointment within 24 hours.
Reserve brake components and battery replacement stock.
Assign a senior diagnostic technician.
```

---

# 23. Model Success Criteria

The model will be accepted only when it meets the following conditions:

| Metric                       | Minimum target |
| ---------------------------- | -------------: |
| Overall accuracy             |            85% |
| Macro F1-score               |           0.82 |
| Critical-class recall        |            90% |
| Critical-class precision     |            80% |
| Critical predicted as Normal |       Below 3% |
| Priority-class recall        |            80% |

The most important condition is:

```text
At least 90% of Critical vehicles must be correctly identified.
```

---

# 24. Deployment Scenario

The trained SVM pipeline will be saved as:

```text
vehicle_service_priority_svm.joblib
```

The model will receive new vehicle data every day.

```text
Connected vehicle sensors
          ↓
Vehicle data platform
          ↓
Data validation
          ↓
Saved SVM pipeline
          ↓
Priority prediction
          ↓
Service management system
          ↓
Customer notification
          ↓
Workshop scheduling
```

---

# 25. Functional Requirements

The final system should:

1. Load historical vehicle records.
2. Validate numerical and categorical values.
3. Remove duplicate diagnostic snapshots.
4. Handle missing values.
5. encode categorical features.
6. scale numerical features.
7. perform group-based train-test splitting.
8. train an SVM classifier.
9. tune `C`, `gamma` and `kernel`.
10. evaluate all three classes.
11. display a confusion matrix.
12. calculate Critical-class recall.
13. predict new vehicle records.
14. generate class probabilities.
15. generate recommended actions.
16. save predictions in CSV format.
17. save the trained pipeline using Joblib.

---

# 26. Expected Project Files

```text
svm_vehicle_service_project/
│
├── data/
│   ├── vehicle_service_history.csv
│   └── new_vehicle_diagnostics.csv
│
├── src/
│   ├── data_validation.py
│   ├── train_svm.py
│   ├── evaluate_model.py
│   └── predict_service_priority.py
│
├── artifacts/
│   └── vehicle_service_priority_svm.joblib
│
├── outputs/
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   ├── model_comparison.csv
│   └── vehicle_priority_predictions.csv
│
├── reports/
│   └── svm_case_study_report.pdf
│
└── requirements.txt
```

---

# 27. Assignment Tasks

## Task 1: Data Understanding

* Display dataset shape.
* Display column information.
* Check class distribution.
* Identify numerical and categorical features.
* Check missing values.
* Check duplicate records.
* Identify invalid sensor values.

## Task 2: Exploratory Analysis

* Compare mileage across priority classes.
* Analyze service-gap distribution.
* Compare brake wear and engine alerts.
* Study battery voltage by target class.
* Create a correlation matrix for numerical features.

## Task 3: Data Preparation

* Remove identifiers from the model features.
* Handle missing numerical values.
* Handle missing categorical values.
* Encode categorical features.
* Scale numerical features.
* prevent leakage using vehicle-based splitting.

## Task 4: Baseline Model

Train an initial SVM using:

```python
kernel="rbf"
C=1.0
gamma="scale"
```

## Task 5: Hyperparameter Tuning

Use `GridSearchCV` to find the best:

* Kernel
* C
* Gamma

## Task 6: Evaluation

Calculate:

* Accuracy
* Precision
* Recall
* F1-score
* Macro F1
* Confusion matrix
* Critical-class recall

## Task 7: Error Analysis

Identify:

* Critical vehicles predicted as Normal
* Critical vehicles predicted as Priority
* Common characteristics of missed Critical vehicles
* Regions or vehicle segments with higher error rates

## Task 8: New Predictions

Predict service priority for new diagnostic records and generate business recommendations.

---

# 28. Final Case Study Question

> Develop an end-to-end machine-learning solution using Support Vector Machine to classify connected vehicles into Normal, Priority and Critical service categories. The solution must handle numerical and categorical features, missing values, class imbalance, feature scaling, nonlinear relationships and repeated vehicle records. The model must prioritize Critical-class recall, provide class probabilities, generate service recommendations and save the complete preprocessing and prediction pipeline for future use.

This problem is complex because it combines **multi-class classification, nonlinear patterns, imbalanced classes, mixed feature types, data leakage prevention, hyperparameter tuning and cost-sensitive business evaluation**.
