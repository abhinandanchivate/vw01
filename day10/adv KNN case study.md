# Advanced KNN Case Study

## Fleet Vehicle Breakdown-Severity and Service-Response Classification

## 1. Business scenario

A nationwide logistics company operates more than 20,000 commercial vehicles. Every vehicle continuously sends telematics, maintenance, driver-behaviour, environmental, and diagnostic data.

The company currently uses fixed rules to determine whether a vehicle should:

* Continue operating
* Be serviced soon
* Be removed from operation immediately

The existing rule-based system generates too many false alarms and sometimes fails to identify vehicles that later experience serious breakdowns.

The organization wants to build a **K-Nearest Neighbors classification model** that compares a vehicle with similar historical vehicles and predicts the required service-response category.

---

# 2. Clear problem statement

Build a multiclass KNN model that predicts the appropriate service response for a vehicle using its operating condition, diagnostic history, maintenance behaviour, environmental exposure, and driver behaviour.

The target variable is:

```text
service_response
```

The possible classes are:

| Class                   | Meaning                              | Required action           |
| ----------------------- | ------------------------------------ | ------------------------- |
| `Continue Operation`    | Vehicle is currently safe to operate | Continue normal operation |
| `Schedule Inspection`   | Warning signs are present            | Inspect within 7 days     |
| `Urgent Service`        | High risk of failure                 | Service within 24 hours   |
| `Remove from Operation` | Critical safety risk                 | Stop vehicle immediately  |

---

# 3. Business objective

The model should help the fleet-management team:

* Reduce unexpected breakdowns
* Avoid unnecessary vehicle stoppages
* Prioritize workshop capacity
* Reduce maintenance costs
* Improve driver and vehicle safety
* Identify vehicles similar to previously failed vehicles

---

# 4. Machine-learning objective

Develop and evaluate a KNN classification pipeline that:

1. Handles numerical and categorical data
2. Handles missing values
3. Scales numerical features
4. Encodes categorical features
5. Identifies and handles class imbalance
6. Selects the best value of `K`
7. Compares different distance metrics
8. Compares uniform and distance-based voting
9. Evaluates multiclass performance
10. Explains predictions using nearest historical vehicles

---

# 5. Dataset description

Assume the dataset contains approximately:

```text
12,000 historical vehicle records
```

Each row represents the condition of one vehicle at the time it was evaluated.

## Dataset name

```text
fleet_vehicle_service_response.csv
```

---

# 6. Dataset columns

## Vehicle information

| Column                      | Data type | Description               |
| --------------------------- | --------- | ------------------------- |
| `vehicle_id`                | String    | Unique vehicle identifier |
| `vehicle_type`              | Category  | Truck, Van, Bus or Pickup |
| `vehicle_age_years`         | Numeric   | Age of the vehicle        |
| `mileage_km`                | Numeric   | Total distance travelled  |
| `average_daily_distance_km` | Numeric   | Average daily distance    |
| `load_utilization_percent`  | Numeric   | Average vehicle load      |

## Maintenance information

| Column                       | Data type | Description                       |
| ---------------------------- | --------- | --------------------------------- |
| `months_since_last_service`  | Numeric   | Time since last scheduled service |
| `missed_service_count`       | Numeric   | Number of missed services         |
| `previous_breakdowns`        | Numeric   | Historical breakdown count        |
| `days_since_last_breakdown`  | Numeric   | Days since previous breakdown     |
| `warranty_claims`            | Numeric   | Number of warranty claims         |
| `maintenance_cost_last_year` | Numeric   | Total maintenance cost            |

## Diagnostic information

| Column                  | Data type | Description                 |
| ----------------------- | --------- | --------------------------- |
| `engine_alert_count`    | Numeric   | Number of engine warnings   |
| `brake_wear_percent`    | Numeric   | Estimated brake wear        |
| `tire_wear_percent`     | Numeric   | Estimated tire wear         |
| `battery_voltage`       | Numeric   | Battery voltage             |
| `engine_temperature_c`  | Numeric   | Current engine temperature  |
| `oil_pressure_psi`      | Numeric   | Current engine-oil pressure |
| `coolant_level_percent` | Numeric   | Remaining coolant level     |
| `vibration_score`       | Numeric   | Engine vibration severity   |
| `fuel_efficiency_kmpl`  | Numeric   | Recent fuel efficiency      |

## Driver and operating conditions

| Column                      | Data type | Description                      |
| --------------------------- | --------- | -------------------------------- |
| `harsh_braking_events`      | Numeric   | Harsh braking events per month   |
| `rapid_acceleration_events` | Numeric   | Rapid acceleration events        |
| `average_speed_kmph`        | Numeric   | Average driving speed            |
| `driver_experience_years`   | Numeric   | Driver experience                |
| `route_type`                | Category  | City, Highway, Mixed or Mountain |
| `driving_shift`             | Category  | Day, Night or Rotational         |

## Environmental information

| Column                  | Data type | Description                    |
| ----------------------- | --------- | ------------------------------ |
| `average_temperature_c` | Numeric   | Average route temperature      |
| `rain_exposure_days`    | Numeric   | Rain exposure during the month |
| `dust_exposure_level`   | Category  | Low, Medium or High            |
| `road_condition`        | Category  | Good, Moderate or Poor         |

## Target column

| Column             | Data type | Description                       |
| ------------------ | --------- | --------------------------------- |
| `service_response` | Category  | Required vehicle-service response |

---

# 7. Sample records

| Mileage | Service gap | Engine alerts | Brake wear | Battery voltage | Vibration | Road condition | Response              |
| ------: | ----------: | ------------: | ---------: | --------------: | --------: | -------------- | --------------------- |
|  32,000 |           3 |             0 |         22 |            14.1 |       1.2 | Good           | Continue Operation    |
|  78,000 |          11 |             3 |         58 |            12.4 |       3.8 | Moderate       | Schedule Inspection   |
| 126,000 |          18 |             7 |         82 |            11.1 |       7.2 | Poor           | Urgent Service        |
| 164,000 |          25 |            10 |         96 |             9.8 |       9.4 | Poor           | Remove from Operation |

---

# 8. Data-quality challenges

The dataset deliberately contains the following problems:

* Missing battery-voltage values
* Missing driver-experience values
* Duplicate vehicle records
* Extreme mileage values
* Incorrect negative maintenance costs
* Inconsistent category names such as:

```text
Highway
highway
HIGHWAY
```

* Class imbalance
* Features with very different numerical ranges
* Irrelevant identifier columns
* Correlated features
* Previously unseen categories in new data

---

# 9. Expected class distribution

| Class                 | Approximate records | Percentage |
| --------------------- | ------------------: | ---------: |
| Continue Operation    |               6,000 |        50% |
| Schedule Inspection   |               3,600 |        30% |
| Urgent Service        |               1,800 |        15% |
| Remove from Operation |                 600 |         5% |

The critical class is highly underrepresented.

---

# 10. Detailed task list

## Task 1: Understand the business requirement

Explain:

* What the model will predict
* Who will use the prediction
* Why incorrect predictions are costly
* Which class is most important
* What business action corresponds to each class

---

## Task 2: Load and inspect the dataset

Perform the following checks:

```text
Number of rows
Number of columns
Column names
Data types
First and last records
Summary statistics
Unique target classes
Class distribution
```

Expected methods may include:

```python
head()
tail()
info()
describe()
value_counts()
nunique()
```

---

## Task 3: Validate the dataset

Check for:

* Missing values
* Duplicate rows
* Invalid negative values
* Impossible percentages
* Incorrect battery voltage
* Inconsistent categorical values
* Unexpected target labels

Example validation rules:

```text
0 <= brake_wear_percent <= 100
0 <= tire_wear_percent <= 100
0 <= coolant_level_percent <= 100
battery_voltage > 0
mileage_km >= 0
maintenance_cost_last_year >= 0
```

---

## Task 4: Clean the data

Required cleaning activities:

1. Remove duplicate records.
2. Standardize categorical text.
3. Replace impossible numerical values.
4. Handle missing values.
5. Remove rows with missing target values.
6. Remove `vehicle_id` from model inputs.

Use appropriate strategies such as:

* Median imputation for numerical features
* Most-frequent imputation for categorical features
* Capping or removal of extreme values

---

## Task 5: Perform exploratory data analysis

Analyze:

* Class distribution
* Mileage distribution
* Service-gap distribution
* Brake-wear distribution
* Engine-alert distribution
* Breakdown history by response category
* Relationship between mileage and brake wear
* Relationship between service gap and engine alerts
* Response category by vehicle type
* Response category by route type
* Correlation among numerical features

Required visualizations:

* Histograms
* Box plots
* Bar charts
* Scatter plots
* Correlation heatmap

---

## Task 6: Separate features and target

Create:

```python
X = input features
y = service_response
```

Exclude:

```text
vehicle_id
service_response
```

Explain why the identifier must not be used as a feature.

---

## Task 7: Split the dataset

Create training and testing datasets using:

```text
80% training data
20% testing data
```

Requirements:

* Use `random_state=42`
* Use `stratify=y`
* Verify class distribution after splitting

Explain why stratification is important for the `Remove from Operation` class.

---

## Task 8: Build the preprocessing pipeline

### Numerical pipeline

The numerical pipeline must include:

```text
Median imputation
Standard scaling
```

### Categorical pipeline

The categorical pipeline must include:

```text
Most-frequent imputation
One-hot encoding
```

Use:

```python
ColumnTransformer
Pipeline
SimpleImputer
StandardScaler
OneHotEncoder
```

The final pipeline should follow:

```text
Raw input
   ↓
Missing-value treatment
   ↓
Numerical scaling
   ↓
Categorical encoding
   ↓
KNN classifier
```

---

## Task 9: Explain why scaling is essential

Demonstrate the problem using two features:

```text
Mileage = 120,000
Engine alerts = 7
```

Without scaling, mileage can dominate the distance calculation.

Compare KNN performance:

1. Without scaling
2. With `StandardScaler`

Record the difference in validation performance.

---

## Task 10: Build a baseline KNN model

Create an initial model using:

```python
KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",
    metric="euclidean"
)
```

Evaluate the baseline model using:

* Accuracy
* Precision
* Recall
* F1 score
* Confusion matrix
* Classification report

---

## Task 11: Find the best value of K

Test at least the following values:

```python
k_values = [
    1, 3, 5, 7, 9,
    11, 15, 21, 25, 31
]
```

For every K value, calculate:

* Training accuracy
* Validation accuracy
* Macro F1 score

Create a graph:

```text
K value versus validation score
```

Identify:

* The K value that overfits
* The K value that underfits
* The most balanced K value

---

## Task 12: Perform GridSearchCV

Tune the following parameters:

```python
parameter_grid = {
    "knn__n_neighbors": [
        3, 5, 7, 9, 11,
        15, 21, 25
    ],

    "knn__weights": [
        "uniform",
        "distance"
    ],

    "knn__metric": [
        "euclidean",
        "manhattan",
        "minkowski"
    ],

    "knn__p": [
        1, 2
    ]
}
```

Use:

```python
GridSearchCV(
    cv=5,
    scoring="f1_macro"
)
```

Report:

* Best parameters
* Best cross-validation score
* Number of parameter combinations
* Total number of model fits

---

## Task 13: Compare distance metrics

Compare the following distance methods:

### Euclidean distance

[
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
]

### Manhattan distance

[
d(x,y)=\sum_{i=1}^{n}|x_i-y_i|
]

### Minkowski distance

[
d(x,y)=
\left(
\sum_{i=1}^{n}|x_i-y_i|^p
\right)^{1/p}
]

Create a comparison table:

| Metric | Best K | Accuracy | Macro F1 | Critical-class recall |
| ------ | -----: | -------: | -------: | --------------------: |

---

## Task 14: Handle class imbalance

Compare at least two approaches.

### Approach A

Use the original imbalanced training dataset.

### Approach B

Apply oversampling only to the training data.

Possible techniques:

* Random oversampling
* SMOTE, after confirming compatibility with processed features

Do not apply oversampling to the test data.

Compare:

* Overall accuracy
* Macro F1
* Recall for `Urgent Service`
* Recall for `Remove from Operation`

---

## Task 15: Evaluate the final model

Evaluate the selected model on the untouched test dataset.

Required metrics:

* Accuracy
* Macro precision
* Macro recall
* Macro F1
* Weighted F1
* Per-class precision
* Per-class recall
* Per-class F1
* Confusion matrix

The most important metric is:

```text
Recall for Remove from Operation
```

This measures how many truly critical vehicles were correctly identified.

---

## Task 16: Analyse model errors

Identify:

* Critical vehicles predicted as safe
* Safe vehicles predicted as critical
* Urgent vehicles predicted as inspection cases
* Frequently confused classes

Create an error-analysis report containing:

```text
Actual class
Predicted class
Vehicle attributes
Prediction confidence
Possible reason for error
```

---

## Task 17: Generate probability predictions

Use:

```python
predict_proba()
```

For every vehicle, produce probabilities such as:

| Vehicle | Continue | Inspection | Urgent | Remove |
| ------- | -------: | ---------: | -----: | -----: |
| V1001   |     0.05 |       0.15 |   0.25 |   0.55 |

Calculate:

```text
Prediction confidence = highest class probability
```

Flag low-confidence predictions:

```text
Prediction confidence < 0.60
```

Such vehicles should be sent for manual review.

---

## Task 18: Explain predictions using nearest neighbours

For selected test vehicles:

1. Transform the record using the preprocessing pipeline.
2. Use `kneighbors()` to retrieve the nearest historical records.
3. Display the neighbour distances.
4. Display the classes of those neighbours.
5. Explain how the final prediction was produced.

Example:

| Neighbour | Distance | Historical class      |
| --------: | -------: | --------------------- |
|         1 |     0.62 | Remove from Operation |
|         2 |     0.78 | Urgent Service        |
|         3 |     0.85 | Remove from Operation |
|         4 |     0.91 | Remove from Operation |
|         5 |     1.05 | Urgent Service        |

Prediction:

```text
Remove from Operation
```

---

## Task 19: Create new vehicle predictions

Prepare at least five new records representing:

1. Healthy low-mileage vehicle
2. Moderately risky vehicle
3. High-mileage poorly maintained vehicle
4. Critical brake and engine condition
5. Unknown-category scenario

The prediction output must contain:

| Field                        | Description            |
| ---------------------------- | ---------------------- |
| `vehicle_id`                 | New vehicle identifier |
| `predicted_service_response` | Predicted category     |
| `prediction_confidence`      | Highest probability    |
| `critical_risk_probability`  | Probability of removal |
| `recommended_action`         | Business response      |
| `manual_review_required`     | Yes or No              |

---

## Task 20: Add business-action mapping

Use the following mapping:

```python
action_mapping = {
    "Continue Operation":
        "Continue normal operation",

    "Schedule Inspection":
        "Schedule inspection within 7 days",

    "Urgent Service":
        "Arrange service within 24 hours",

    "Remove from Operation":
        "Stop vehicle and arrange immediate inspection"
}
```

---

## Task 21: Save the model and preprocessing pipeline

Save the complete fitted pipeline using:

```python
joblib
```

Expected artifact:

```text
artifacts/
└── knn_vehicle_service_pipeline.joblib
```

The saved artifact must contain:

* Imputation logic
* Scaling logic
* Encoding logic
* Trained KNN model

---

## Task 22: Build a separate prediction script

Create a script that:

1. Loads the saved pipeline.
2. Reads new vehicle data from CSV.
3. Validates required columns.
4. Generates predictions.
5. Generates class probabilities.
6. Adds business actions.
7. Saves the final report.

Expected files:

```text
data/new_vehicle_input.csv
outputs/new_vehicle_predictions.csv
```

---

## Task 23: Measure prediction performance

Record prediction time for:

```text
100 records
1,000 records
5,000 records
10,000 records
```

Explain why KNN prediction becomes slower as the training dataset becomes larger.

Unlike many algorithms, KNN must retain and search the training data during prediction.

---

## Task 24: Investigate dimensionality

Compare model performance using:

1. All features
2. Selected important features
3. PCA-transformed features

Measure:

* Accuracy
* Macro F1
* Prediction time
* Number of dimensions

Explain the **curse of dimensionality** in KNN.

---

## Task 25: Prepare the final report

The final report must include:

* Business problem
* Dataset summary
* Data-quality findings
* Exploratory analysis
* Preprocessing decisions
* Baseline-model results
* Hyperparameter-tuning results
* Final-model results
* Confusion matrix
* Error analysis
* Business recommendations
* Model limitations
* Deployment considerations

---

# 11. Required deliverables

```text
knn_fleet_case_study/
│
├── README.md
│
├── case_study/
│   └── knn_fleet_problem_statement.md
│
├── data/
│   ├── fleet_vehicle_service_response.csv
│   └── new_vehicle_input.csv
│
├── notebooks/
│   └── knn_analysis.ipynb
│
├── src/
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict_new_vehicles.py
│
├── artifacts/
│   └── knn_vehicle_service_pipeline.joblib
│
├── outputs/
│   ├── model_comparison.csv
│   ├── confusion_matrix.png
│   ├── k_value_comparison.png
│   ├── error_analysis.csv
│   └── new_vehicle_predictions.csv
│
└── reports/
    └── final_knn_report.pdf
```

---

# 12. Acceptance criteria

The completed solution should satisfy the following requirements:

| Requirement       | Acceptance condition                                |
| ----------------- | --------------------------------------------------- |
| Pipeline          | Preprocessing and KNN are combined                  |
| Data leakage      | No preprocessing is fitted on test data             |
| Scaling           | Numerical features are scaled                       |
| Encoding          | Categorical features are one-hot encoded            |
| Missing values    | Missing values are handled                          |
| Tuning            | GridSearchCV is implemented                         |
| Evaluation        | Multiclass metrics are reported                     |
| Critical recall   | `Remove from Operation` recall is clearly reported  |
| Explainability    | Nearest neighbours are shown for sample predictions |
| Prediction report | Probabilities and recommended actions are included  |
| Persistence       | Trained pipeline is saved using joblib              |
| Reusability       | Separate prediction script works on new CSV data    |

---

# 13. Main technical challenge

The goal is not simply to produce high accuracy.

A model that predicts most vehicles as `Continue Operation` may obtain acceptable accuracy while failing to detect critical vehicles.

The final model should balance:

```text
Overall performance
        +
Critical-class recall
        +
False-alarm reduction
        +
Prediction speed
        +
Business usability
```

---

# 14. Expected learner outcome

After completing the case study, the learner should be able to:

* Explain how KNN performs multiclass classification
* Explain the importance of feature scaling
* Select an appropriate K value
* Compare distance metrics
* Use distance-weighted voting
* Tune KNN using GridSearchCV
* Evaluate imbalanced multiclass predictions
* Explain predictions using nearest records
* Build a reusable Scikit-learn pipeline
* Generate business-ready prediction reports
