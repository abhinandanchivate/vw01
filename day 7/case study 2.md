# EDA Case Study: Vehicle Urgent-Service Risk Prediction

## 1. Business problem

A connected-car service company wants to predict whether a vehicle will need **urgent service within the next 30 days**.

Before training an ML model, the team must perform **Exploratory Data Analysis** to answer:

* Is the dataset complete and reliable?
* Which features are related to service risk?
* Are there missing values, duplicates, or outliers?
* Is the target class balanced?
* Which variables should be used for model training?

> This case study uses a synthetic 3,000-record automobile dataset created for training purposes.

---

# 2. Dataset features

| Feature               | Description                   | Type        |
| --------------------- | ----------------------------- | ----------- |
| `vehicle_id`          | Unique vehicle identifier     | Identifier  |
| `vehicle_age_years`   | Vehicle age in years          | Numeric     |
| `mileage_km`          | Total kilometres travelled    | Numeric     |
| `avg_monthly_km`      | Average monthly kilometres    | Numeric     |
| `service_gap_months`  | Months since the last service | Numeric     |
| `engine_alerts`       | Number of engine warnings     | Numeric     |
| `brake_wear_percent`  | Brake wear percentage         | Numeric     |
| `battery_voltage`     | Current battery voltage       | Numeric     |
| `battery_health`      | Good, Normal, or Low          | Categorical |
| `warranty_claims`     | Previous warranty claims      | Numeric     |
| `customer_complaints` | Number of complaints          | Numeric     |
| `driving_pattern`     | City, Highway, or Mixed       | Categorical |
| `target`              | High Risk or Low Risk         | Target      |

---

# 3. EDA workflow

```text
Business problem
       ↓
Load or create data
       ↓
Understand rows and columns
       ↓
Find duplicates and missing values
       ↓
Univariate analysis
       ↓
Bivariate analysis
       ↓
Multivariate analysis
       ↓
Outlier detection
       ↓
Feature engineering
       ↓
Prepare clean ML dataset
       ↓
Business conclusions
```

---


---

# 5. Step-by-step explanation

## Step 1: Define the business objective

The target is:

```text
High Risk = vehicle may require urgent service
Low Risk  = vehicle is unlikely to require urgent service
```

EDA is not performed only to draw charts. It must help answer business questions.

| Business question                      | EDA analysis                 |
| -------------------------------------- | ---------------------------- |
| Do older vehicles have greater risk?   | Age versus target            |
| Does mileage affect brake wear?        | Mileage versus brake wear    |
| Do engine alerts indicate risk?        | Engine alerts versus target  |
| Does low battery health increase risk? | Battery health versus target |
| Is the service gap important?          | Service gap versus target    |
| Are high-risk vehicles uncommon?       | Target-distribution analysis |

---

## Step 2: Understand the dataset structure

Use:

```python
data.shape
data.head()
data.columns
data.dtypes
data.info()
```

### Expected observation

Before cleaning:

```text
Rows:    3005
Columns: 13
```

There are 3,000 original records and five intentionally duplicated records.

---

## Step 3: Examine descriptive statistics

Use:

```python
data.describe().T
```

This gives:

* Count
* Mean
* Standard deviation
* Minimum
* 25th percentile
* Median
* 75th percentile
* Maximum

### Example interpretation

Suppose the mileage maximum is unusually large compared with the 75th percentile. That may indicate:

* A genuinely high-mileage commercial vehicle
* A data-entry error
* An incorrect unit
* An outlier requiring investigation

EDA identifies the issue but does not automatically prove that the record is incorrect.

---

## Step 4: Check missing values

The code inserts 60 missing values into each of these columns:

| Column               | Missing values |
| -------------------- | -------------: |
| `service_gap_months` |             60 |
| `brake_wear_percent` |             60 |
| `battery_health`     |             60 |
| `driving_pattern`    |             60 |

### Missing-value solution

Numeric columns are filled with the **median**:

```python
cleaned_data[column] = cleaned_data[column].fillna(
    cleaned_data[column].median()
)
```

Categorical columns are filled with the **mode**:

```python
cleaned_data[column] = cleaned_data[column].fillna(
    cleaned_data[column].mode()[0]
)
```

### Why use the median?

The mean can be strongly influenced by extreme values.

Example:

```text
Mileage = 20,000, 30,000, 40,000, 50,000, 500,000
```

The 500,000 value greatly increases the mean, but the median remains more representative.

---

## Step 5: Remove duplicate records

Use:

```python
cleaned_data = data.drop_duplicates().copy()
```

### Expected result

| Measurement                  | Count |
| ---------------------------- | ----: |
| Rows before cleaning         | 3,005 |
| Duplicate rows               |     5 |
| Rows after duplicate removal | 3,000 |

Duplicates can cause the ML model to give excessive importance to repeated records.

---

# 6. Univariate EDA solution

Univariate EDA studies one variable at a time.

## Mileage analysis

```python
sns.histplot(
    data=cleaned_data,
    x="mileage_km",
    kde=True
)
```

Questions answered:

* Where are most mileage values concentrated?
* Is the distribution symmetric or skewed?
* Are extremely high-mileage values present?

## Box-plot analysis

```python
sns.boxplot(
    data=cleaned_data,
    x="mileage_km"
)
```

The box plot displays:

* Median
* Lower quartile
* Upper quartile
* Spread
* Potential outliers

## Target analysis

The generated dataset normally produces approximately:

| Target    | Approximate percentage |
| --------- | ---------------------: |
| Low Risk  |                  63.5% |
| High Risk |                  36.5% |

This is moderately imbalanced but not extremely imbalanced.

A 99% versus 1% distribution would require more careful class-imbalance treatment.

---

# 7. Bivariate EDA solution

Bivariate EDA studies two variables together.

## Average values by risk group

With the fixed random seed, the results will be approximately:

| Feature             |   High Risk |    Low Risk |
| ------------------- | ----------: | ----------: |
| Mileage             |  116,686 km |   58,913 km |
| Service gap         | 8.91 months | 7.17 months |
| Engine alerts       |        2.27 |        1.05 |
| Brake wear          |      74.48% |      42.73% |
| Battery voltage     |     12.19 V |     12.51 V |
| Warranty claims     |        1.23 |        0.63 |
| Customer complaints |        0.69 |        0.34 |

## Interpretation

High-risk vehicles generally have:

* Nearly twice the mileage
* Greater brake wear
* More engine alerts
* Longer service gaps
* Lower battery voltage
* More warranty claims
* More customer complaints

These findings suggest that the features may be useful for ML model training.

---

## Battery-health analysis

The approximate high-risk percentages are:

| Battery health | High-risk percentage |
| -------------- | -------------------: |
| Good           |                21.7% |
| Normal         |                55.1% |
| Low            |                89.2% |

### Interpretation

Low battery health has a strong relationship with urgent-service risk.

However:

> EDA shows an association. It does not prove that low battery health alone causes urgent service.

---

## Driving-pattern analysis

Approximate high-risk percentages:

| Driving pattern | High-risk percentage |
| --------------- | -------------------: |
| City            |                37.4% |
| Highway         |                35.6% |
| Mixed           |                35.9% |

### Interpretation

Driving pattern has only a small relationship with risk in this dataset.

It may still help when combined with other variables, but by itself it is not a strong separator.

---

# 8. Multivariate EDA solution

Multivariate EDA studies several variables together.

## Correlation heatmap

```python
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    center=0
)
```

Approximate correlations with the high-risk target:

| Feature             | Correlation |
| ------------------- | ----------: |
| Mileage             |        0.62 |
| Brake wear          |        0.61 |
| Vehicle age         |        0.60 |
| Engine alerts       |        0.42 |
| Warranty claims     |        0.28 |
| Service gap         |        0.27 |
| Customer complaints |        0.23 |
| Average monthly km  |        0.01 |
| Battery voltage     |       -0.45 |

## Interpretation

### Positive correlation

Mileage has a positive correlation with high risk:

```text
Mileage increases → urgent-service risk generally increases
```

### Negative correlation

Battery voltage has a negative correlation:

```text
Battery voltage decreases → urgent-service risk generally increases
```

### Weak relationship

Average monthly kilometres has almost no linear relationship with the target in this generated dataset.

---

## Multicollinearity observation

Mileage, vehicle age, and brake wear may also be strongly correlated with one another.

```text
Vehicle age
    ↓
Higher mileage
    ↓
Greater brake wear
```

This matters because some ML models, especially linear models, can be affected when several features carry nearly the same information.

Possible solutions include:

* Remove one highly redundant feature
* Apply regularization
* Use PCA
* Use tree-based models
* Keep all features but evaluate feature importance carefully

---

# 9. Outlier-detection solution

The code uses the IQR method.

[
IQR=Q_3-Q_1
]

Potential outliers are outside:

[
Q_1-1.5(IQR)
]

and

[
Q_3+1.5(IQR)
]

### Code

```python
Q1 = cleaned_data["mileage_km"].quantile(0.25)
Q3 = cleaned_data["mileage_km"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```

## Important decision

Do not automatically delete every outlier.

A mileage value of 250,000 km may be valid for:

* A taxi
* A fleet vehicle
* A delivery vehicle
* A commercial vehicle

The correct process is:

```text
Detect outlier
     ↓
Check business validity
     ↓
Check units and source system
     ↓
Correct, cap, retain, or remove
```

In the case study, the original mileage is retained, and a capped feature is created:

```python
cleaned_data["mileage_km_capped"] = (
    cleaned_data["mileage_km"]
    .clip(lower=lower_bound, upper=upper_bound)
)
```

---

# 10. Feature-engineering solution

## Mileage category

```text
0–30,000 km       → Low
30,001–60,000 km  → Medium
60,001–100,000 km → High
Above 100,000 km  → Very High
```

## Service-gap category

```text
Up to 6 months → On Time
6–12 months    → Delayed
Above 12 months → Highly Delayed
```

## Maintenance-warning count

One point is added for each condition:

* Engine alerts greater than 2
* Brake wear above 75%
* Low battery health
* Service gap above 12 months

Example:

| Engine alerts | Brake wear | Battery | Service gap | Warning count |
| ------------: | ---------: | ------- | ----------: | ------------: |
|             1 |        40% | Good    |    4 months |             0 |
|             4 |        82% | Normal  |    9 months |             2 |
|             5 |        88% | Low     |   15 months |             4 |

This engineered feature combines multiple maintenance signals.

---

# 11. Prepare data for machine learning

## Remove identifier

```python
ml_data = cleaned_data.drop(
    columns=["vehicle_id"]
)
```

Vehicle ID normally does not help predict risk and may cause memorization.

## Convert target

```python
ml_data["target"] = ml_data["target"].map({
    "Low Risk": 0,
    "High Risk": 1
})
```

## Encode categorical values

```python
ml_data = pd.get_dummies(
    ml_data,
    columns=[
        "battery_health",
        "driving_pattern",
        "mileage_category",
        "service_gap_category"
    ],
    drop_first=True,
    dtype=int
)
```

Example conversion:

| Original driving pattern | City | Mixed |
| ------------------------ | ---: | ----: |
| Highway                  |    0 |     0 |
| City                     |    1 |     0 |
| Mixed                    |    0 |     1 |

Highway becomes the reference category because `drop_first=True`.

---

# 12. Final EDA findings

## Data-quality findings

| Finding                    | Solution                                    |
| -------------------------- | ------------------------------------------- |
| Five duplicate records     | Removed duplicates                          |
| Missing numeric values     | Filled using median                         |
| Missing categorical values | Filled using mode                           |
| Unusual mileage values     | Flagged with IQR and created capped feature |
| Categorical values         | Converted using one-hot encoding            |
| Target text values         | Converted into 0 and 1                      |

## Business findings

| Finding                                      | Meaning                                                   |
| -------------------------------------------- | --------------------------------------------------------- |
| High mileage is strongly related to risk     | Older, heavily used vehicles need closer monitoring       |
| Brake wear is strongly related to risk       | Brake condition is an important maintenance indicator     |
| Engine alerts increase with risk             | Warning counts should be monitored continuously           |
| Low battery health has very high risk        | Battery inspection should be prioritized                  |
| Longer service gaps increase risk            | Delayed servicing contributes to maintenance risk         |
| Driving pattern has a weak individual effect | It may be useful only in combination with other variables |
| Average monthly km has little relationship   | It may provide limited predictive value in this dataset   |

---

# 13. Recommended business actions

| Condition                   | Recommended action                |
| --------------------------- | --------------------------------- |
| Low battery health          | Schedule battery inspection       |
| Brake wear above 75%        | Arrange brake-service appointment |
| More than two engine alerts | Perform diagnostic scan           |
| Service gap above 12 months | Send urgent service reminder      |
| Multiple warning conditions | Contact the customer immediately  |
| High predicted risk         | Reserve a priority service slot   |

---

# 14. Final outcome

After completing EDA, the organization has:

* A clean 3,000-record dataset
* No duplicate records
* No missing values
* Identified and managed outliers
* Encoded categorical features
* Created useful maintenance features
* Identified important risk indicators
* Produced an ML-ready CSV file

The next ML step would be:

```text
Train-test split
      ↓
Feature scaling where required
      ↓
Train classification models
      ↓
Evaluate confusion matrix, precision, recall, F1, ROC, and AUC
```
