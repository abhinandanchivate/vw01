## Case Study 1: Automobile Service Risk Prediction — **High Bias / Underfitting**

### 1. Business Problem

A car company wants to predict whether a vehicle will have **High Service Risk** in the next 30 days.

Example business use:

> Send early service alerts to customers before breakdown happens.

---

## 2. Dataset

| Feature               | Meaning                   | Example              |
| --------------------- | ------------------------- | -------------------- |
| `mileage_km`          | Total running of vehicle  | 52,000               |
| `service_gap_months`  | Months since last service | 15                   |
| `engine_alerts`       | Engine warning count      | 6                    |
| `brake_wear_percent`  | Brake wear percentage     | 84                   |
| `battery_voltage`     | Battery condition         | Low / Normal         |
| `customer_complaints` | Complaints raised         | 3                    |
| `target`              | Output                    | High Risk / Low Risk |

---

## 3. Initial Model

The team uses a very simple Decision Tree:

```python
DecisionTreeClassifier(max_depth=1)
```

This means the tree can ask only **one main question**.

---

## 4. Visual: Very Simple Tree

```text
                 Is mileage_km > 50,000?
                    /                 \
                 Yes                   No
              High Risk              Low Risk
```

This tree only checks mileage.

But real automobile risk depends on many things:

```text
mileage + engine alerts + brake wear + battery + service gap + complaints
```

So this model is too simple.

---

## 5. Model Result

| Model                       | Training Accuracy | Testing Accuracy | Result                   |
| --------------------------- | ----------------: | ---------------: | ------------------------ |
| Decision Tree `max_depth=1` |               65% |              63% | High Bias / Underfitting |

---

## 6. Diagnosis

Both training and testing accuracy are low.

| Observation                       | Meaning                                   |
| --------------------------------- | ----------------------------------------- |
| Training accuracy is low          | Model cannot learn training data properly |
| Testing accuracy is also low      | Model cannot perform well on new data     |
| Train and test accuracy are close | Not overfitting                           |
| Both are poor                     | Underfitting / High Bias                  |

---

## 7. Why Bias Happened?

| Problem                    | Explanation                                              |
| -------------------------- | -------------------------------------------------------- |
| Model is too simple        | Only one decision rule is used                           |
| Important features ignored | Engine alerts, brake wear, battery are not properly used |
| Tree depth is too low      | `max_depth=1` cannot learn complex vehicle risk          |
| Strong assumption          | Model assumes mileage alone decides risk                 |

---

## 8. Real Example

| Vehicle | Mileage | Engine Alerts | Brake Wear | Battery | Actual Risk | Simple Model Prediction |
| ------- | ------: | ------------: | ---------: | ------- | ----------- | ----------------------- |
| Car A   |  60,000 |             0 |        25% | Normal  | Low Risk    | High Risk               |
| Car B   |  30,000 |             8 |        90% | Low     | High Risk   | Low Risk                |

### Problem

The model says:

```text
High mileage = High Risk
Low mileage = Low Risk
```

But this is not always true.

Car B has low mileage but many danger signals.

So the model gives wrong prediction.

---

## 9. Solution Steps

### Step 1: Increase Model Complexity

Use:

```python
DecisionTreeClassifier(max_depth=3)
```

Instead of:

```python
DecisionTreeClassifier(max_depth=1)
```

---

### Step 2: Add More Useful Features

| Existing Feature | Additional Useful Feature |
| ---------------- | ------------------------- |
| Mileage          | Average daily running     |
| Service gap      | Last service quality      |
| Engine alerts    | Alert severity            |
| Brake wear       | Brake replacement history |
| Battery voltage  | Battery age               |
| Complaints       | Complaint type            |

---

### Step 3: Improve Feature Engineering

Create new useful columns:

| New Feature          | Meaning                    |
| -------------------- | -------------------------- |
| `vehicle_age_years`  | Age of vehicle             |
| `risk_score`         | Combined risk score        |
| `alert_per_month`    | Engine alerts frequency    |
| `service_delay_flag` | Whether service is overdue |

---

### Step 4: Try Better Algorithms

| Algorithm                       | Why Useful          |
| ------------------------------- | ------------------- |
| Decision Tree with proper depth | Easy to explain     |
| Random Forest                   | Reduces overfitting |
| Gradient Boosting               | Better accuracy     |
| Logistic Regression             | Good baseline       |

---

## 10. Improved Model Result

| Model            | Training Accuracy | Testing Accuracy | Result           |
| ---------------- | ----------------: | ---------------: | ---------------- |
| `max_depth=1`    |               65% |              63% | Underfitting     |
| `max_depth=3`    |               88% |              85% | Good Fit         |
| `max_depth=None` |               99% |              72% | Overfitting risk |

---

## 11. Visual: Model Improvement

```text
Accuracy
100 |                         Train: 99%
 90 |              Train: 88%
 80 |              Test : 85%
 70 |                         Test : 72%
 60 | Train: 65%
    | Test : 63%
    +-----------------------------------------
        max_depth=1     max_depth=3     max_depth=None
        Underfit        Good Fit        Overfit
```

---

## 12. Final Solution

Best solution:

```python
DecisionTreeClassifier(max_depth=3, random_state=42)
```

Why?

| Reason              | Explanation                           |
| ------------------- | ------------------------------------- |
| Not too simple      | Can check multiple vehicle conditions |
| Not too complex     | Does not memorize every record        |
| Good train accuracy | Learns useful patterns                |
| Good test accuracy  | Works well on new vehicles            |

---

# Case Study 2: Automobile Warranty Claim Prediction — **Overfitting / High Variance**

---

## 1. Business Problem

An automobile company wants to predict whether a vehicle may raise a **warranty claim** in the next 3 months.

Example:

> Predict which vehicles may have warranty issues so that the company can plan service support and spare parts.

---

## 2. Dataset

| Feature                  | Meaning                    | Example          |
| ------------------------ | -------------------------- | ---------------- |
| `vehicle_age_months`     | Age of vehicle             | 28               |
| `mileage_km`             | Total running              | 42,000           |
| `engine_temperature_avg` | Average engine temperature | 96°C             |
| `service_count`          | Number of services done    | 4                |
| `part_replacement_count` | Previous part replacements | 2                |
| `driving_pattern`        | City / Highway / Mixed     | City             |
| `manufacturing_batch`    | Batch number               | B102             |
| `target`                 | Output                     | Claim / No Claim |

---

## 3. First Model: Too Complex

The team uses:

```python
DecisionTreeClassifier(max_depth=None)
```

This means the tree can grow fully.

It can keep splitting until it memorizes small details.

---

## 4. Visual: Overfitting Tree

```text
Vehicle age > 24 months?
│
├── Yes
│   ├── Mileage > 40,000?
│   │   ├── Engine temp > 95?
│   │   │   ├── Batch = B102?
│   │   │   │   ├── Service count = 4?
│   │   │   │   │   └── Claim
│
└── No
    └── No Claim
```

This tree is too deep.

It may learn unnecessary patterns like:

```text
Batch B102 + service count 4 + mileage 42,000 = Claim
```

But this may be true only for a few training records.

---

## 5. Model Result

| Model                          | Training Accuracy | Testing Accuracy | Result                      |
| ------------------------------ | ----------------: | ---------------: | --------------------------- |
| Decision Tree `max_depth=None` |               99% |              72% | Overfitting / High Variance |

---

## 6. Diagnosis

| Observation                    | Meaning                       |
| ------------------------------ | ----------------------------- |
| Training accuracy is very high | Model memorized training data |
| Testing accuracy is much lower | Model failed on new data      |
| Big gap between train and test | Overfitting                   |
| Model too complex              | High variance                 |

---

## 7. Why Overfitting Happened?

| Problem              | Explanation                             |
| -------------------- | --------------------------------------- |
| Tree is too deep     | It creates too many rules               |
| Learns noise         | Learns random patterns in training data |
| Memorizes rare cases | Works only for old examples             |
| Poor generalization  | Fails on new vehicles                   |

---

## 8. Real Example

Training data:

| Vehicle | Mileage | Batch | Engine Temp | Service Count | Actual |
| ------- | ------: | ----- | ----------: | ------------: | ------ |
| V1      |  42,000 | B102  |          96 |             4 | Claim  |
| V2      |  42,500 | B102  |          97 |             4 | Claim  |
| V3      |  43,000 | B102  |          96 |             4 | Claim  |

The model may learn:

```text
If batch = B102 and service_count = 4, then Claim
```

But in real testing data:

| Vehicle | Mileage | Batch | Engine Temp | Service Count | Actual   |
| ------- | ------: | ----- | ----------: | ------------: | -------- |
| V4      |  42,200 | B102  |          94 |             4 | No Claim |

The model may still predict:

```text
Claim
```

This is wrong.

Because it memorized batch-specific data instead of learning general risk.

---

## 9. Solution Steps

### Step 1: Limit Tree Depth

Use:

```python
DecisionTreeClassifier(max_depth=4)
```

Instead of:

```python
DecisionTreeClassifier(max_depth=None)
```

---

### Step 2: Use Minimum Samples Per Leaf

```python
DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=20
)
```

Meaning:

> A final decision should not be made using only 1 or 2 records.

This prevents the model from creating tiny memorized rules.

---

### Step 3: Use Cross-Validation

Instead of checking only one train-test split, test the model on multiple splits.

```text
Split 1 → Train/Test
Split 2 → Train/Test
Split 3 → Train/Test
Split 4 → Train/Test
Split 5 → Train/Test
```

This gives more reliable performance.

---

### Step 4: Remove Leakage Features

Sometimes a column gives unfair future information.

Example:

| Leakage Feature         | Why Problematic                      |
| ----------------------- | ------------------------------------ |
| `claim_ticket_created`  | This is created after warranty claim |
| `repair_bill_generated` | This happens after failure           |
| `claim_status`          | Directly related to target           |

These columns should not be used during training.

---

### Step 5: Try Random Forest

Random Forest uses many trees and averages the result.

It reduces overfitting compared to one deep tree.

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
```

---

## 10. Improved Model Result

| Model                      | Training Accuracy | Testing Accuracy | Result       |
| -------------------------- | ----------------: | ---------------: | ------------ |
| Simple Tree `max_depth=1`  |               62% |              60% | Underfitting |
| Deep Tree `max_depth=None` |               99% |              72% | Overfitting  |
| Pruned Tree `max_depth=4`  |               87% |              84% | Good Fit     |
| Random Forest              |               90% |              86% | Good Fit     |

---

## 11. Visual: Underfitting vs Good Fit vs Overfitting

```text
Model Complexity
Low ---------------- Medium ---------------- High

max_depth=1          max_depth=4             max_depth=None
Underfitting         Good Fit                Overfitting

Train: 62%           Train: 87%              Train: 99%
Test : 60%           Test : 84%              Test : 72%
```

---

## 12. Final Solution

Best practical solution:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=20,
    random_state=42
)
```

Why?

| Reason                | Explanation                     |
| --------------------- | ------------------------------- |
| Controls overfitting  | Tree depth is limited           |
| Avoids memorization   | Uses minimum samples per leaf   |
| Better generalization | Works better on unseen vehicles |
| More stable           | Multiple trees reduce variance  |

---

# Final Comparison

| Topic                      | Underfitting / High Bias | Overfitting / High Variance | Good Fit              |
| -------------------------- | ------------------------ | --------------------------- | --------------------- |
| Model behavior             | Too simple               | Too complex                 | Balanced              |
| Training accuracy          | Low                      | Very high                   | Good                  |
| Testing accuracy           | Low                      | Low/medium                  | Good                  |
| Gap between train and test | Small gap, both low      | Big gap                     | Small gap, both good  |
| Example                    | `max_depth=1`            | `max_depth=None`            | `max_depth=3 or 4`    |
| Problem                    | Cannot learn pattern     | Memorizes training data     | Learns useful pattern |
| Solution                   | Increase complexity      | Reduce complexity           | Tune properly         |

---

## Simple Teaching Summary

```text
High Bias = Underfitting
The model is too simple.
It performs badly on training data and testing data.

High Variance = Overfitting
The model is too complex.
It performs very well on training data but poorly on testing data.

Good Fit
The model learns useful patterns.
Training and testing accuracy are both good and close.
```

## Best Visual Summary

```text
                 Model Performance

                Training Accuracy     Testing Accuracy

Underfitting        Low                    Low
Good Fit            Good                   Good
Overfitting         Very High              Low/Medium


                 Main Cause

Underfitting  → Model too simple
Overfitting   → Model too complex
Good Fit      → Balanced complexity
```
