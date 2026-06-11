
# Practical 2: Advanced Supervised ML Case Study

## Case Study: Enterprise Email Spam and Phishing Risk Detection

### Business Problem

A large enterprise receives thousands of emails every day. Some are normal business emails, some are promotional spam, and some are phishing emails.

The company wants an ML model that can predict whether an incoming email is:

| Label | Meaning              |
| ----- | -------------------- |
| 0     | Safe / Not Spam      |
| 1     | Spam / Phishing Risk |

This is not only a basic classification problem. Senior candidates must think about:

| Area                | Expected Senior-Level Thinking                                          |
| ------------------- | ----------------------------------------------------------------------- |
| Business risk       | Missing phishing emails is more dangerous than blocking one valid email |
| Data quality        | Some fields may be missing or noisy                                     |
| Feature engineering | Raw email metadata must be converted into ML features                   |
| Model evaluation    | Accuracy alone is not enough                                            |
| Imbalance           | Spam emails may be fewer than normal emails                             |
| Threshold tuning    | Decide when to block, quarantine, or allow                              |
| Deployment          | Model should be usable in production                                    |

---

# Objective

Train a supervised ML model to classify emails as **Safe** or **Spam/Phishing Risk** using multiple realistic features.

---

# Dataset Features

| Feature             | Meaning                                                  |
| ------------------- | -------------------------------------------------------- |
| `offer_words`       | Count of words like free, win, discount, offer           |
| `links`             | Number of links in the email                             |
| `attachments`       | Number of attachments                                    |
| `sender_reputation` | Score from 0 to 100, higher is safer                     |
| `domain_age_days`   | Age of sender domain                                     |
| `email_length`      | Total email length                                       |
| `uppercase_ratio`   | Percentage of uppercase words                            |
| `has_urgent_words`  | Whether email has words like urgent, immediately, verify |
| `has_bank_words`    | Whether email talks about bank, payment, account         |
| `spf_pass`          | SPF email authentication passed or failed                |
| `dkim_pass`         | DKIM authentication passed or failed                     |
| `dmarc_pass`        | DMARC authentication passed or failed                    |
| `sender_type`       | Internal, known external, unknown external               |
| `label`             | Final output: 0 safe, 1 spam/phishing                    |

---

# Step 1: Install Libraries

```python
pip install pandas scikit-learn matplotlib
```

---

# Step 2: Import Required Libraries

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve
)

import matplotlib.pyplot as plt
```

---

# Step 3: Create Realistic Sample Dataset

For training purpose, we are generating synthetic enterprise email data.

In real projects, this data may come from:

| Source           | Example                                  |
| ---------------- | ---------------------------------------- |
| Email gateway    | Microsoft Defender, Proofpoint, Mimecast |
| Security system  | SIEM logs                                |
| User reports     | Marked as spam / phishing                |
| Email headers    | SPF, DKIM, DMARC                         |
| CRM or HR system | Known sender validation                  |

```python
np.random.seed(42)

total_records = 1000

data = pd.DataFrame({
    "offer_words": np.random.poisson(3, total_records),
    "links": np.random.poisson(2, total_records),
    "attachments": np.random.poisson(1, total_records),
    "sender_reputation": np.random.randint(10, 100, total_records),
    "domain_age_days": np.random.randint(1, 3000, total_records),
    "email_length": np.random.randint(50, 5000, total_records),
    "uppercase_ratio": np.round(np.random.uniform(0.01, 0.70, total_records), 2),
    "has_urgent_words": np.random.choice([0, 1], total_records, p=[0.65, 0.35]),
    "has_bank_words": np.random.choice([0, 1], total_records, p=[0.70, 0.30]),
    "spf_pass": np.random.choice(["yes", "no"], total_records, p=[0.80, 0.20]),
    "dkim_pass": np.random.choice(["yes", "no"], total_records, p=[0.75, 0.25]),
    "dmarc_pass": np.random.choice(["yes", "no"], total_records, p=[0.78, 0.22]),
    "sender_type": np.random.choice(
        ["internal", "known_external", "unknown_external"],
        total_records,
        p=[0.35, 0.45, 0.20]
    )
})
```

---

# Step 4: Create Target Label

Here we simulate business logic.

An email becomes risky if:

| Condition              | Risk                    |
| ---------------------- | ----------------------- |
| Many offer words       | Spam possibility        |
| Many links             | Phishing possibility    |
| Poor sender reputation | Suspicious sender       |
| New domain             | Risky domain            |
| SPF/DKIM/DMARC fail    | Authentication issue    |
| Urgent/banking words   | Social engineering risk |

```python
risk_score = (
    data["offer_words"] * 0.8 +
    data["links"] * 1.2 +
    data["attachments"] * 0.5 +
    data["has_urgent_words"] * 2 +
    data["has_bank_words"] * 1.5 +
    (100 - data["sender_reputation"]) * 0.05 +
    (data["domain_age_days"] < 180).astype(int) * 2 +
    (data["spf_pass"] == "no").astype(int) * 2 +
    (data["dkim_pass"] == "no").astype(int) * 1.5 +
    (data["dmarc_pass"] == "no").astype(int) * 2 +
    (data["sender_type"] == "unknown_external").astype(int) * 2
)

data["label"] = (risk_score > 8).astype(int)

print(data.head())
print(data["label"].value_counts())
```

---

# Step 5: Understand the Target Distribution

```python
print(data["label"].value_counts(normalize=True) * 100)
```

### Why this matters?

If the dataset has 90% safe emails and 10% spam emails, then a model can simply predict everything as safe and still get 90% accuracy.

So, for experienced candidates, we should evaluate:

| Metric           | Meaning                                            |
| ---------------- | -------------------------------------------------- |
| Accuracy         | Overall correctness                                |
| Precision        | Out of predicted spam, how many were actually spam |
| Recall           | Out of actual spam, how many were detected         |
| F1-score         | Balance between precision and recall               |
| ROC-AUC          | Model ranking ability                              |
| Confusion Matrix | Actual vs predicted result                         |

---

# Step 6: Split Input and Output

```python
X = data.drop("label", axis=1)
y = data["label"]
```

```python
numeric_features = [
    "offer_words",
    "links",
    "attachments",
    "sender_reputation",
    "domain_age_days",
    "email_length",
    "uppercase_ratio",
    "has_urgent_words",
    "has_bank_words"
]

categorical_features = [
    "spf_pass",
    "dkim_pass",
    "dmarc_pass",
    "sender_type"
]
```

---

# Step 7: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

### Why `stratify=y`?

It keeps the same spam/safe ratio in both training and testing data.

This is important when the dataset is imbalanced.

---

# Step 8: Create ML Pipeline

Senior-level candidates should not manually transform each column separately.

Use a pipeline.

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ))
    ]
)
```

### Why Pipeline?

| Benefit             | Explanation                                           |
| ------------------- | ----------------------------------------------------- |
| Cleaner code        | Preprocessing and model training are combined         |
| Production ready    | Same steps used during training and prediction        |
| Avoids data leakage | Transformations happen correctly inside training flow |
| Reusable            | Can be saved and deployed                             |

---

# Step 9: Train the Model

```python
model.fit(X_train, y_train)
```

---

# Step 10: Evaluate the Model

```python
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

---

# Step 11: Business Interpretation of Confusion Matrix

Confusion matrix output will look like this:

```text
[[TN FP]
 [FN TP]]
```

| Term | Meaning                                |
| ---- | -------------------------------------- |
| TN   | Safe email correctly predicted as safe |
| FP   | Safe email wrongly blocked as spam     |
| FN   | Spam email wrongly allowed as safe     |
| TP   | Spam email correctly detected          |

For this case study:

| Error                | Business Impact                 |
| -------------------- | ------------------------------- |
| False Positive       | Genuine email blocked           |
| False Negative       | Phishing email reaches employee |
| More dangerous error | False Negative                  |

So, we should focus more on **Recall**.

---

# Step 12: Threshold Tuning

By default, ML models use threshold `0.5`.

Meaning:

```text
If spam probability >= 0.5, predict spam
```

But in security use cases, we may want to catch more spam emails.

So we can reduce threshold to improve recall.

```python
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

f1_scores = 2 * (precision * recall) / (precision + recall + 0.000001)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print("Best Threshold:", best_threshold)
print("Precision at Best Threshold:", precision[best_index])
print("Recall at Best Threshold:", recall[best_index])
print("F1 at Best Threshold:", f1_scores[best_index])
```

Now apply custom threshold:

```python
custom_threshold = best_threshold

y_pred_custom = (y_prob >= custom_threshold).astype(int)

print(classification_report(y_test, y_pred_custom))
print(confusion_matrix(y_test, y_pred_custom))
```

---

# Step 13: Predict New Enterprise Email

```python
new_email = pd.DataFrame([{
    "offer_words": 7,
    "links": 6,
    "attachments": 2,
    "sender_reputation": 25,
    "domain_age_days": 45,
    "email_length": 1200,
    "uppercase_ratio": 0.45,
    "has_urgent_words": 1,
    "has_bank_words": 1,
    "spf_pass": "no",
    "dkim_pass": "no",
    "dmarc_pass": "no",
    "sender_type": "unknown_external"
}])

spam_probability = model.predict_proba(new_email)[0][1]

print("Spam Probability:", spam_probability)

if spam_probability >= custom_threshold:
    print("Spam / Phishing Risk")
else:
    print("Safe Email")
```

---

# Step 14: Add Business Action Layer

In real projects, we may not simply say Spam or Not Spam.

We can classify into action levels.

```python
def email_action(probability):
    if probability >= 0.80:
        return "Block Email"
    elif probability >= 0.60:
        return "Quarantine for Security Review"
    elif probability >= 0.40:
        return "Deliver with Warning Banner"
    else:
        return "Allow Email"

action = email_action(spam_probability)

print("Spam Probability:", spam_probability)
print("Recommended Action:", action)
```

---

# Step 15: Feature Importance

Experienced candidates should explain **why** the model predicted something.

```python
classifier = model.named_steps["classifier"]

encoded_cat_features = model.named_steps["preprocess"] \
    .named_transformers_["cat"] \
    .get_feature_names_out(categorical_features)

all_features = numeric_features + list(encoded_cat_features)

feature_importance = pd.DataFrame({
    "feature": all_features,
    "importance": classifier.feature_importances_
}).sort_values(by="importance", ascending=False)

print(feature_importance.head(10))
```

---

# Step 16: Visualize Important Features

```python
top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features for Spam Detection")
plt.gca().invert_yaxis()
plt.show()
```

---

# Final Case Study Summary

| Area            | Basic Candidate       | Experienced Candidate                            |
| --------------- | --------------------- | ------------------------------------------------ |
| Dataset         | Uses 2 columns        | Uses multiple realistic business features        |
| Model           | Direct model training | Pipeline-based training                          |
| Encoding        | Manual mapping        | ColumnTransformer + OneHotEncoder                |
| Evaluation      | Accuracy only         | Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| Risk Thinking   | Predict spam/not spam | Thinks about false positives and false negatives |
| Threshold       | Uses default 0.5      | Tunes threshold based on business impact         |
| Output          | Spam / Not Spam       | Block / Quarantine / Warning / Allow             |
| Production View | Not considered        | Pipeline, monitoring, retraining, explainability |

---

# Candidate Assignment

Ask candidates to complete the following:

| Task                    | Expected Output                                        |
| ----------------------- | ------------------------------------------------------ |
| Build dataset           | Use at least 1000 records                              |
| Perform EDA             | Class distribution, feature analysis                   |
| Build ML pipeline       | Preprocessing + model                                  |
| Train model             | Random Forest / Logistic Regression / XGBoost optional |
| Evaluate model          | Accuracy, Precision, Recall, F1, ROC-AUC               |
| Tune threshold          | Choose threshold based on business risk                |
| Explain prediction      | Show top important features                            |
| Create business rule    | Block, quarantine, warning, allow                      |
| Save model              | Use `joblib`                                           |
| Prepare deployment plan | API-based prediction service                           |

---

# Additional Senior-Level Discussion Questions

| Question                                  | Expected Senior-Level Answer                                     |
| ----------------------------------------- | ---------------------------------------------------------------- |
| Why is accuracy not enough?               | Because dataset can be imbalanced                                |
| Which is more dangerous: FP or FN?        | FN, because phishing email reaches user                          |
| Why use pipeline?                         | Same transformation during training and prediction               |
| Why tune threshold?                       | Business risk may need higher recall                             |
| How will you monitor model in production? | Data drift, prediction drift, false positive rate, feedback loop |
| How will labels be collected?             | User reports, SOC analyst review, email gateway logs             |
| How will model be retrained?              | Scheduled retraining with newly labeled emails                   |
| How to handle new sender types?           | `handle_unknown="ignore"` in encoder                             |
| What is the deployment pattern?           | REST API or batch scoring pipeline                               |
| What is the security concern?             | Model should not expose email content or sensitive data          |

---

# Production Extension


```text
Email Gateway → Feature Extraction Service → ML Model API → Risk Score → Action Engine → SOC Dashboard
```

| Component         | Responsibility                                     |
| ----------------- | -------------------------------------------------- |
| Email Gateway     | Receives incoming emails                           |
| Feature Extractor | Extracts links, domain, SPF, DKIM, DMARC, keywords |
| ML Model API      | Predicts spam probability                          |
| Action Engine     | Decides allow/block/quarantine                     |
| SOC Dashboard     | Security team reviews risky emails                 |
| Feedback Loop     | Analyst decisions are stored for retraining        |
