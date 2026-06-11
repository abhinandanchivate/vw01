## Case Study: Loan Approval Prediction using Decision Tree ML

### Business Problem

A bank wants to decide whether a customer’s loan application should be:

| Output   |
| -------- |
| Approved |
| Rejected |

based on customer details like:

| Feature          | Meaning               |
| ---------------- | --------------------- |
| income           | Monthly income        |
| credit_score     | Customer credit score |
| existing_loan    | Existing loan amount  |
| employment_years | Work experience       |
| age              | Customer age          |

---

# Step 1: Install Required Libraries

Use this in **Google Colab / Jupyter Notebook**.

```python
pip install pandas scikit-learn matplotlib
```

---

# Step 2: Import Libraries

```python
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import export_text, plot_tree

import matplotlib.pyplot as plt
```

---

# Step 3: Create Sample Dataset

```python
data = {
    "income": [25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000,
               65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000],
    
    "credit_score": [520, 540, 560, 580, 600, 620, 640, 660,
                     680, 700, 720, 740, 760, 780, 800, 820],
    
    "existing_loan": [200000, 180000, 160000, 150000, 120000, 100000, 90000, 80000,
                      70000, 60000, 50000, 40000, 30000, 25000, 20000, 10000],
    
    "employment_years": [1, 1, 2, 2, 3, 3, 4, 4,
                         5, 5, 6, 6, 7, 8, 9, 10],
    
    "age": [22, 24, 25, 27, 28, 30, 32, 34,
            35, 36, 38, 40, 42, 45, 48, 50],
    
    "loan_status": ["Rejected", "Rejected", "Rejected", "Rejected",
                    "Rejected", "Rejected", "Approved", "Approved",
                    "Approved", "Approved", "Approved", "Approved",
                    "Approved", "Approved", "Approved", "Approved"]
}

df = pd.DataFrame(data)

print(df)
```

---

# Step 4: Understand the Dataset

```python
print(df.head())
print(df.info())
print(df["loan_status"].value_counts())
```

### Explanation

```python
df.head()
```

shows first 5 records.

```python
df.info()
```

shows column names, data types, and null values.

```python
df["loan_status"].value_counts()
```

shows how many records are approved and rejected.

---

# Step 5: Convert Output Label into Numbers

Machine learning models understand numbers better than text.

```python
df["loan_status_num"] = df["loan_status"].map({
    "Rejected": 0,
    "Approved": 1
})

print(df)
```

### Meaning

| Text Label | Numeric Label |
| ---------- | ------------- |
| Rejected   | 0             |
| Approved   | 1             |

---

# Step 6: Split Data into X and y

```python
X = df[["income", "credit_score", "existing_loan", "employment_years", "age"]]

y = df["loan_status_num"]
```

### Explanation

| Variable | Meaning         |
| -------- | --------------- |
| X        | Input features  |
| y        | Output / target |

Example:

```python
X
```

contains:

```text
income, credit_score, existing_loan, employment_years, age
```

```python
y
```

contains:

```text
Approved or Rejected in numeric form
```

---

# Step 7: Split into Training and Testing Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.25, 
    random_state=42
)
```

### Explanation

| Term            | Meaning                           |
| --------------- | --------------------------------- |
| X_train         | Data used to train the model      |
| X_test          | Data used to test the model       |
| y_train         | Correct answers for training data |
| y_test          | Correct answers for testing data  |
| test_size=0.25  | 25% data used for testing         |
| random_state=42 | Keeps result stable every time    |

---

# Step 8: Create Decision Tree Model

```python
tree_model = DecisionTreeClassifier(random_state=42)
```

### Explanation

Decision Tree learns rules like:

```text
IF credit_score > 630 AND income > 50000 THEN Approved
ELSE Rejected
```

The important point is:
we do not manually write these rules.
The algorithm learns these rules from data.

---

# Step 9: Train the Model

```python
tree_model.fit(X_train, y_train)
```

### Meaning

The model studies old customer records and learns patterns.

For example:

```text
Low income + low credit score + high existing loan = Rejected

High income + high credit score + low existing loan = Approved
```

---

# Step 10: Test the Model

```python
y_pred = tree_model.predict(X_test)

print(y_pred)
```

This gives predictions for test customers.

---

# Step 11: Check Accuracy

```python
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

### Explanation

Accuracy tells how many predictions were correct.

Example:

```text
Accuracy: 1.0
```

means:

```text
100% correct on test data
```

Because our dataset is small and simple, accuracy may look very high.

---

# Step 12: Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)

print(cm)
```

### Meaning

Confusion matrix shows:

| Actual / Predicted |         Rejected |         Approved |
| ------------------ | ---------------: | ---------------: |
| Rejected           | Correct rejected |   Wrong approved |
| Approved           |   Wrong rejected | Correct approved |

---

# Step 13: Classification Report

```python
print(classification_report(y_test, y_pred))
```

It shows:

| Metric    | Meaning                                                        |
| --------- | -------------------------------------------------------------- |
| precision | Out of predicted approved/rejected, how many were correct      |
| recall    | Out of actual approved/rejected, how many were found correctly |
| f1-score  | Balance between precision and recall                           |

---

# Step 14: Predict One New Customer

```python
new_customer = [[65000, 720, 50000, 5, 35]]

prediction = tree_model.predict(new_customer)

if prediction[0] == 1:
    print("Approved")
else:
    print("Rejected")
```

### What is this?

```python
new_customer = [[65000, 720, 50000, 5, 35]]
```

Means:

| Feature          | Value |
| ---------------- | ----: |
| income           | 65000 |
| credit_score     |   720 |
| existing_loan    | 50000 |
| employment_years |     5 |
| age              |    35 |

The model checks this customer and predicts whether loan should be approved or rejected.

---

# Step 15: Predict Multiple Customers

```python
new_customers = [
    [30000, 550, 180000, 1, 24],
    [70000, 710, 60000, 5, 36],
    [45000, 600, 120000, 3, 28],
    [90000, 780, 25000, 8, 45]
]

predictions = tree_model.predict(new_customers)

for customer, pred in zip(new_customers, predictions):
    result = "Approved" if pred == 1 else "Rejected"
    print(customer, "=>", result)
```

### Explanation of Loop

```python
for customer, pred in zip(new_customers, predictions):
```

This combines customer data and prediction together.

Example:

```text
[30000, 550, 180000, 1, 24] => Rejected
[70000, 710, 60000, 5, 36] => Approved
```

---

# Step 16: See Rules Learned by Decision Tree

```python
rules = export_text(
    tree_model, 
    feature_names=["income", "credit_score", "existing_loan", "employment_years", "age"]
)

print(rules)
```

### Example Output

You may see rules like:

```text
|--- credit_score <= 630.00
|   |--- class: 0
|--- credit_score > 630.00
|   |--- class: 1
```

### Meaning

```text
IF credit_score <= 630 THEN Rejected
ELSE Approved
```

This is why Decision Tree is easy to explain.

---

# Step 17: Visualize Decision Tree

```python
plt.figure(figsize=(15, 8))

plot_tree(
    tree_model,
    feature_names=["income", "credit_score", "existing_loan", "employment_years", "age"],
    class_names=["Rejected", "Approved"],
    filled=True
)

plt.show()
```

### Explanation

The diagram shows how the model takes decisions.

Example:

```text
credit_score <= 630?
    Yes -> Rejected
    No  -> Approved
```

---

# Full Code Together

```python
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import export_text, plot_tree

import matplotlib.pyplot as plt


data = {
    "income": [25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000,
               65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000],
    
    "credit_score": [520, 540, 560, 580, 600, 620, 640, 660,
                     680, 700, 720, 740, 760, 780, 800, 820],
    
    "existing_loan": [200000, 180000, 160000, 150000, 120000, 100000, 90000, 80000,
                      70000, 60000, 50000, 40000, 30000, 25000, 20000, 10000],
    
    "employment_years": [1, 1, 2, 2, 3, 3, 4, 4,
                         5, 5, 6, 6, 7, 8, 9, 10],
    
    "age": [22, 24, 25, 27, 28, 30, 32, 34,
            35, 36, 38, 40, 42, 45, 48, 50],
    
    "loan_status": ["Rejected", "Rejected", "Rejected", "Rejected",
                    "Rejected", "Rejected", "Approved", "Approved",
                    "Approved", "Approved", "Approved", "Approved",
                    "Approved", "Approved", "Approved", "Approved"]
}

df = pd.DataFrame(data)

df["loan_status_num"] = df["loan_status"].map({
    "Rejected": 0,
    "Approved": 1
})

X = df[["income", "credit_score", "existing_loan", "employment_years", "age"]]
y = df["loan_status_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.25, 
    random_state=42
)

tree_model = DecisionTreeClassifier(random_state=42)

tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

new_customers = [
    [30000, 550, 180000, 1, 24],
    [70000, 710, 60000, 5, 36],
    [45000, 600, 120000, 3, 28],
    [90000, 780, 25000, 8, 45]
]

predictions = tree_model.predict(new_customers)

for customer, pred in zip(new_customers, predictions):
    result = "Approved" if pred == 1 else "Rejected"
    print(customer, "=>", result)

rules = export_text(
    tree_model, 
    feature_names=["income", "credit_score", "existing_loan", "employment_years", "age"]
)

print("Decision Tree Rules:")
print(rules)

plt.figure(figsize=(15, 8))

plot_tree(
    tree_model,
    feature_names=["income", "credit_score", "existing_loan", "employment_years", "age"],
    class_names=["Rejected", "Approved"],
    filled=True
)

plt.show()
```

---

# Final Business Explanation

Decision Tree works like a decision-making flowchart.

For loan approval, it may learn rules such as:

```text
IF credit_score is low THEN reject loan

IF credit_score is high AND income is good THEN approve loan

IF existing loan is very high THEN reject loan
```

So, the model is useful when we want:

| Requirement              | Decision Tree Advantage            |
| ------------------------ | ---------------------------------- |
| Easy explanation         | Rules are visible                  |
| Business decision making | Works like IF-ELSE logic           |
| Classification problem   | Can predict Approved / Rejected    |
| Non-technical audience   | Easy to explain using tree diagram |
