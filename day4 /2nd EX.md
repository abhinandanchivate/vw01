
## Example: Loan Approval using Decision Tree

### Use Case

Predict whether a customer loan should be:

```text
Approved
Rejected
```

based on:

| Feature        | Meaning               |
| -------------- | --------------------- |
| `income`       | Monthly income        |
| `credit_score` | Customer credit score |

---

## Code

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Step 1: Create sample data
data = {
    "income": [25000, 30000, 35000, 60000, 75000, 90000],
    "credit_score": [500, 550, 580, 700, 750, 800],
    "loan_status": [0, 0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

print(df)
```

Output meaning:

| income | credit_score | loan_status |
| -----: | -----------: | ----------: |
|  25000 |          500 |           0 |
|  30000 |          550 |           0 |
|  35000 |          580 |           0 |
|  60000 |          700 |           1 |
|  75000 |          750 |           1 |
|  90000 |          800 |           1 |

Here:

```text
0 = Rejected
1 = Approved
```

---

## Step 2: Split input and output

```python
X = df[["income", "credit_score"]]
y = df["loan_status"]
```

| Variable | Meaning       |
| -------- | ------------- |
| `X`      | Input columns |
| `y`      | Output column |

---

## Step 3: Train Decision Tree model

```python
tree_model = DecisionTreeClassifier()

tree_model.fit(X, y)
```

The model learns rules like:

```text
IF income is high AND credit_score is high
THEN loan approved
ELSE loan rejected
```

---

## Step 4: Predict new customer

```python
new_customer = [[65000, 720]]

prediction = tree_model.predict(new_customer)

print("Approved" if prediction[0] == 1 else "Rejected")
```

Output:

```text
Approved
```

Because:

```text
income = 65000
credit_score = 720
```

Both values are high, so the model predicts **Approved**.

---

## Another test

```python
new_customer = [[28000, 520]]

prediction = tree_model.predict(new_customer)

print("Approved" if prediction[0] == 1 else "Rejected")
```

Output:

```text
Rejected
```

Because:

```text
income = 28000
credit_score = 520
```

Both values are low.

---

## Decision Tree rule explanation

The Decision Tree may learn something like:

```text
IF credit_score > 640
    THEN Approved
ELSE
    Rejected
```

Or:

```text
IF income > 50000 AND credit_score > 650
    THEN Approved
ELSE
    Rejected
```

---

## Compare with your spam example

| Spam Example    | Loan Example        |
| --------------- | ------------------- |
| `offer_words`   | `income`            |
| `links`         | `credit_score`      |
| Spam / Not Spam | Approved / Rejected |

So this:

```python
tree_model.predict([[6, 7]])
```

means:

```text
Predict spam using:
offer_words = 6
links = 7
```

And this:

```python
tree_model.predict([[65000, 720]])
```

means:

```text
Predict loan approval using:
income = 65000
credit_score = 720
```
