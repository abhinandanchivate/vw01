## ML-Based Solution: Entropy in a Decision Tree

### Business problem

An automobile company wants to predict whether a vehicle has:

* **High Service Risk**
* **Low Service Risk**

The Decision Tree considers the feature:

> **Are engine alerts greater than 3?**

---

## Step 1: Training data

| Vehicle | Engine alerts | Actual class |
| ------: | ------------: | ------------ |
|       1 |             7 | High Risk    |
|       2 |             6 | High Risk    |
|       3 |             5 | High Risk    |
|       4 |             8 | High Risk    |
|       5 |             4 | High Risk    |
|       6 |             2 | High Risk    |
|       7 |             1 | Low Risk     |
|       8 |             0 | Low Risk     |
|       9 |             2 | Low Risk     |
|      10 |             1 | Low Risk     |

Total records:

* High Risk = 6
* Low Risk = 4
* Total = 10

---

## Step 2: Calculate entropy before splitting

H(S)=-\sum_{i=1}^{n}p_i\log_2(p_i)

Probabilities:

[
p(\text{High Risk})=\frac{6}{10}=0.6
]

[
p(\text{Low Risk})=\frac{4}{10}=0.4
]

Substitute into the entropy formula:

[
H(S)=-
\left[
0.6\log_2(0.6)+0.4\log_2(0.4)
\right]
]

[
H(S)=0.971
]

### Interpretation

The parent dataset has entropy **0.971**, meaning it contains a high level of uncertainty because both High Risk and Low Risk vehicles are mixed together.

---

# Step 3: Split using engine alerts

The Decision Tree tests this condition:

```text
engine_alerts > 3
```

The resulting tree is:

```text
                 engine_alerts > 3?
                  /               \
               Yes                 No
          5 High Risk         1 High Risk
          0 Low Risk          4 Low Risk
```

---

## Step 4: Entropy of the left child

For `engine_alerts > 3`:

* High Risk = 5
* Low Risk = 0
* Total = 5

Probabilities:

[
p(\text{High Risk})=\frac{5}{5}=1
]

[
p(\text{Low Risk})=\frac{0}{5}=0
]

Entropy:

[
H(\text{Left})=-
\left[
1\log_2(1)+0\log_2(0)
\right]
]

[
H(\text{Left})=0
]

This node is completely pure. Every vehicle is High Risk.

---

## Step 5: Entropy of the right child

For `engine_alerts <= 3`:

* High Risk = 1
* Low Risk = 4
* Total = 5

Probabilities:

[
p(\text{High Risk})=\frac{1}{5}=0.2
]

[
p(\text{Low Risk})=\frac{4}{5}=0.8
]

Entropy:

[
H(\text{Right})=-
\left[
0.2\log_2(0.2)+0.8\log_2(0.8)
\right]
]

[
H(\text{Right})=0.722
]

---

## Step 6: Calculate weighted entropy

Because both child nodes contain five records:

[
H(\text{After Split})
=====================

\frac{5}{10}H(\text{Left})
+
\frac{5}{10}H(\text{Right})
]

[
H(\text{After Split})
=====================

\frac{5}{10}(0)
+
\frac{5}{10}(0.722)
]

[
H(\text{After Split})=0.361
]

---

## Step 7: Calculate Information Gain

Information Gain tells us how much uncertainty was removed by the split.

[
IG=
H(\text{Parent})
----------------

H(\text{After Split})
]

[
IG=0.971-0.361
]

[
\boxed{IG=0.610}
]

### Interpretation

The condition `engine_alerts > 3` reduced entropy from:

[
0.971 \rightarrow 0.361
]

Therefore, it provides an Information Gain of **0.610**, which is a good reduction in uncertainty.

The Decision Tree will compare this Information Gain with other possible features, such as:

* mileage
* brake wear
* service gap
* warranty claims
* customer complaints

The feature producing the highest Information Gain is normally selected for the split.

---

# Python ML implementation

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# -------------------------------------------------
# Step 1: Create automobile service-risk dataset
# -------------------------------------------------

data = {
    "engine_alerts": [7, 6, 5, 8, 4, 2, 1, 0, 2, 1],
    "mileage_km": [
        85000, 78000, 69000, 92000, 65000,
        58000, 30000, 22000, 35000, 27000
    ],
    "brake_wear_percent": [
        90, 85, 80, 95, 75,
        65, 30, 20, 40, 25
    ],
    "service_risk": [
        "High Risk",
        "High Risk",
        "High Risk",
        "High Risk",
        "High Risk",
        "High Risk",
        "Low Risk",
        "Low Risk",
        "Low Risk",
        "Low Risk"
    ]
}

df = pd.DataFrame(data)

print(df)
```

---

## Prepare input and output

```python
X = df[
    [
        "engine_alerts",
        "mileage_km",
        "brake_wear_percent"
    ]
]

y = df["service_risk"]
```

Here:

* `X` contains input features.
* `y` contains the target that the model must predict.

---

## Split the dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## Train Decision Tree using entropy

```python
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)
```

The important parameter is:

```python
criterion="entropy"
```

It instructs the Decision Tree to evaluate splits using entropy and Information Gain.

---

## Make predictions

```python
predictions = model.predict(X_test)

print("Predictions:")
print(predictions)

print("Actual values:")
print(y_test.values)
```

---

## Evaluate the model

```python
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)
```

---

## Predict a new vehicle

```python
new_vehicle = pd.DataFrame({
    "engine_alerts": [6],
    "mileage_km": [72000],
    "brake_wear_percent": [82]
})

prediction = model.predict(new_vehicle)

print("Predicted Service Risk:", prediction[0])
```

Possible output:

```text
Predicted Service Risk: High Risk
```

The model may classify it as High Risk because it has:

* 6 engine alerts
* 72,000 km mileage
* 82% brake wear

---

## Display the Decision Tree

```python
plt.figure(figsize=(14, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True
)

plt.title("Vehicle Service Risk Decision Tree")
plt.show()
```

---

## Complete ML flow

```text
Vehicle historical data
        ↓
Select input features and target
        ↓
Calculate entropy of target
        ↓
Try different feature splits
        ↓
Calculate child-node entropy
        ↓
Calculate weighted entropy
        ↓
Calculate Information Gain
        ↓
Select split with highest Information Gain
        ↓
Build Decision Tree
        ↓
Predict High Risk or Low Risk
```

The entropy formula is therefore not normally calculated manually for every split. The `DecisionTreeClassifier` calculates it internally when we use:

```python
DecisionTreeClassifier(criterion="entropy")
```
