# Case Study: Vehicle Service Risk Prediction

## Business Problem

An automobile company wants to predict whether a vehicle has:

* **High Service Risk**
* **Low Service Risk**

The company has two possible features for creating the first Decision Tree split:

1. `brake_wear_percent`
2. `engine_alerts`

The Decision Tree will calculate **Information Gain** and select the feature that reduces uncertainty the most.

---

## Sample Dataset

| Vehicle | Brake Wear % | Engine Alerts | Actual Service Risk |
| ------: | -----------: | ------------: | ------------------- |
|       1 |           85 |             5 | High                |
|       2 |           90 |             6 | High                |
|       3 |           78 |             4 | High                |
|       4 |           82 |             5 | High                |
|       5 |           75 |             2 | High                |
|       6 |           60 |             4 | High                |
|       7 |           45 |             1 | Low                 |
|       8 |           50 |             2 | Low                 |
|       9 |           68 |             3 | Low                 |
|      10 |           72 |             1 | Low                 |

### Target distribution

| Risk      | Count |
| --------- | ----: |
| High Risk |     6 |
| Low Risk  |     4 |
| Total     |    10 |

---

# Step 1: Calculate Parent Entropy

Before splitting, the dataset contains:

* 6 High Risk vehicles
* 4 Low Risk vehicles

Probabilities:

[
P(High)=\frac{6}{10}=0.6
]

[
P(Low)=\frac{4}{10}=0.4
]

H(S)=-\sum_{i=1}^{n}p_i\log_2(p_i)

Substitute the values:

[
Entropy(Parent)
===============

-\left(0.6\log_2(0.6)+0.4\log_2(0.4)\right)
]

[
Entropy(Parent)\approx0.971
]

### Interpretation

The entropy is close to `1`, so the original dataset is highly mixed.

The Decision Tree now tries different features to reduce this entropy.

---

# Candidate Split 1: Brake Wear Greater Than 70%

The first proposed condition is:

```text
Is brake_wear_percent > 70?
```

## Split result

### Branch 1: Brake wear > 70%

Vehicles: 1, 2, 3, 4, 5 and 10

| Risk  | Count |
| ----- | ----: |
| High  |     5 |
| Low   |     1 |
| Total |     6 |

### Branch 2: Brake wear ≤ 70%

Vehicles: 6, 7, 8 and 9

| Risk  | Count |
| ----- | ----: |
| High  |     1 |
| Low   |     3 |
| Total |     4 |

---

## Step 2: Entropy of Branch 1

Probabilities:

[
P(High)=\frac{5}{6}
]

[
P(Low)=\frac{1}{6}
]

[
Entropy(Branch_1)
=================

-\left(
\frac{5}{6}\log_2\frac{5}{6}
+
\frac{1}{6}\log_2\frac{1}{6}
\right)
]

[
Entropy(Branch_1)\approx0.650
]

This branch is mostly High Risk, but it still contains one Low Risk vehicle.

---

## Step 3: Entropy of Branch 2

Probabilities:

[
P(High)=\frac{1}{4}
]

[
P(Low)=\frac{3}{4}
]

[
Entropy(Branch_2)
=================

-\left(
\frac{1}{4}\log_2\frac{1}{4}
+
\frac{3}{4}\log_2\frac{3}{4}
\right)
]

[
Entropy(Branch_2)\approx0.811
]

This branch is also mixed.

---

## Step 4: Weighted Child Entropy

We cannot simply average the two child entropies because the branches contain different numbers of vehicles.

* Branch 1 has 6 vehicles
* Branch 2 has 4 vehicles

[
Weighted\ Entropy
=================

\frac{6}{10}(0.650)
+
\frac{4}{10}(0.811)
]

[
Weighted\ Entropy
=================

0.390+0.324
]

[
Weighted\ Entropy\approx0.715
]

---

## Step 5: Information Gain for Brake Wear

Information\ Gain=Entropy(Parent)-Weighted\ Entropy(Children)

[
IG(Brake\ Wear)
===============

0.971-0.715
]

[
\boxed{IG(Brake\ Wear)\approx0.256}
]

Brake wear reduces uncertainty by approximately `0.256`.

---

# Candidate Split 2: Engine Alerts Greater Than 3

The Decision Tree also tests:

```text
Are engine_alerts > 3?
```

## Split result

### Branch 1: Engine alerts > 3

Vehicles: 1, 2, 3, 4 and 6

| Risk  | Count |
| ----- | ----: |
| High  |     5 |
| Low   |     0 |
| Total |     5 |

This branch is completely pure.

[
Entropy(Branch_1)=0
]

---

### Branch 2: Engine alerts ≤ 3

Vehicles: 5, 7, 8, 9 and 10

| Risk  | Count |
| ----- | ----: |
| High  |     1 |
| Low   |     4 |
| Total |     5 |

---

## Step 6: Entropy of Branch 2

Probabilities:

[
P(High)=\frac{1}{5}=0.2
]

[
P(Low)=\frac{4}{5}=0.8
]

[
Entropy(Branch_2)
=================

-\left(
0.2\log_2(0.2)
+
0.8\log_2(0.8)
\right)
]

[
Entropy(Branch_2)\approx0.722
]

---

## Step 7: Weighted Child Entropy

Both branches contain 5 vehicles.

[
Weighted\ Entropy
=================

\frac{5}{10}(0)
+
\frac{5}{10}(0.722)
]

[
Weighted\ Entropy
=================

0+0.361
]

[
Weighted\ Entropy\approx0.361
]

---

## Step 8: Information Gain for Engine Alerts

[
IG(Engine\ Alerts)
==================

0.971-0.361
]

[
\boxed{IG(Engine\ Alerts)\approx0.610}
]

Engine alerts reduce uncertainty by approximately `0.610`.

---

# Compare the Two Features

| Candidate split   | Parent entropy | Weighted child entropy | Information gain |
| ----------------- | -------------: | ---------------------: | ---------------: |
| Brake wear > 70%  |          0.971 |                  0.715 |            0.256 |
| Engine alerts > 3 |          0.971 |                  0.361 |        **0.610** |

## Decision

The Decision Tree selects:

```text
engine_alerts > 3
```

because it has the **highest Information Gain**.

[
0.610 > 0.256
]

---

# Final Decision Tree

```text
                  Engine alerts > 3?
                         |
             ┌───────────┴───────────┐
             |                       |
            Yes                      No
             |                       |
      5 High, 0 Low           1 High, 4 Low
       Entropy = 0            Entropy = 0.722
             |                       |
      Predict High Risk       Mostly Low Risk
```

---

# Why Engine Alerts Is Better

## Brake-wear split

```text
Brake wear > 70%       Brake wear ≤ 70%
5 High, 1 Low          1 High, 3 Low
```

Both branches remain mixed.

## Engine-alert split

```text
Alerts > 3             Alerts ≤ 3
5 High, 0 Low          1 High, 4 Low
```

The first branch is completely pure, and the second branch is mostly Low Risk.

Therefore, engine alerts separate the classes more clearly.

---

# Python Verification

```python
import math


def entropy(high_count, low_count):
    total = high_count + low_count

    if total == 0:
        return 0

    probabilities = [
        high_count / total,
        low_count / total
    ]

    result = 0

    for probability in probabilities:
        if probability > 0:
            result -= probability * math.log2(probability)

    return result


# Parent node: 6 High, 4 Low
parent_entropy = entropy(6, 4)

print("Parent Entropy:", round(parent_entropy, 3))


# --------------------------------------------------
# Split 1: Brake wear > 70%
# --------------------------------------------------

brake_branch_1 = entropy(5, 1)
brake_branch_2 = entropy(1, 3)

brake_weighted_entropy = (
    (6 / 10) * brake_branch_1
    + (4 / 10) * brake_branch_2
)

brake_information_gain = (
    parent_entropy - brake_weighted_entropy
)

print("\nBrake Wear Split")
print("Branch 1 Entropy:", round(brake_branch_1, 3))
print("Branch 2 Entropy:", round(brake_branch_2, 3))
print("Weighted Entropy:", round(brake_weighted_entropy, 3))
print("Information Gain:", round(brake_information_gain, 3))


# --------------------------------------------------
# Split 2: Engine alerts > 3
# --------------------------------------------------

alerts_branch_1 = entropy(5, 0)
alerts_branch_2 = entropy(1, 4)

alerts_weighted_entropy = (
    (5 / 10) * alerts_branch_1
    + (5 / 10) * alerts_branch_2
)

alerts_information_gain = (
    parent_entropy - alerts_weighted_entropy
)

print("\nEngine Alerts Split")
print("Branch 1 Entropy:", round(alerts_branch_1, 3))
print("Branch 2 Entropy:", round(alerts_branch_2, 3))
print("Weighted Entropy:", round(alerts_weighted_entropy, 3))
print("Information Gain:", round(alerts_information_gain, 3))


if alerts_information_gain > brake_information_gain:
    print("\nBest feature: Engine Alerts")
else:
    print("\nBest feature: Brake Wear")
```

## Expected Output

```text
Parent Entropy: 0.971

Brake Wear Split
Branch 1 Entropy: 0.65
Branch 2 Entropy: 0.811
Weighted Entropy: 0.715
Information Gain: 0.256

Engine Alerts Split
Branch 1 Entropy: 0.0
Branch 2 Entropy: 0.722
Weighted Entropy: 0.361
Information Gain: 0.61

Best feature: Engine Alerts
```

## Final takeaway

* **Entropy** measures how mixed the High Risk and Low Risk vehicles are.
* **Information Gain** measures how much that mixing decreases after a split.
* The split with the **highest Information Gain** becomes the preferred Decision Tree condition.
* In this case, `engine_alerts > 3` is the best first split.
