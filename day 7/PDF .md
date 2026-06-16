
---

# 1. Probability Density Function — PDF

## Simple definition

A **Probability Density Function** describes how the values of a **continuous numerical variable** are distributed.

Examples of continuous variables:

* Vehicle mileage
* Temperature
* Fuel efficiency
* Battery voltage
* Repair time
* Customer age

A PDF helps us understand where most values are concentrated and where values are rare.

---

## PDF — 5W1H

| Question              | Explanation                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| **What is PDF?**      | A function that describes the probability density of a continuous variable.                      |
| **Why is it used?**   | To understand the shape, centre, spread, and unusual regions of data.                            |
| **When is it used?**  | During EDA, probability analysis, anomaly detection, and model assumptions.                      |
| **Where is it used?** | Machine learning, statistics, reliability analysis, finance, manufacturing, and quality control. |
| **Who uses it?**      | Data scientists, ML engineers, statisticians, analysts, and reliability engineers.               |
| **How is it used?**   | Probability is calculated as the area under the density curve over a specified range.            |

---

## PDF formula

For a continuous random variable (X), the probability that (X) lies between (a) and (b) is:

P(a \leq X \leq b)=\int_a^b f(x),dx

Where:

| Symbol           | Meaning                         |
| ---------------- | ------------------------------- |
| (X)              | Continuous random variable      |
| (f(x))           | Probability density function    |
| (a)              | Lower limit                     |
| (b)              | Upper limit                     |
| Area under curve | Probability between (a) and (b) |

---

## Important PDF rules

### Rule 1: Density cannot be negative

[
f(x)\geq 0
]

### Rule 2: Total area under the PDF is 1

\int_{-\infty}^{\infty}f(x),dx=1

### Rule 3: Probability is calculated over a range

For continuous data:

[
P(X=50{,}000)=0
]

Instead, we calculate something such as:

[
P(40{,}000\leq X\leq60{,}000)
]

This means:

> What is the probability that a vehicle has travelled between 40,000 km and 60,000 km?

---

## Simple PDF example

Assume the mileage of vehicles arriving at a service centre is distributed as follows:

| Mileage range    | Number of vehicles |
| ---------------- | -----------------: |
| 0–20,000 km      |                 10 |
| 20,001–40,000 km |                 25 |
| 40,001–60,000 km |                 40 |
| 60,001–80,000 km |                 20 |
| Above 80,000 km  |                  5 |

The PDF or density curve would be highest around:

```text
40,000–60,000 km
```

because most vehicles belong to that range.

It would be lower near:

```text
Above 80,000 km
```

because fewer vehicles have extremely high mileage.

### Interpretation

The PDF answers:

> In which mileage region are vehicles most concentrated?

---

# 2. Cumulative Distribution Function — CDF

## Simple definition

A **Cumulative Distribution Function** gives the probability that a variable is less than or equal to a selected value.

For example:

> What percentage of vehicles have mileage less than or equal to 60,000 km?

---

## CDF — 5W1H

| Question              | Explanation                                                                                   |
| --------------------- | --------------------------------------------------------------------------------------------- |
| **What is CDF?**      | A function that gives cumulative probability up to a selected value.                          |
| **Why is it used?**   | To calculate percentiles, thresholds, cumulative probabilities, and risk limits.              |
| **When is it used?**  | When answering “less than,” “up to,” or “at most” probability questions.                      |
| **Where is it used?** | Risk analysis, quality control, threshold selection, anomaly detection, and service planning. |
| **Who uses it?**      | Data scientists, analysts, statisticians, and business decision-makers.                       |
| **How is it used?**   | Probabilities are accumulated from the smallest possible value up to (x).                     |

---

## CDF formula

F(x)=P(X\leq x)

For a continuous variable:

F(x)=\int_{-\infty}^{x}f(t),dt

Where:

| Symbol    | Meaning                               |
| --------- | ------------------------------------- |
| (F(x))    | Cumulative probability up to (x)      |
| (f(t))    | Probability density function          |
| (X\leq x) | Variable is less than or equal to (x) |

---

## Important CDF properties

* The CDF starts near 0.
* The CDF ends at 1.
* It never decreases.
* Its value is always between 0 and 1.
* It gives probability directly.

[
0\leq F(x)\leq1
]

---

## Simple CDF example

Suppose the mileage of five vehicles is:

```text
20,000, 40,000, 60,000, 80,000, 100,000
```

The cumulative distribution is:

| Mileage threshold | Vehicles at or below threshold |  CDF |
| ----------------: | -----------------------------: | ---: |
|            20,000 |                     1 out of 5 | 0.20 |
|            40,000 |                     2 out of 5 | 0.40 |
|            60,000 |                     3 out of 5 | 0.60 |
|            80,000 |                     4 out of 5 | 0.80 |
|           100,000 |                     5 out of 5 | 1.00 |

Therefore:

[
F(60{,}000)=0.60
]

### Interpretation

> 60% of the vehicles have mileage less than or equal to 60,000 km.

Similarly:

[
F(80{,}000)=0.80
]

This means:

> 80% of the vehicles have mileage less than or equal to 80,000 km.

---

# 3. PDF versus CDF

| PDF                                 | CDF                                         |
| ----------------------------------- | ------------------------------------------- |
| Probability Density Function        | Cumulative Distribution Function            |
| Shows where values are concentrated | Shows accumulated probability               |
| Probability is obtained from area   | CDF value directly gives probability        |
| Can increase or decrease            | Never decreases                             |
| Total area equals 1                 | Final value approaches 1                    |
| Useful for distribution shape       | Useful for percentiles and thresholds       |
| Answers “between which values?”     | Answers “less than or equal to what value?” |

---

# 4. PDF and CDF relationship

The CDF is obtained by accumulating the area under the PDF.

```text
PDF
Shows density at different values
             ↓
Add the areas from the beginning
             ↓
CDF
Shows cumulative probability
```

For example:

* PDF tells us that many vehicles are concentrated around 50,000 km.
* CDF tells us that 70% of vehicles are below a certain mileage.

---

# 5. Case Study: Vehicle Mileage and Urgent-Service Planning

## Business problem

A connected-car company has mileage data for 1,000 vehicles.

The company wants to:

1. Understand the mileage distribution.
2. Identify where most vehicles are concentrated.
3. Determine the mileage below which 80% of vehicles fall.
4. Identify unusually high-mileage vehicles.
5. Use the results to plan preventive-service campaigns.

---

## Sample data

| Vehicle | Mileage |
| ------- | ------: |
| V1      |  18,000 |
| V2      |  22,000 |
| V3      |  25,000 |
| V4      |  31,000 |
| V5      |  35,000 |
| V6      |  42,000 |
| V7      |  48,000 |
| V8      |  55,000 |
| V9      |  62,000 |
| V10     |  75,000 |
| V11     |  82,000 |
| V12     |  95,000 |
| V13     | 120,000 |

---

# 6. Step 1: State the business questions

| Business question                                            | Suitable method                   |
| ------------------------------------------------------------ | --------------------------------- |
| Where are most vehicle-mileage values concentrated?          | PDF                               |
| What percentage of vehicles are below 60,000 km?             | CDF                               |
| What is the 80th-percentile mileage?                         | CDF                               |
| Which vehicles have unusually high mileage?                  | PDF, CDF, and percentile analysis |
| Which customers should receive preventive-service reminders? | CDF-based threshold               |

---

# 7. Step 2: Create the dataset in Python

```python
import pandas as pd

data = pd.DataFrame({
    "vehicle_id": [
        "V1", "V2", "V3", "V4", "V5",
        "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13"
    ],
    "mileage_km": [
        18000, 22000, 25000, 31000, 35000,
        42000, 48000, 55000, 62000, 75000,
        82000, 95000, 120000
    ]
})

print(data)
```

---

# 8. Step 3: Calculate descriptive statistics

```python
print(data["mileage_km"].describe())
```

Approximate results:

| Measurement |      Value |
| ----------- | ---------: |
| Count       |         13 |
| Minimum     |  18,000 km |
| Median      |  48,000 km |
| Maximum     | 120,000 km |

The maximum is much higher than the median, which indicates a right-skewed distribution.

---

# 9. Step 4: Plot the PDF

In practical EDA, the PDF is often estimated using a **Kernel Density Estimate**, or KDE.

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(9, 5))

sns.histplot(
    data=data,
    x="mileage_km",
    bins=8,
    kde=True,
    stat="density"
)

plt.title("PDF of Vehicle Mileage")
plt.xlabel("Mileage in kilometres")
plt.ylabel("Probability density")
plt.tight_layout()
plt.show()
```

## PDF interpretation

The plot may show:

* Many vehicles concentrated between approximately 20,000 and 60,000 km.
* Fewer vehicles above 80,000 km.
* A long right-hand tail because of vehicles with 95,000 and 120,000 km.

### Business meaning

Most vehicles are in the normal operating-mileage range, but a smaller group has much higher usage and may require closer inspection.

---

# 10. Step 5: Calculate the empirical CDF

```python
sorted_data = data.sort_values(
    by="mileage_km"
).copy()

sorted_data["cdf"] = (
    range(1, len(sorted_data) + 1)
)

sorted_data["cdf"] = (
    sorted_data["cdf"] / len(sorted_data)
)

print(sorted_data)
```

The CDF is calculated as:

[
CDF=\frac{\text{Number of observations at or below the value}}
{\text{Total number of observations}}
]

---

# 11. Step 6: Plot the CDF

```python
plt.figure(figsize=(9, 5))

plt.step(
    sorted_data["mileage_km"],
    sorted_data["cdf"],
    where="post"
)

plt.title("CDF of Vehicle Mileage")
plt.xlabel("Mileage in kilometres")
plt.ylabel("Cumulative probability")
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

# 12. Step 7: Answer probability questions

## Question 1: What percentage of vehicles have mileage at or below 60,000 km?

```python
threshold = 60000

probability = (
    data["mileage_km"]
    .le(threshold)
    .mean()
)

print(
    f"Probability: {probability:.2f}"
)

print(
    f"Percentage: {probability * 100:.2f}%"
)
```

There are eight vehicles at or below 60,000 km out of 13.

[
P(X\leq60{,}000)=\frac{8}{13}
]

[
P(X\leq60{,}000)\approx0.615
]

### Interpretation

> Approximately 61.5% of vehicles have mileage less than or equal to 60,000 km.

---

## Question 2: What percentage of vehicles are above 80,000 km?

We can use:

[
P(X>80{,}000)=1-F(80{,}000)
]

```python
threshold = 80000

probability_above = (
    data["mileage_km"]
    .gt(threshold)
    .mean()
)

print(
    f"Percentage above 80,000 km: "
    f"{probability_above * 100:.2f}%"
)
```

Vehicles above 80,000 km:

* 82,000
* 95,000
* 120,000

Therefore:

[
P(X>80{,}000)=\frac{3}{13}\approx0.231
]

### Interpretation

> Approximately 23.1% of vehicles have travelled more than 80,000 km.

---

# 13. Step 8: Calculate percentile thresholds

## 80th percentile

```python
percentile_80 = data["mileage_km"].quantile(0.80)

print(
    "80th percentile mileage:",
    percentile_80
)
```

The 80th percentile means:

> Approximately 80% of vehicles have mileage below this value.

## 90th percentile

```python
percentile_90 = data["mileage_km"].quantile(0.90)

print(
    "90th percentile mileage:",
    percentile_90
)
```

Vehicles above the 90th-percentile threshold may be considered unusually high-usage vehicles.

---

# 14. Step 9: Create service-risk categories

```python
p80 = data["mileage_km"].quantile(0.80)
p90 = data["mileage_km"].quantile(0.90)

def assign_mileage_risk(mileage):
    if mileage >= p90:
        return "Critical Mileage"
    elif mileage >= p80:
        return "High Mileage"
    else:
        return "Normal Mileage"

data["mileage_risk"] = (
    data["mileage_km"]
    .apply(assign_mileage_risk)
)

print(data)
```

Example business interpretation:

| CDF/percentile position | Risk category    | Recommended action               |
| ----------------------- | ---------------- | -------------------------------- |
| Below 80th percentile   | Normal mileage   | Regular monitoring               |
| 80th–90th percentile    | High mileage     | Send preventive-service reminder |
| Above 90th percentile   | Critical mileage | Schedule priority inspection     |

---

# 15. Complete case-study code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------
# STEP 1: CREATE DATA
# ------------------------------------------------

data = pd.DataFrame({
    "vehicle_id": [
        "V1", "V2", "V3", "V4", "V5",
        "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13"
    ],
    "mileage_km": [
        18000, 22000, 25000, 31000, 35000,
        42000, 48000, 55000, 62000, 75000,
        82000, 95000, 120000
    ]
})

# ------------------------------------------------
# STEP 2: DESCRIPTIVE STATISTICS
# ------------------------------------------------

print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(
    data["mileage_km"]
    .describe()
    .round(2)
)

# ------------------------------------------------
# STEP 3: PDF USING HISTOGRAM AND KDE
# ------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=data,
    x="mileage_km",
    bins=8,
    kde=True,
    stat="density"
)

plt.title("PDF of Vehicle Mileage")
plt.xlabel("Mileage in kilometres")
plt.ylabel("Probability density")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# STEP 4: CALCULATE EMPIRICAL CDF
# ------------------------------------------------

sorted_data = data.sort_values(
    by="mileage_km"
).copy()

sorted_data["cdf"] = (
    range(1, len(sorted_data) + 1)
)

sorted_data["cdf"] = (
    sorted_data["cdf"]
    / len(sorted_data)
)

print("\n" + "=" * 60)
print("EMPIRICAL CDF")
print("=" * 60)

print(sorted_data)

# ------------------------------------------------
# STEP 5: PLOT CDF
# ------------------------------------------------

plt.figure(figsize=(9, 5))

plt.step(
    sorted_data["mileage_km"],
    sorted_data["cdf"],
    where="post"
)

plt.title("CDF of Vehicle Mileage")
plt.xlabel("Mileage in kilometres")
plt.ylabel("Cumulative probability")
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------
# STEP 6: PROBABILITY AT OR BELOW 60,000 KM
# ------------------------------------------------

threshold_60000 = 60000

probability_below_60000 = (
    data["mileage_km"]
    .le(threshold_60000)
    .mean()
)

print(
    "\nPercentage at or below 60,000 km:",
    f"{probability_below_60000 * 100:.2f}%"
)

# ------------------------------------------------
# STEP 7: PROBABILITY ABOVE 80,000 KM
# ------------------------------------------------

threshold_80000 = 80000

probability_above_80000 = (
    data["mileage_km"]
    .gt(threshold_80000)
    .mean()
)

print(
    "Percentage above 80,000 km:",
    f"{probability_above_80000 * 100:.2f}%"
)

# ------------------------------------------------
# STEP 8: PERCENTILES
# ------------------------------------------------

percentile_80 = (
    data["mileage_km"]
    .quantile(0.80)
)

percentile_90 = (
    data["mileage_km"]
    .quantile(0.90)
)

print(
    "80th percentile mileage:",
    round(percentile_80, 2)
)

print(
    "90th percentile mileage:",
    round(percentile_90, 2)
)

# ------------------------------------------------
# STEP 9: CREATE RISK CATEGORIES
# ------------------------------------------------

def assign_mileage_risk(mileage):
    if mileage >= percentile_90:
        return "Critical Mileage"
    elif mileage >= percentile_80:
        return "High Mileage"
    else:
        return "Normal Mileage"

data["mileage_risk"] = (
    data["mileage_km"]
    .apply(assign_mileage_risk)
)

print("\n" + "=" * 60)
print("FINAL VEHICLE RISK CATEGORIES")
print("=" * 60)

print(
    data.sort_values(
        by="mileage_km"
    )
)
```

---

# 16. Final case-study findings

| Finding                                                           | Interpretation                                |
| ----------------------------------------------------------------- | --------------------------------------------- |
| Most mileage values are concentrated in the lower-to-middle range | The PDF shows the most common operating range |
| The distribution has a right tail                                 | A few vehicles have extremely high mileage    |
| About 61.5% are at or below 60,000 km                             | Obtained from the CDF                         |
| About 23.1% are above 80,000 km                                   | These vehicles may require closer monitoring  |
| Vehicles above the 80th percentile have high usage                | Send preventive-maintenance reminders         |
| Vehicles above the 90th percentile have critical usage            | Arrange priority inspections                  |

---

# 17. How PDF and CDF help machine learning

| ML activity             | PDF usage                                          | CDF usage                                        |
| ----------------------- | -------------------------------------------------- | ------------------------------------------------ |
| Understand distribution | Identifies shape and concentration                 | Shows cumulative position                        |
| Find outliers           | Shows rare tail regions                            | Identifies extreme percentiles                   |
| Set thresholds          | Helps identify low-density regions                 | Provides 80th, 90th, or 95th-percentile limits   |
| Feature transformation  | Detects skewed variables                           | Supports percentile-based transformation         |
| Anomaly detection       | Rare PDF regions may indicate anomalies            | Extreme CDF values identify unusual observations |
| Model assumptions       | Checks whether data resembles a known distribution | Compares empirical and theoretical distributions |

---

# Final understanding

* **PDF answers:** Where are the values concentrated?
* **CDF answers:** What percentage of values are less than or equal to a threshold?
* **PDF uses area to calculate probability.**
* **CDF directly provides cumulative probability.**
* **PDF is useful for studying distribution shape.**
* **CDF is useful for percentiles, thresholds, and risk categories.**
