## Live Use Case: Predicting Brake-System Failure in a Fleet

A logistics company operates **1,000 delivery vehicles**. It uses an ML model to predict whether each vehicle is likely to have a brake-related failure within the next 30 days.

The maintenance team uses the prediction as follows:

* **Predicted Risky** → inspect the vehicle immediately.
* **Predicted Normal** → continue regular operation.

### Actual data after 30 days

* 950 vehicles had no brake failure.
* 50 vehicles developed a brake problem.

Suppose the model produces this confusion matrix:

| Actual / Predicted  | Predicted Risky | Predicted Normal |
| ------------------- | --------------: | ---------------: |
| **Actually Risky**  |              40 |               10 |
| **Actually Normal** |              30 |              920 |

### Meaning

#### True Positive: 40

These 40 vehicles were correctly identified as risky.

The company inspected them and replaced worn brake components before failure.

#### False Negative: 10

These 10 vehicles were actually risky, but the model predicted them as normal.

This is the most dangerous error because:

* The vehicles were not inspected.
* They continued operating.
* Brake problems could occur during delivery.
* Drivers and passengers could be at risk.
* Emergency repair and towing costs could increase.
* Delivery operations could be delayed.

#### False Positive: 30

These vehicles were normal, but the model predicted them as risky.

They were inspected unnecessarily.

This creates:

* Extra inspection cost
* Maintenance workload
* Temporary vehicle downtime

However, this is generally less dangerous than missing an actual brake failure.

#### True Negative: 920

These vehicles were correctly identified as normal and continued operating.

---

## Accuracy

[
\text{Accuracy}=\frac{40+920}{1000}=96%
]

The model has **96% accuracy**, which looks good.

But the company should also check how many risky vehicles were found.

## Recall

[
\text{Recall}=\frac{40}{40+10}=80%
]

This means:

> The model identified 40 out of 50 risky vehicles, but missed 10.

## Business Decision

The company may decide that 10 missed brake failures are unacceptable.

It can lower the prediction threshold so that the model identifies more risky vehicles.

For example:

| Model Setting     | Risky Vehicles Found | Risky Vehicles Missed | Unnecessary Inspections |
| ----------------- | -------------------: | --------------------: | ----------------------: |
| Current threshold |                   40 |                    10 |                      30 |
| Lower threshold   |                   47 |                     3 |                      70 |

The lower threshold creates more inspections, but it reduces the number of dangerous missed failures.

## Why the confusion matrix is important

Accuracy only says:

> 960 out of 1,000 predictions were correct.

The confusion matrix reveals:

> 10 vehicles that may have brake failure were wrongly allowed to continue operating.

Therefore, in safety-related automobile systems, the company may prioritize **high recall for the failure class**, even if it causes some extra inspections.
