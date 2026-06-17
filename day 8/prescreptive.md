# Automobile Analytics Case Study

## Business Problem

A car manufacturer wants to reduce unexpected vehicle breakdowns, improve customer safety, and minimize emergency service costs.

The organization collects vehicle condition and customer-service data to identify high-risk vehicles and take preventive action.

## Current Vehicle Data

| Feature                   | Example Value |
| ------------------------- | ------------: |
| Mileage                   |     72,000 km |
| Months since last service |     14 months |
| Engine alerts             |             5 |
| Brake wear                |           82% |
| Battery voltage           |           Low |
| Customer complaints       |             3 |

---

## Step 1: Descriptive Analytics

Descriptive analytics examines historical vehicle-service records to understand previous breakdown patterns and service-related problems.

### Findings

* 30% of vehicles with mileage above 70,000 km experienced component failures.
* Vehicles with brake wear above 75% received more emergency service requests.
* Vehicles not serviced for more than 12 months had twice the normal failure rate.
* Battery-related complaints increased by 15% during winter.

### Question Answered

> **What happened in the past?**

### Business Interpretation

The historical data indicates that high mileage, excessive brake wear, long service gaps, and low battery performance are strongly associated with unexpected vehicle failures.

---

## Step 2: Predictive Analytics

Predictive analytics uses historical data and current vehicle information to estimate the likelihood of a future breakdown.

A machine-learning model processes the following risk indicators:

* Mileage: 72,000 km
* Service gap: 14 months
* Engine alerts: 5
* Brake wear: 82%
* Battery voltage: Low
* Customer complaints: 3

### Model Output

| Prediction                                         |    Result |
| -------------------------------------------------- | --------: |
| Probability of service failure in the next 30 days |       89% |
| Risk category                                      | High Risk |

### Question Answered

> **What is likely to happen?**

### Business Interpretation

The vehicle has a high probability of experiencing a service failure within the next 30 days. Immediate preventive action is required.

---

## Step 3: Prescriptive Analytics

Prescriptive analytics uses the prediction, business rules, service capacity, spare-part availability, costs, and safety requirements to recommend suitable actions.

### Recommended Actions

1. Contact the customer immediately.
2. Schedule a service appointment within 48 hours.
3. Inspect the braking system and battery.
4. Diagnose the engine alerts.
5. Reserve the required spare parts before the vehicle arrives.
6. Assign the case a high-priority service status.
7. Provide a temporary replacement vehicle when required.
8. Send automatic reminders until the appointment is confirmed.

### Question Answered

> **What should the company do?**

### Expected Business Benefits

* Reduced unexpected vehicle breakdowns
* Improved customer safety
* Lower emergency repair costs
* Better spare-parts planning
* Reduced vehicle downtime
* Improved customer satisfaction
* More efficient service-center operations

---

## Analytics Workflow

```text
Historical Vehicle and Service Data
                 |
                 v
        Descriptive Analytics
   Identify past patterns and failures
                 |
                 v
        Predictive Analytics
 Estimate future breakdown probability
                 |
                 v
       Prescriptive Analytics
 Recommend preventive service actions
                 |
                 v
       Service Team Execution
 Contact customer, inspect vehicle,
 reserve parts, and complete service
                 |
                 v
       Outcome and Feedback Data
                 |
                 +--------------------+
                                      |
                                      v
                           Model Improvement
```

---

## Comparison of Analytics Types

| Parameter          | Descriptive Analytics              | Predictive Analytics                          | Prescriptive Analytics                                |
| ------------------ | ---------------------------------- | --------------------------------------------- | ----------------------------------------------------- |
| Main question      | What happened?                     | What is likely to happen?                     | What should be done?                                  |
| Time focus         | Past                               | Future                                        | Future action                                         |
| Purpose            | Understand historical performance  | Forecast outcomes                             | Recommend decisions and actions                       |
| Input              | Historical data                    | Historical and current data                   | Predictions, rules, costs, resources, and constraints |
| Output             | Reports, summaries, and dashboards | Forecasts, probabilities, and risk categories | Recommended actions and optimized decisions           |
| Complexity         | Low to medium                      | Medium to high                                | High                                                  |
| Automobile example | 200 vehicles failed last month     | 40 vehicles may fail next month               | Service these 40 vehicles first                       |

---

## End-to-End Case Summary

| Analytics Stage | Automobile Case Result                                                                                               |
| --------------- | -------------------------------------------------------------------------------------------------------------------- |
| Descriptive     | Vehicles with high mileage, excessive brake wear, long service gaps, and battery problems had more breakdowns.       |
| Predictive      | The selected vehicle has an 89% probability of failure in the next 30 days.                                          |
| Prescriptive    | Contact the customer and schedule preventive service within 48 hours, with brake and battery inspection prioritized. |

## Final Business Decision

Because the vehicle is classified as **High Risk**, the manufacturer should prioritize it for preventive maintenance instead of waiting for an unexpected breakdown.

This approach converts historical data into a prediction and then converts the prediction into a practical business action.
::: 
