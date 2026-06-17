### Small Example: Vehicle Service Prediction

A car company uses historical vehicle data to predict whether a vehicle may require service in the next 30 days.

|   Mileage | Engine alerts | Brake wear | Service gap | Prediction          |
| --------: | ------------: | ---------: | ----------: | ------------------- |
| 25,000 km |             0 |        25% |    4 months | Low Service Risk    |
| 52,000 km |             3 |        65% |   10 months | Medium Service Risk |
| 78,000 km |             6 |        88% |   16 months | High Service Risk   |

**How predictive analytics works here:**

1. The model learns from previously serviced vehicles.
2. It identifies patterns such as high mileage, more engine alerts, high brake wear, and long service gaps.
3. For a new vehicle, it predicts the future service risk.

**Example input:**

```text
Mileage: 75,000 km
Engine alerts: 5
Brake wear: 85%
Service gap: 14 months
```

**Prediction:**

```text
High Service Risk
```

**Business action:** The service centre contacts the customer and schedules preventive maintenance before a breakdown occurs.
