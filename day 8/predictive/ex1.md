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


| Mileage (km) | Engine alerts | Brake wear (%) | Service gap (months) | Service risk |
| -----------: | ------------: | -------------: | -------------------: | ------------ |
|       20,000 |             0 |             20 |                    3 | Low Risk     |
|       28,000 |             1 |             30 |                    5 | Low Risk     |
|       35,000 |             1 |             40 |                    6 | Low Risk     |
|       45,000 |             2 |             50 |                    8 | Low Risk     |
|       52,000 |             3 |             65 |                   10 | High Risk    |
|       60,000 |             4 |             72 |                   12 | High Risk    |
|       70,000 |             5 |             82 |                   14 | High Risk    |
|       80,000 |             7 |             90 |                   18 | High Risk    |

