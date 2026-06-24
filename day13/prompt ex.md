
## Customer input

```text
My Volkswagen Taigun 2022 has run 78,500 km.
It takes longer to start, braking noise is coming,
fuel efficiency has reduced, and engine warning light appeared 4 times.
Battery voltage is 11.4 V and brake wear is 81%.
What should I do?
```

## Internal LLM steps

```text
Customer text
      ↓
Step 1: Extract facts
      ↓
Step 2: Identify symptoms
      ↓
Step 3: Detect missing information
      ↓
Step 4: Send structured data to ML/rule engine
      ↓
Step 5: Receive risk score
      ↓
Step 6: Generate safe customer response
```

## Step 1: LLM extracts facts

```json
{
  "vehicle_model": "Volkswagen Taigun",
  "model_year": 2022,
  "mileage_km": 78500,
  "slow_start": true,
  "braking_noise": true,
  "fuel_efficiency_reduced": true,
  "engine_warning_alerts": 4,
  "battery_voltage": 11.4,
  "brake_wear_percent": 81
}
```

## Step 2: LLM identifies possible concern areas

```text
Slow start + 11.4 V
→ Battery / charging / starter inspection needed

Braking noise + 81% brake wear
→ Brake pad/disc inspection needed

Fuel efficiency reduced + engine alerts
→ Engine diagnostics scan needed

High mileage + long symptoms
→ Service urgency increases
```

## Step 3: LLM detects missing data

```json
{
  "missing_information": [
    "Was battery voltage measured with engine ON or OFF?",
    "Any dashboard warning currently active?",
    "Last service date?",
    "Petrol or diesel variant?",
    "Any vibration while braking?"
  ]
}
```

## Step 4: ML/rule engine receives features

```json
{
  "mileage_km": 78500,
  "battery_voltage": 11.4,
  "engine_alert_count": 4,
  "brake_wear_percent": 81,
  "complaints_count": 3,
  "slow_start": 1,
  "braking_noise": 1,
  "fuel_efficiency_reduced": 1
}
```

## Step 5: Risk engine output

Example output:

```json
{
  "risk_category": "High",
  "service_urgency_probability": 0.86,
  "priority": "Inspect within 24–48 hours",
  "drive_advice": "Avoid long drives before inspection",
  "recommended_inspections": [
    "Battery health and load test",
    "Alternator charging test",
    "Starter motor inspection",
    "Brake pad and disc inspection",
    "OBD diagnostic scan",
    "Fuel system and air filter inspection"
  ]
}
```

## Step 6: LLM generates final response

```text
Based on the symptoms and values provided, this vehicle should be treated as High Priority.

The main reasons are:
- Battery voltage is low at 11.4 V and the vehicle is taking longer to start.
- Brake wear is high at 81% and braking noise is present.
- Fuel efficiency has reduced and engine warning alerts appeared 4 times.
- Mileage is 78,500 km, so preventive inspection is important.

You should avoid long drives before inspection. If the vehicle starts normally and no red warning light is active, short local driving may be acceptable, but service should be scheduled within 24–48 hours.

Recommended inspections:
1. Battery health test
2. Battery load test
3. Alternator charging test
4. Starter motor inspection
5. Brake pad and disc inspection
6. OBD engine diagnostic scan
7. Fuel system and air filter inspection

This is not a confirmed diagnosis. A technician should validate the issue using proper diagnostic tools.
```

## Simple architecture

```text
Customer Complaint
       ↓
LLM Extraction
       ↓
Structured JSON
       ↓
ML / Rule Risk Engine
       ↓
Risk Score + Recommendations
       ↓
LLM Explanation
       ↓
Customer-Friendly Advisory
```
