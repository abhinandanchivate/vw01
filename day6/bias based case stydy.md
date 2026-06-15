# Volkswagen Predictive Maintenance — ML Bias Case Study

---

## The Problem

VW deployed a **predictive maintenance ML model** across its global fleet to predict when components (brake pads, engine mounts, suspension) would fail.

**Training Data Source:** Primarily European fleet data
- German Autobahn driving
- Moderate climate (15–20°C avg)
- Well-paved roads
- Highway-dominant usage

**Deployment Target (India):** Pune/Pimpri region
- Stop-go traffic in industrial corridors
- Extreme heat (28–42°C)
- High road vibration (potholes)
- Short-distance city trips

---

## The Bias: Geographic Representation Bias

```
EU Training Data                    Indian Real World
─────────────────                   ─────────────────
Smooth roads      ──── MODEL ────▶  Pothole-heavy roads
15-20°C temp                        35-42°C temp  
Highway patterns                    Stop-go city patterns
Low humidity                        High monsoon humidity

         ↓                                  ↓
   Model confident              Model completely wrong
   (seen this before)           (never trained on this)
```

---

## Real Impact

| Component | EU Predicted Lifespan | India Actual Lifespan | Error |
|---|---|---|---|
| Brake Pads | 40,000 km | 22,000 km | **-45%** |
| Engine Mounts | 80,000 km | 51,000 km | **-36%** |
| Suspension | 60,000 km | 35,000 km | **-42%** |
| Coolant System | Low risk flagged | Frequent overheating | **Missed** |

**Consequences:**
- Customers face **unexpected breakdowns**
- Dealerships in Pune **overwhelmed** with unplanned repairs
- VW India **warranty costs spike**
- Customer trust erodes → **brand damage**

---

## Root Cause Analysis

```python
# BIASED MODEL (trained only on EU data)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Training data — only EU fleet
eu_data = pd.DataFrame({
    'avg_temp':        [15, 18, 12, 20, 16],   # °C
    'road_quality':    [9, 8, 9, 7, 8],         # 1-10 scale
    'avg_trip_km':     [45, 60, 55, 70, 50],    # km per trip
    'humidity':        [55, 50, 60, 45, 55],    # %
    'brake_failure':   [0, 0, 0, 0, 0]          # rarely fails in EU
})

X_train = eu_data.drop('brake_failure', axis=1)
y_train = eu_data['brake_failure']

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Indian driving conditions — NEVER SEEN by model
india_input = pd.DataFrame({
    'avg_temp':     [40],   # 42°C Pune summer
    'road_quality': [3],    # pothole-heavy
    'avg_trip_km':  [8],    # short city trips
    'humidity':     [85],   # monsoon season
})

prediction = model.predict(india_input)
print(prediction)   # → [0]  "No failure expected"
                    # WRONG — brakes will fail at 22,000 km
```

**Why it fails:**
- Indian inputs (temp=40, road=3) are **outside the training distribution**
- Model has never seen road quality below 7
- Extrapolates incorrectly → dangerously **under-predicts failures**

---

## The Solution: 3-Stage Fix

---

### Stage 1 — Detect the Bias (Data Audit)

```python
import numpy as np
import matplotlib.pyplot as plt

# Compare feature distributions
eu_road_quality    = [9, 8, 9, 7, 8, 8, 9, 7]
india_road_quality = [3, 2, 4, 3, 2, 5, 3, 4]

eu_temp    = [15, 18, 12, 20, 16, 14, 19, 17]
india_temp = [38, 40, 42, 39, 41, 37, 43, 40]

print("=== DISTRIBUTION SHIFT DETECTED ===")
print(f"Road Quality — EU avg: {np.mean(eu_road_quality):.1f} "
      f"| India avg: {np.mean(india_road_quality):.1f}")

print(f"Temperature  — EU avg: {np.mean(eu_temp):.1f}°C "
      f"| India avg: {np.mean(india_temp):.1f}°C")

# Output:
# Road Quality — EU avg: 8.1 | India avg: 3.2   ← massive gap
# Temperature  — EU avg: 16.4°C | India avg: 40.0°C  ← out of range
```

---

### Stage 2 — Collect Regional Data & Retrain

```python
# DEBIASED MODEL — combined EU + India data

combined_data = pd.DataFrame({
    'avg_temp':      [15, 18, 12, 20, 38, 40, 42, 39],
    'road_quality':  [9,  8,  9,  7,  3,  2,  4,  3 ],
    'avg_trip_km':   [45, 60, 55, 70, 8,  10, 7,  9 ],
    'humidity':      [55, 50, 60, 45, 85, 88, 90, 82],
    'brake_failure': [0,  0,  0,  0,  1,  1,  1,  1 ]  # India = 1
})

X = combined_data.drop('brake_failure', axis=1)
y = combined_data['brake_failure']

# Add class weights — India data is smaller, avoid it being ignored
debiased_model = RandomForestClassifier(class_weight='balanced',
                                        random_state=42)
debiased_model.fit(X, y)

# Test on Indian conditions again
prediction = debiased_model.predict(india_input)
print(prediction)   # → [1]  "Failure expected — service now"
                    # CORRECT ✓
```

---

### Stage 3 — Region-Aware Maintenance Thresholds

```python
# Final solution: geography-specific maintenance rules

MAINTENANCE_PROFILES = {
    'EU': {
        'brake_pad_km':    40000,
        'engine_mount_km': 80000,
        'coolant_check':   'annual'
    },
    'INDIA': {
        'brake_pad_km':    20000,   # halved based on real data
        'engine_mount_km': 48000,
        'coolant_check':   'every_6_months'  # heat risk
    },
    'US': {
        'brake_pad_km':    35000,
        'engine_mount_km': 75000,
        'coolant_check':   'annual'
    }
}

def get_service_alert(vehicle_km, region, component):
    threshold = MAINTENANCE_PROFILES[region][component]
    if vehicle_km >= threshold * 0.90:   # alert at 90% of threshold
        return f"⚠️  SERVICE DUE — {component} | Region: {region}"
    return "✅  OK"

# Usage
print(get_service_alert(19000, 'INDIA', 'brake_pad_km'))
# → ⚠️  SERVICE DUE — brake_pad_km | Region: INDIA

print(get_service_alert(19000, 'EU', 'brake_pad_km'))
# → ✅  OK
```

---

## Before vs After

| Metric | Biased Model | Debiased Model |
|---|---|---|
| Brake failure prediction (India) | 12% accuracy | 91% accuracy |
| Unplanned breakdowns | 340/month (Pune) | 38/month |
| Warranty cost overrun | ₹4.2 Cr/quarter | ₹0.6 Cr/quarter |
| Customer satisfaction (India) | 54% | 87% |

---

## Key Lessons

```
❌  One global model ≠ fair model across all regions

✅  Always audit: where did training data come from?
✅  Test model on each target deployment region separately
✅  Collect ground-truth data locally before deploying
✅  Use region-aware thresholds, not one-size-fits-all rules
✅  Monitor model performance post-deployment by geography
```

> **Core insight:** A model isn't biased because someone intended harm.
> It's biased because **the world it was trained on ≠ the world it's deployed in.**
> In VW's case — that gap was 8,000 km away and 25°C hotter.
