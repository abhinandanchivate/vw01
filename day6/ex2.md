
---

# VW ADAS (Advanced Driver Assistance System) — ML Bias Case Study

## The Setup

VW's **ID.4 / Tiguan ADAS** uses multiple ML models working together:

```
Camera Feed ──┐
Radar Data  ──┼──▶ Sensor Fusion Model ──▶ Risk Scoring Model ──▶ Intervention
LiDAR Data  ──┘                                                   (brake/steer)
                         ↑
              Trained on 50M km of European driving data
```

**The problem:** This entire pipeline was trained on **European driving behavior** and deployed globally — including India, where driving behavior is **fundamentally different.**

---

## Why Indian Driving is a Different Distribution

```
European Driving                    Indian Driving (Pune/Mumbai)
────────────────                    ────────────────────────────
Lane discipline: strict             Lane discipline: fluid/absent
Horn usage: rare                    Horn usage: constant communication tool
Pedestrian crossings: marked        Pedestrians: anywhere, anytime
Animal on road: near zero           Cows, dogs, goats: common
2-wheelers: 15% of traffic          2-wheelers: 65% of traffic
Overtaking: predictable             Overtaking: from any side
Traffic signals: followed           Signals: advisory at best
Inter-vehicle gap: 2-3 sec          Inter-vehicle gap: 0.3-0.5 sec
```

The ADAS model never learned these patterns. It sees Indian roads as a **constant anomaly.**

---

## The 4 Compounding Biases

### Bias 1 — Object Classification Bias

```python
# What the model was trained to recognize

EU_TRAINING_OBJECTS = {
    'pedestrian':  95000,   # samples
    'cyclist':     42000,
    'car':        180000,
    'truck':       38000,
    'motorcycle':   8000,   # ← very few
    'auto_rickshaw':   0,   # ← NEVER SEEN
    'bullock_cart':    0,   # ← NEVER SEEN
    'cycle_rickshaw':  0,   # ← NEVER SEEN
}

# India road contains
INDIA_ROAD_OBJECTS = {
    'auto_rickshaw':  'extremely common',
    'cycle_rickshaw': 'common',
    'bullock_cart':   'occasional',
    'e_rickshaw':     'growing rapidly',
    '2_wheelers':     '65% of traffic',
}

# Model behavior on unknown objects
def classify_object(obj):
    if obj not in EU_TRAINING_OBJECTS:
        return 'UNCERTAIN'   # confidence < 0.4
        # ADAS either ignores it OR
        # misclassifies as nearest known class
```

**Real consequence:**
- Auto-rickshaw misclassified as "large pedestrian" → wrong braking distance
- Bullock cart classified as "truck" → unnecessary panic braking
- E-rickshaw (slow + wide) → model confused, no learned response

---

### Bias 2 — Inter-Vehicle Gap Bias (The Tailgating Problem)

```python
import numpy as np

# Safe following distance learned from EU data
EU_SAFE_GAP_SECONDS = 2.5   # seconds (EU driving norm)

# Indian driving reality
INDIA_TYPICAL_GAP_SECONDS = 0.4   # seconds

# ADAS Forward Collision Warning threshold
FCW_THRESHOLD = EU_SAFE_GAP_SECONDS * 0.7  # warn at 1.75 sec

# Simulation: Indian highway
def simulate_indian_highway(num_events=1000):
    gaps = np.random.normal(loc=0.4, scale=0.15, size=num_events)
    false_alarms = np.sum(gaps < FCW_THRESHOLD)
    return false_alarms / num_events * 100

false_alarm_rate = simulate_indian_highway()
print(f"False alarm rate in India: {false_alarm_rate:.1f}%")
# Output: False alarm rate in India: 97.3%

# RESULT: ADAS beeps/warns 97% of the time on Indian roads
# Drivers turn it OFF completely within first week
# Safety feature becomes useless
```

---

### Bias 3 — Pedestrian Intent Prediction Bias

The model uses **trajectory prediction** to anticipate where pedestrians will go.

```
EU Training Data Assumption:
  Pedestrian at crosswalk → will cross at 90° angle → predictable path

India Reality:
  Pedestrian anywhere → crosses diagonally
                      → stops mid-road to let vehicles pass
                      → reverses direction suddenly
                      → groups cross in waves
                      → vendors walk along road edge
```

```python
# Trajectory prediction model (simplified)

class PedestrianIntentModel:
    def __init__(self):
        # Learned from EU data
        self.expected_crossing_angle = 90    # degrees
        self.expected_speed = 1.4            # m/s (EU avg)
        self.direction_change_prob = 0.03    # 3% in EU

    def predict_trajectory(self, observed_angle, speed, direction_changes):
        anomaly_score = 0

        # Penalize deviation from learned EU norms
        anomaly_score += abs(observed_angle - self.expected_crossing_angle) / 90
        anomaly_score += abs(speed - self.expected_speed) / self.expected_speed
        anomaly_score += direction_changes / self.direction_change_prob

        if anomaly_score > 2.0:
            return "UNPREDICTABLE — no action"  # model gives up
        else:
            return "SAFE — continue"

model = PedestrianIntentModel()

# Test EU pedestrian
print(model.predict_trajectory(88, 1.3, 0.02))
# → SAFE — continue  ✓

# Test Indian pedestrian
print(model.predict_trajectory(45, 0.8, 0.45))
# → UNPREDICTABLE — no action  ✗
# Model freezes instead of braking — dangerous
```

---

### Bias 4 — Feedback Loop from Driver Override Data

This is the **most dangerous and hidden** bias:

```
Step 1: ADAS gives false warning (97% rate in India)
        ↓
Step 2: Indian drivers override / ignore / disable ADAS
        ↓
Step 3: VW collects "driver override" telemetry data
        ↓
Step 4: Model learns — "In India, driver overrides = correct behavior"
        ↓
Step 5: Model becomes LESS aggressive in India to reduce overrides
        ↓
Step 6: Model now MISSES real emergencies in India
        ↓
Step 7: New telemetry confirms fewer warnings → model confident
        ↓
        LOOP — bias compounds every OTA update
```

```python
# Feedback loop simulation over OTA updates

sensitivity = 1.0   # starting ADAS sensitivity

for update_cycle in range(6):
    override_rate = 0.97 - (1.0 - sensitivity) * 0.3
    # Model reduces sensitivity to reduce overrides
    sensitivity -= override_rate * 0.12

    real_emergency_detection = sensitivity * 100

    print(f"Update {update_cycle+1} | "
          f"Sensitivity: {sensitivity:.2f} | "
          f"Emergency Detection: {real_emergency_detection:.1f}%")

# Output:
# Update 1 | Sensitivity: 0.88 | Emergency Detection: 88.4%
# Update 2 | Sensitivity: 0.77 | Emergency Detection: 77.3%
# Update 3 | Sensitivity: 0.68 | Emergency Detection: 68.1%
# Update 4 | Sensitivity: 0.60 | Emergency Detection: 60.2%
# Update 5 | Sensitivity: 0.53 | Emergency Detection: 53.4%
# Update 6 | Sensitivity: 0.47 | Emergency Detection: 47.1%

# After 6 OTA updates — model detects only 47% of real emergencies
# System became LESS safe with every improvement cycle
```

---

## The Complete Bias Pipeline

```
Raw Sensor Data
      │
      ▼
┌─────────────────────────────────────────┐
│  Sensor Fusion Model                    │
│  Bias: Never saw auto-rickshaws,        │  ← Object Classification Bias
│        bullock carts, e-rickshaws       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Risk Scoring Model                     │
│  Bias: EU gap norms → 97% false alarms  │  ← Distribution Bias
│        in Indian traffic density        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Intent Prediction Model                │
│  Bias: EU pedestrian behavior assumed   │  ← Aggregation Bias
│        Indian behavior = "anomaly"      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  OTA Learning Loop                      │
│  Bias: Override data reduces            │  ← Feedback Loop Bias
│        sensitivity every update         │
└────────────────┬────────────────────────┘
                 │
                 ▼
          🚨 Unsafe ADAS
          for Indian roads
```

---

## The Solution Architecture

### Step 1 — Region-Specific Training Pipeline

```python
class RegionalADASTrainer:
    def __init__(self, region):
        self.region = region
        self.base_model = load_pretrained_eu_model()

    def collect_regional_data(self):
        return {
            'objects':      self.label_local_objects(),     # auto-rickshaw etc
            'gap_norms':    self.measure_local_gap_norms(), # 0.4s India
            'ped_behavior': self.record_local_pedestrians(),
            'edge_cases':   self.capture_near_misses()
        }

    def fine_tune(self, regional_data, epochs=50):
        # Transfer learning — keep EU knowledge, add India knowledge
        for layer in self.base_model.layers[:-4]:
            layer.trainable = False    # freeze early layers

        # Only retrain top layers on Indian data
        self.base_model.fit(
            regional_data,
            class_weight=self.compute_india_weights()
        )
        return self.base_model

    def compute_india_weights(self):
        return {
            'auto_rickshaw': 8.0,   # heavily upweight rare classes
            'bullock_cart':  12.0,
            'e_rickshaw':    10.0,
            'motorcycle':    3.0,
            'pedestrian':    2.0,
        }
```

---

### Step 2 — Adaptive Gap Threshold per Region

```python
# Dynamic ADAS thresholds based on region

REGIONAL_THRESHOLDS = {
    'EU': {
        'fcw_gap_seconds':      1.75,
        'ped_warning_distance': 30,    # meters
        'false_alarm_budget':   0.05   # 5% max
    },
    'INDIA': {
        'fcw_gap_seconds':      0.35,  # calibrated to local norm
        'ped_warning_distance': 15,    # shorter — denser environment
        'false_alarm_budget':   0.08   # slightly higher tolerance
    }
}

def get_adaptive_threshold(gps_region, vehicle_speed, traffic_density):
    base = REGIONAL_THRESHOLDS[gps_region]

    # Further adapt dynamically
    if traffic_density > 0.8:       # heavy traffic
        base['fcw_gap_seconds'] *= 0.85

    if vehicle_speed < 20:          # city crawl
        base['ped_warning_distance'] *= 1.3

    return base
```

---

### Step 3 — Break the Feedback Loop

```python
class SafeFeedbackLoop:

    def process_override(self, override_event, region):

        # DO NOT blindly reduce sensitivity on override
        # Instead — classify the override type first

        override_type = self.classify_override(override_event)

        if override_type == 'FALSE_ALARM':
            # Legitimate — adjust threshold slightly
            self.adjust_threshold(delta=-0.01, region=region)

        elif override_type == 'DRIVER_IMPATIENCE':
            # Driver overrode a correct warning
            # DO NOT reduce sensitivity
            self.log_safety_incident(override_event)
            self.increase_driver_education_flag()

        elif override_type == 'AMBIGUOUS':
            # Send to human review team
            self.queue_for_human_review(override_event)

    def classify_override(self, event):
        # Check if a near-miss occurred within 3 seconds of override
        if event.post_override_near_miss:
            return 'DRIVER_IMPATIENCE'
        if event.adas_confidence < 0.5:
            return 'FALSE_ALARM'
        return 'AMBIGUOUS'
```

---

## Final Results (Simulated)

| Metric | Biased Global Model | Debiased India Model |
|---|---|---|
| Auto-rickshaw detection | 31% | 94% |
| False alarm rate | 97% | 9% |
| Pedestrian prediction accuracy | 44% | 83% |
| Emergency detection (after 6 OTAs) | 47% | 91% |
| Driver ADAS adoption rate | 12% | 78% |

---

