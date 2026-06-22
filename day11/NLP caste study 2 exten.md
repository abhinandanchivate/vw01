# Additional  NLP Tasks

## Task 13: Cross-Document Incident Linking

A single vehicle incident may be described across multiple documents:

* Customer complaint
* Dealer response
* Technician report
* Diagnostic log
* Roadside-assistance report
* Warranty claim

The system must determine which documents belong to the same incident.

### Example inputs

**Customer complaint**

```text
My Tata Nexon EV lost power near Lonavala on 18 June 2026.
```

**Dealer report**

```text
Vehicle MH12AB4587 was received for battery-warning diagnosis.
```

**Diagnostic log**

```text
Service ID SER-98754
Fault code: BMS-P0A80
Timestamp: 18 June 2026
```

### Expected output

```json
{
  "incident_id": "INC-10021",
  "linked_documents": [
    "CUSTOMER-COMP-457",
    "DEALER-REPORT-985",
    "DIAGNOSTIC-LOG-784"
  ],
  "link_confidence": 0.94
}
```

### Complexity

The documents may not contain the same identifiers. The system must use:

* Vehicle model
* Registration number
* Date
* Dealer
* Location
* Diagnostic code
* Semantic similarity

---

## Task 14: Entity Resolution and Alias Matching

The same entity may be written differently across reports.

### Examples

```text
Sunrise Motors Pune
Sunrise Motors Pvt. Ltd.
Sunrise Pune Workshop
Sunrise Service Centre
```

These may all refer to the same dealer.

Similarly:

```text
Tata Nexon EV
Nexon Electric
Nexon EV Max
Tata Electric Nexon
```

The NLP system must determine whether different names represent the same real-world entity.

### Expected output

```json
{
  "canonical_entity": "Sunrise Motors Pune",
  "entity_type": "DEALER",
  "aliases": [
    "Sunrise Motors Pvt. Ltd.",
    "Sunrise Pune Workshop",
    "Sunrise Service Centre"
  ],
  "resolution_confidence": 0.91
}
```

### Required techniques

* Fuzzy string matching
* Embedding similarity
* Address matching
* Knowledge-base lookup
* Entity clustering
* Context comparison

---

## Task 15: Causal Chain Extraction

The system should identify the sequence of causes and effects in an incident.

### Input

```text
The battery temperature increased rapidly during charging.
The BMS disabled the power system, causing the vehicle to stop.
Because the vehicle stopped in traffic, another car hit it from behind.
```

### Expected causal chain

```text
Battery temperature increase
        ↓ caused
BMS power-system shutdown
        ↓ caused
Vehicle stopped in traffic
        ↓ contributed to
Rear-end collision
```

### Structured output

```json
{
  "causal_chain": [
    {
      "cause": "Battery temperature increase",
      "effect": "BMS power-system shutdown"
    },
    {
      "cause": "BMS power-system shutdown",
      "effect": "Vehicle stopped in traffic"
    },
    {
      "cause": "Vehicle stopped in traffic",
      "effect": "Rear-end collision"
    }
  ]
}
```

### Complexity

The system must distinguish between:

* Cause
* Symptom
* Result
* Coincidental event
* Customer assumption
* Technician-confirmed cause

---

## Task 16: Evidence-Based Root-Cause Confidence

The system must not only suggest a root cause; it must show the supporting evidence and confidence.

### Input evidence

```text
Customer: Vehicle lost power.

Diagnostic log: BMS-P0A80 detected.

Technician: Battery-cell imbalance observed.

Dealer: Battery module replaced.

Customer: Same issue returned after two days.
```

### Expected output

```json
{
  "suspected_root_cause": "Battery-cell degradation",
  "confidence": 0.87,
  "supporting_evidence": [
    "BMS-P0A80 diagnostic code",
    "Battery-cell imbalance detected",
    "Repeated failure after battery-module replacement"
  ],
  "conflicting_evidence": [
    "Dealer marked the vehicle safe after repair"
  ]
}
```

### Important requirement

The system must separate:

* Confirmed fact
* Customer observation
* Technician conclusion
* NLP inference
* Missing evidence

---

## Task 17: Repair Effectiveness Analysis

The system must determine whether a previous repair actually solved the issue.

### Input timeline

```text
10 June 2026: Customer reported battery warning.
12 June 2026: Dealer replaced the battery sensor.
14 June 2026: Vehicle returned to customer.
18 June 2026: Battery warning appeared again.
20 June 2026: Vehicle lost power.
```

### Expected output

```json
{
  "repair_action": "Battery sensor replacement",
  "repair_date": "12 June 2026",
  "issue_reoccurred": true,
  "days_until_recurrence": 6,
  "repair_effectiveness": "Unsuccessful",
  "recommended_next_step": "Perform battery-module and BMS diagnostics"
}
```

### Complexity

The model must understand:

* Previous issue
* Repair action
* Time gap
* Repeated symptom
* Whether the recurrence is identical or related

---

## Task 18: Counterfactual Safety Reasoning

The system should estimate what might have happened if a particular safety action had not been taken.

### Input

```text
The customer stopped the vehicle immediately after smoke appeared
from the battery compartment.
```

### Expected reasoning

```text
Observed action:
Customer stopped the vehicle.

Possible avoided outcome:
Continued driving may have increased battery temperature and fire risk.
```

### Structured output

```json
{
  "observed_action": "Vehicle stopped immediately",
  "possible_avoided_risk": "Battery fire or further thermal damage",
  "counterfactual_confidence": 0.73,
  "evidence_type": "Safety-rule inference"
}
```

### Important restriction

Counterfactual conclusions must be marked as **inferences**, not confirmed facts.

---

## Task 19: Uncertainty and Ambiguity Detection

Customer complaints may contain uncertain language.

### Example

```text
I think the smoke may have come from the battery,
but I am not completely sure.
```

The system should detect uncertainty.

### Expected output

```json
{
  "claim": "Smoke originated from the battery",
  "certainty": "Low",
  "uncertainty_phrases": [
    "I think",
    "may have",
    "not completely sure"
  ],
  "requires_confirmation": true
}
```

### Other uncertainty expressions

```text
possibly
maybe
appears to be
seems like
might have
I am not sure
probably
```

This prevents the system from treating every customer assumption as a confirmed technical fact.

---

## Task 20: Model Explanation and Decision Trace

For every critical prediction, the system must explain why the incident was classified as critical.

### Input

```text
The brakes stopped responding at 90 km/h,
and the vehicle nearly hit another car.
```

### Expected output

```json
{
  "predicted_category": "Brake System Failure",
  "severity": 5,
  "urgency": "Critical",
  "decision_factors": [
    {
      "phrase": "brakes stopped responding",
      "impact": "Critical braking failure"
    },
    {
      "phrase": "90 km/h",
      "impact": "High-speed incident"
    },
    {
      "phrase": "nearly hit another car",
      "impact": "Potential collision"
    }
  ]
}
```

### Complexity

The explanation should combine:

* Model probabilities
* Extracted entities
* Safety rules
* Attention or feature importance
* Business thresholds

---

## Task 21: Active Learning for Human Review

The system should identify complaints for which the model is uncertain and send them to human experts.

### Example prediction

```text
Engine Issue: 0.36
Battery Issue: 0.33
Transmission Issue: 0.31
```

The model should not automatically assign a category.

### Expected output

```json
{
  "incident_id": "INC-10045",
  "automatic_decision": false,
  "review_required": true,
  "reason": "Low classification confidence",
  "candidate_categories": [
    "Engine Issue",
    "Battery Issue",
    "Transmission Issue"
  ]
}
```

### Human feedback cycle

```text
Low-confidence complaint
        ↓
Expert review
        ↓
Correct label
        ↓
Training dataset update
        ↓
Model retraining
```

---

## Task 22: NLP Model Drift Detection

Customer language and vehicle technology change over time.

New terms may appear:

```text
OTA update
ADAS malfunction
regenerative braking
thermal runaway
L2 driving assistance
digital key
battery preconditioning
```

The system must identify when the production model no longer performs well.

### Drift indicators

* Increase in unknown words
* Reduction in model confidence
* New diagnostic codes
* New vehicle models
* New complaint categories
* Increase in human corrections
* Change in language distribution

### Expected output

```json
{
  "drift_detected": true,
  "drift_type": "New terminology",
  "new_terms": [
    "battery preconditioning",
    "digital key failure",
    "ADAS calibration"
  ],
  "recommended_action": "Update taxonomy and retrain the model"
}
```

---

# Updated Advanced Task List

| Task | Complex NLP requirement         | Main output                                 |
| ---- | ------------------------------- | ------------------------------------------- |
| 13   | Cross-document incident linking | Connected reports belonging to one incident |
| 14   | Entity resolution               | Canonical dealer, model and component names |
| 15   | Causal-chain extraction         | Cause-and-effect incident sequence          |
| 16   | Evidence-based root cause       | Root cause with evidence and confidence     |
| 17   | Repair-effectiveness analysis   | Successful or unsuccessful repair           |
| 18   | Counterfactual safety reasoning | Possible avoided risk                       |
| 19   | Uncertainty detection           | Confirmed versus uncertain claims           |
| 20   | Explainable NLP decisions       | Reasons behind category and severity        |
| 21   | Active learning                 | Human review for uncertain predictions      |
| 22   | Model-drift detection           | Need for taxonomy or model updates          |

These additions make the project an **enterprise-level NLP and incident reasoning platform**, rather than only a text-classification application.
