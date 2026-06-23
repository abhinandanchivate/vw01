Here's a clean problem set — participants read the scenario and write both the prompt and the solution themselves.

---

# VW Predictive Maintenance — Problem Set

### Instructions for participants

Each problem gives you a real-world scenario, input data, expected output behaviour, and constraints. You must:

1. Write a prompt that would instruct an LLM to generate the complete solution
2. Use that prompt to generate the code and verify it runs correctly

You will be evaluated on the quality of your prompt as much as the quality of the code it produces.

---

## Problem 1 — Raw input validation

**Difficulty:** Beginner
**Domain:** Data quality, defensive programming

### Scenario

Vehicle intake forms arrive from 340 Volkswagen service centres across India. Technicians fill these manually — values are sometimes missing, out of range, or entered in the wrong unit. Before any ML model sees the data, every record must pass a validation gate.

You are a Senior ML Engineer. Your job is to build the validation module.

### Input

A plain Python dictionary (not a typed dataclass) representing one vehicle record:

```python
{
    "model_year": 2022,
    "mileage_km": 78500,
    "months_since_service": 14,
    "engine_alerts": 4,
    "brake_wear_pct": 81.0,
    "battery_voltage_v": 11.4,
    "measurement_context": "unknown"
}
```

### Expected behaviour

Your module must catch and correctly classify each of these real-world data problems:

- `mileage_km` between 1 and 500 — likely entered in miles, not km. Flag it but do not reject the record.
- `brake_wear_pct` above 100 — physically impossible. Reject.
- `battery_voltage_v` is exactly 0.0 — sensor failure, not a real reading. Treat as missing.
- `measurement_context` is not one of `"off"`, `"cranking"`, `"running"`, `"unknown"` — flag as invalid.
- Any field that is `None`, absent from the dict, or an empty string — flag as missing.

### Output contract

The module must return a structured report — not a print statement, not a raw dict. The report must tell the caller: which fields passed, which warned, which failed, whether the overall record is valid, and the worst severity seen.

### Constraints

- Standard library only
- All numeric thresholds defined as module-level constants
- Function signature: `validate(raw: dict) -> ValidationReport`
- Record is invalid if any field is `"Critical"`
- Must include at least 5 unit tests covering the edge cases above

---

## Problem 2 — Feature engineering pipeline

**Difficulty:** Beginner–Intermediate
**Domain:** Feature engineering, ML preprocessing

### Scenario

Raw vehicle fields carry little predictive signal on their own. A 14-month service gap means different things for a 3-year-old car versus a 10-year-old one. 78,500 km means different things for a vehicle that is 2 years old versus 6 years old. Your job is to transform raw inputs into ML-ready derived features.

### Input

A validated `VehicleInput` dataclass with these fields:

```python
model_year                  = 2022
mileage_km                  = 78500
months_since_service        = 14
engine_alerts               = 4
brake_wear_pct              = 81.0
battery_voltage_v           = 11.4
measurement_context         = "unknown"
customer_complaints         = [
    "Vehicle takes longer to start",
    "Braking noise is heard",
    "Fuel efficiency has reduced"
]
current_year                = 2025
```

### Required derived features

You must compute all of the following. Each one must be normalized to `[0.0, 1.0]` before it enters a weighted sum:

| Feature | Description |
|---|---|
| `vehicle_age_years` | How old the vehicle is |
| `avg_annual_mileage` | Mileage divided by age — flags unusually high-use vehicles |
| `service_delay_score` | How far overdue the service is, normalized to a severity scale |
| `brake_severity_score` | Wear percentage mapped to a risk scale with a threshold |
| `engine_alert_score` | Alert count normalized to a ceiling |
| `battery_risk_score` | Depends on measurement context — must be `0.5` when context is unknown |
| `complaint_score` | Count of complaints normalized to a ceiling |
| `safety_signal` | `1.0` if any complaint contains a safety-related keyword, else `0.0` |
| `combined_risk_score` | Weighted sum of all above scores |

### Output contract

A typed `EngineeredFeatures` dataclass containing all derived features plus the combined score. The combined score must be between `0.0` and `1.0`.

### Constraints

- All weights and thresholds in a single constants block at the top of the file
- Weights must sum to exactly `1.0` — enforce this with an assertion at import time
- `battery_risk_score` must never produce a confirmed low/high reading when context is `"unknown"`
- Must include unit tests verifying: score range, weight sum assertion, battery context handling, safety signal detection

---

## Problem 3 — Heuristic risk scorer with safety floors

**Difficulty:** Intermediate
**Domain:** Risk scoring, business rules, safety-critical systems

### Scenario

Your feature pipeline produces a `combined_risk_score` between 0 and 1. But a raw weighted average is not enough for a safety-adjacent system. A vehicle with critical brake wear and a braking noise complaint must never be classified as Medium Risk, even if other features pull the score down. You need a scorer that applies business rule overrides on top of the base probability.

### Input

An `EngineeredFeatures` dataclass produced by Problem 2.

### Required behaviour

Start with `base_probability = combined_risk_score`. Then apply these overrides in order:

1. If `safety_signal == 1.0` → probability must be at least `0.55`
2. If `brake_severity_score >= 0.8` AND `safety_signal == 1.0` → probability must be at least `0.72`
3. If `engine_alert_score >= 0.8` AND `battery_risk_score >= 0.6` → probability must be at least `0.65`
4. Final probability must be clamped to `[0.0, 1.0]`

Classify using these thresholds:

| Probability | Category |
|---|---|
| Below 0.40 | Low Risk |
| 0.40 – 0.69 | Medium Risk |
| 0.70 and above | High Risk |

Confidence must be `"Medium"` when `measurement_context` was `"unknown"`, `"High"` when probability is above `0.80` or below `0.20`, and `"Medium"` otherwise.

### Output contract

A `RiskPrediction` dataclass with: `probability`, `risk_category`, `confidence`, `prediction_horizon_days` (always 30), and `prediction_type` (always `"Heuristic estimate — not a trained production model"`).

### Constraints

- Safety floors applied in the exact order listed above — order matters
- Floor values must be constants, not magic numbers
- `prediction_type` string must never be altered
- Must include unit tests for: each floor trigger condition, clamp boundary (probability never exceeds 1.0), correct classification at threshold boundaries (0.40 and 0.70 exactly)

---

## Problem 4 — Inspection recommendation engine

**Difficulty:** Intermediate
**Domain:** Rule-based reasoning, business logic

### Scenario

After a risk score is produced, the service advisor needs to know exactly which systems to inspect and in what order. Inspections are not a fixed list — they are prioritised dynamically based on which signals are active. A vehicle with critical brake wear gets brake inspections first. A vehicle with engine alerts gets a DTC scan first. Battery context unknown means charging system inspection is always included.

### Input

Three objects together:

```python
vehicle: VehicleInput        # raw inputs
features: EngineeredFeatures # derived features
prediction: RiskPrediction   # risk score and category
```

### Required behaviour

The engine must produce a prioritised inspection list where:

- Items are ordered by urgency — safety-critical systems always appear first
- Each item has a `system`, an `action` (specific test to perform), and a `priority` (`"Urgent"` / `"Routine"`)
- These systems are always included regardless of inputs: brake pads, brake discs, brake fluid, scheduled maintenance items, tyres
- These systems are conditionally included:
  - Battery condition and charging system — always if `battery_risk_score > 0.3` or `measurement_context == "unknown"`
  - DTC scan and engine alerts — always if `engine_alert_score > 0.0`
  - Fuel system and air intake — if `engine_alert_score >= 0.4`
  - Ignition system — if any complaint mentions starting difficulty
  - Emission system — if any complaint mentions fuel efficiency

### Output contract

A list of `InspectionItem` dataclasses ordered by priority descending, then alphabetically within the same priority. The list must never be empty.

### Constraints

- No inspection item may recommend replacing a component — only inspecting and testing
- Priority ordering must be deterministic (same inputs always produce same order)
- Must include unit tests for: conditional inclusion logic, ordering correctness, no replacement language in any action string, non-empty output guarantee

---

## Problem 5 — Customer-facing report generator

**Difficulty:** Intermediate
**Domain:** Natural language generation, report formatting

### Scenario

The risk score and inspection list need to be communicated to two audiences: the service advisor (technical) and the vehicle owner (non-technical). Your report generator must produce both from the same inputs, with tone, vocabulary, and detail level adjusted for each audience.

### Input

```python
vehicle: VehicleInput
features: EngineeredFeatures
prediction: RiskPrediction
inspections: list[InspectionItem]
safety: dict   # keys: safety_symptoms, avoid_driving, driving_recommendation
```

### Required behaviour

Produce two reports from a single function call:

**Technical report** (for service advisor):
- Lists all engineered feature values and their risk impact
- Shows the top 5 risk contributors with observed values
- States the heuristic probability and confidence level
- Lists all inspections with priority and specific test action
- Includes a limitations section

**Customer report** (for vehicle owner):
- Uses plain language — no ML terminology, no feature names
- Eight sections: risk category, probability as plain percentage, major observations, why this risk level, recommended checks, driving advice, timeline, limitations
- Must include this exact disclaimer: `"This is an ML-assisted estimate. An authorized Volkswagen technician must inspect your vehicle before any action is taken."`
- Driving advice must differ clearly between `avoid_driving = True` and `avoid_driving = False`

### Output contract

A `ReportBundle` dataclass with two string fields: `technical_report` and `customer_report`. Both must be non-empty strings.

### Constraints

- No hardcoded vehicle-specific strings — all values interpolated from inputs
- Disclaimer text must appear verbatim in the customer report
- Must include unit tests for: disclaimer presence, all 8 sections present in customer report, driving advice differs between avoid/no-avoid cases, no ML jargon in customer report (check for absence of words: `"heuristic"`, `"probability"`, `"feature"`, `"score"`)

---

## Problem 6 — Full pipeline integration

**Difficulty:** Intermediate–Advanced
**Domain:** System design, pipeline orchestration, CLI

### Scenario

Problems 1 through 5 are individual modules. Now assemble them into a single runnable pipeline with a CLI entry point, structured project layout, configuration via environment variables, and a test suite that runs the full pipeline end to end.

### Required project structure

```
vw_maintenance/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── settings.py
├── vw_maintenance/
│   ├── __init__.py
│   ├── models/
│   │   └── schemas.py
│   ├── core/
│   │   ├── validator.py
│   │   ├── features.py
│   │   ├── predictor.py
│   │   ├── inspector.py
│   │   └── reporter.py
│   ├── utils/
│   │   └── formatting.py
│   └── cli.py
└── tests/
    ├── test_validator.py
    ├── test_features.py
    ├── test_predictor.py
    ├── test_inspector.py
    ├── test_reporter.py
    └── test_pipeline.py
```

### Required behaviour

- `python -m vw_maintenance.cli` runs the full pipeline on the Taigun test case
- `VW_OUTPUT_FORMAT=json python -m vw_maintenance.cli` prints only the JSON output
- `python -m pytest tests/ -v` runs all tests and all pass
- `test_pipeline.py` must include an end-to-end test that runs the full pipeline from raw dict input to final report and asserts: risk category is `"High Risk"`, probability is `>= 0.70`, customer report contains the disclaimer, inspection list is non-empty

### Constraints

- Standard library only (pytest for tests)
- All constants in `config/settings.py` — never duplicated across modules
- Every module imports constants from `config.settings`, never redefines them
- `cli.py` must support both text and JSON output modes via environment variable
- `README.md` must include installation steps, usage, project structure, and a limitations section

---

## Problem 7 — Confidence calibration and uncertainty reporting

**Difficulty:** Advanced
**Domain:** Probability calibration, uncertainty quantification, ML engineering

### Scenario

The current heuristic scorer produces a single probability number. But stakeholders — both service advisors and VW engineering leads — need to understand how reliable that number is. A score of `0.71` means something very different when the battery measurement context is known versus unknown, or when all 7 input signals agree versus when they conflict.

Your job is to add a calibration and uncertainty layer on top of the existing scorer.

### Input

```python
features: EngineeredFeatures
prediction: RiskPrediction
validation: list[ValidationResult]
```

### Required behaviour

Compute three things:

**1. Epistemic uncertainty score** (`0.0` to `1.0`)
Measures how much missing or ambiguous input data reduces confidence. Increases when:
- `measurement_context == "unknown"` (+0.15)
- Any `ValidationResult` has status `"Warning"` (+0.05 each, max +0.20)
- Any `ValidationResult` has status `"Critical"` (+0.10 each, max +0.30)
- `complaint_count == 0` (+0.10)

**2. Aleatory uncertainty score** (`0.0` to `1.0`)
Measures natural variability — how much the input signals disagree with each other. Compute the standard deviation of the 7 normalized sub-scores from `EngineeredFeatures`. Higher standard deviation means signals are conflicting. Normalize to `[0.0, 1.0]` using a fixed ceiling of `0.5`.

**3. Calibrated probability interval**
Adjust the point estimate into a range:
- Lower bound: `max(0.0, probability - epistemic_uncertainty * 0.3)`
- Upper bound: `min(1.0, probability + epistemic_uncertainty * 0.3)`
- Report as `(lower, upper)` tuple rounded to 3 decimal places

### Output contract

A `CalibrationReport` dataclass with: `epistemic_uncertainty`, `aleatory_uncertainty`, `calibrated_lower`, `calibrated_upper`, `point_estimate`, and a `plain_language_summary` string that explains the uncertainty in one sentence without using the words `"epistemic"` or `"aleatory"`.

### Constraints

- All penalty weights defined as constants
- `plain_language_summary` must be generated programmatically — not hardcoded strings
- Must include unit tests for: interval always contains point estimate, bounds clamped to `[0.0, 1.0]`, summary never contains forbidden words, high missing data produces wider interval than low missing data

---

## Problem 8 — Production ML design document generator

**Difficulty:** Advanced
**Domain:** ML system design, production engineering

### Scenario

The heuristic system works for the pilot. Now VW India's ML platform team needs a production design document to build the real trained model. The document must cover every aspect of the production system: data pipeline, model selection, training strategy, evaluation metrics, deployment, monitoring, and retraining. It must be generated programmatically from a config — not written by hand — so it can be versioned and regenerated as decisions change.

### Input

A `ProductionConfig` dataclass specifying:

```python
target_variable         = "urgent_service_within_30_days"
positive_class_label    = "Urgent"
class_imbalance_ratio   = 0.15   # 15% of historical records were urgent
baseline_model          = "LogisticRegression"
advanced_models         = ["RandomForest", "XGBoost", "LightGBM"]
primary_metric          = "Recall"
false_negative_cost     = "High"
false_positive_cost     = "Low"
explainability_method   = "SHAP"
deployment_target       = "REST API"
retraining_trigger      = "monthly or when recall drops below 0.80"
```

### Required behaviour

Generate a structured Markdown document with these sections in order:

1. Business objective and target definition
2. Target variable and label strategy
3. Feature inventory (list all features from Problem 2 with type and source)
4. Data preprocessing steps
5. Missing value handling strategy
6. Outlier handling strategy
7. Class imbalance handling strategy
8. Model selection rationale (baseline + advanced, with tradeoffs)
9. Train-validation-test split strategy
10. Cross-validation strategy
11. Evaluation metrics (with justification for each, false negative cost prominently stated)
12. Probability calibration approach
13. Explainability approach
14. Deployment architecture
15. Model monitoring plan
16. Data drift detection approach
17. Prediction drift detection approach
18. Retraining criteria and process
19. Human approval requirements
20. Limitations and assumptions

### Output contract

A single string containing valid Markdown. Every section must be present. Section 11 must include a table of metrics. Section 8 must include a comparison table of models.

### Constraints

- Every section generated from `ProductionConfig` values — no hardcoded model names or metric names
- Section 11 metric table must include: Recall, Precision, F1, PR-AUC, ROC-AUC, Brier Score, Calibration Curve, False-Negative Cost
- Must include unit tests for: all 20 sections present, model names from config appear in section 8, metric names appear in section 11, false negative cost appears prominently in section 11, no section is empty

---

## Evaluation rubric (applies to all problems)

| Component | Weight | What is assessed |
|---|---|---|
| Prompt clarity | 25% | Is the task, input, output, and constraints unambiguous? |
| Prompt completeness | 25% | Does the prompt specify all edge cases, signatures, and constraints? |
| Code correctness | 30% | Does the code run, handle edge cases, and produce correct output? |
| Test quality | 20% | Do the tests actually verify the right behaviour, not just that the code runs? |

---

## Submission format

For each problem, submit:

```
problem_N/
├── prompt.txt        ← the prompt you wrote
├── solution.py       ← the code generated (or written) from your prompt
└── tests.py          ← the test suite
```

Run `python -m pytest problem_N/tests.py -v` before submitting. All tests must pass.
