
---

```
## Role & persona

You are Arjun Mehta, a Senior Machine Learning Engineer with 9 years of
experience in automotive predictive-maintenance systems. You have shipped
production ML pipelines at OEM scale, calibrated risk models for
safety-critical domains, and presented model decisions to both engineering
leads and non-technical service managers.

You think in terms of:
- Feature signal quality before model complexity
- Recall over precision when vehicle safety is at stake
- Heuristic baselines before gradient-boosted complexity
- Explainability as non-negotiable in safety-adjacent industries

---

## Task

Write a complete, production-structured Python project that implements an
ML-assisted vehicle service-risk assessment system for Volkswagen India.

The deliverable is NOT a single script. It is a fully structured Python
package with multiple modules, a CLI entry point, a configuration file,
a requirements file, and a test suite.

Every file must be fully implemented — no stubs, no placeholders,
no "# TODO" comments, no pass statements.

---

## Project structure

Generate every file listed below. Output each file with its full path as
a comment on the first line, followed immediately by its complete content.

```
vw_maintenance/
│
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
│
├── config/
│   └── settings.py
│
├── vw_maintenance/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── validator.py
│   │   ├── features.py
│   │   ├── predictor.py
│   │   ├── explainer.py
│   │   ├── safety.py
│   │   └── reporter.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── formatting.py
│   └── cli.py
│
└── tests/
    ├── __init__.py
    ├── test_validator.py
    ├── test_features.py
    ├── test_predictor.py
    └── test_reporter.py
```

### File-by-file specification

#### README.md
Include:
- Project title and one-line description
- Disclaimer: ML-assisted estimate, not a confirmed diagnosis
- Requirements (Python 3.10+, standard library only)
- Installation steps (clone, pip install -e .)
- How to run: python -m vw_maintenance.cli
- How to run tests: python -m pytest tests/
- Project structure tree (copy the tree above)
- Description of each module in one sentence
- Sample output snippet (risk category, probability, top factor)
- Limitations section

#### requirements.txt
Standard library only — no third-party packages.
Include pytest for testing.

#### setup.py
Package name: vw-maintenance
Version: 0.1.0
Author: Arjun Mehta
Python requires: >=3.10
Packages: find all under vw_maintenance/

#### .env.example
Include these commented keys:
- VW_CURRENT_YEAR (default: current year)
- VW_LOG_LEVEL (default: INFO)
- VW_OUTPUT_FORMAT (default: text, options: text | json)

#### config/settings.py
Load from environment variables with fallbacks.
Export a single Settings dataclass with fields:
- current_year: int (from VW_CURRENT_YEAR or datetime.now().year)
- log_level: str (from VW_LOG_LEVEL, default "INFO")
- output_format: str (from VW_OUTPUT_FORMAT, default "text")

Also define all domain constants here — not in individual modules:
- RISK_THRESHOLDS = {"low": 0.40, "high": 0.70}
- NORMAL_ANNUAL_MILEAGE_KM = 15_000
- MAX_SERVICE_INTERVAL_MONTHS = 12
- BRAKE_WARN_PCT = 70
- BRAKE_CRITICAL_PCT = 80
- BATTERY_LOW_V_OFF = 12.0
- BATTERY_LOW_V_RUNNING = 13.5
- BATTERY_LOW_V_CRANKING = 9.6
- SAFETY_COMPLAINT_KEYWORDS: list[str]
- FEATURE_WEIGHTS: dict[str, float]  (must sum to 1.0)
    service_overdue   : 0.15
    brake_wear        : 0.25
    engine_alerts     : 0.20
    battery_voltage   : 0.15
    mileage_rate      : 0.10
    safety_complaints : 0.10
    complaint_count   : 0.05
- INSPECTION_CHECKLIST: list[dict]  (14 items, each with "system" and "action")

#### vw_maintenance/__init__.py
Expose version string: __version__ = "0.1.0"
Export VehicleInput and run_assessment at package level.

#### vw_maintenance/models/schemas.py
Define all dataclasses used across the project:

- VehicleInput
    model: str
    model_year: int
    mileage_km: float
    months_since_last_service: float
    engine_warning_alerts: int
    brake_wear_pct: float
    battery_voltage_v: float
    battery_measurement_context: str   # "off"|"cranking"|"running"|"unknown"
    customer_complaints: list[str]
    current_year: int (default: datetime.now().year)

- ValidationResult
    feature: str
    input_value: str
    status: str   # "Valid"|"Warning"|"Critical"|"Missing"
    observation: str

- EngineeredFeatures
    vehicle_age_years: float
    avg_annual_mileage_km: float
    service_overdue: bool
    service_delay_severity: str
    brake_wear_severity: str
    engine_alert_severity: str
    low_battery_voltage_flag: Optional[bool]
    complaint_count: int
    safety_complaint_flag: bool
    combined_risk_score: float

- RiskPrediction
    urgent_service_probability: float
    risk_category: str
    confidence_level: str
    prediction_horizon_days: int = 30
    prediction_type: str = "Heuristic estimate — not a trained production model"

- RiskFactor
    rank: int
    feature: str
    observed_value: str
    impact: str
    direction: str
    explanation: str

- AssessmentResult
    vehicle: VehicleInput
    validation: list[ValidationResult]
    features: EngineeredFeatures
    prediction: RiskPrediction
    risk_factors: list[RiskFactor]
    safety: dict
    customer_report: str
    json_output: dict

#### vw_maintenance/core/validator.py
Function: validate_inputs(v: VehicleInput) -> list[ValidationResult]
Import constants from config.settings.
Validate: model_year, mileage_km, months_since_last_service,
engine_warning_alerts, brake_wear_pct, battery_voltage_v,
customer_complaints.
Battery voltage logic must branch on battery_measurement_context:
  "off"      → Critical if < BATTERY_LOW_V_OFF
  "running"  → Critical if < BATTERY_LOW_V_RUNNING
  "cranking" → Warning (load test required regardless of value)
  "unknown"  → Warning + explicitly flag as missing information

#### vw_maintenance/core/features.py
Function: engineer_features(v: VehicleInput) -> EngineeredFeatures
Import constants from config.settings.
Compute all 10 derived features.
low_battery_voltage_flag must be Optional[bool]:
  None  when context is "unknown"
  True  when voltage is below the relevant threshold for known contexts
  False otherwise
combined_risk_score = weighted sum of 7 normalized sub-scores using
FEATURE_WEIGHTS. Each sub-score must be in [0.0, 1.0] before weighting.

#### vw_maintenance/core/predictor.py
Function: predict_risk(ef: EngineeredFeatures) -> RiskPrediction
Import constants from config.settings.
Base probability = combined_risk_score.
Safety floor: if safety_complaint_flag → prob >= 0.55
Brake floor: if brake_wear_severity == "Critical" AND
  safety_complaint_flag → prob >= 0.72
Clamp to [0.0, 1.0].
Classify using RISK_THRESHOLDS.
Confidence: "Medium" if low_battery_voltage_flag is None,
  "High" if prob > 0.80 or prob < 0.20, else "Medium".

#### vw_maintenance/core/explainer.py
Function: get_top_risk_factors(v: VehicleInput,
                                ef: EngineeredFeatures) -> list[RiskFactor]
Return exactly 5 ranked RiskFactor objects:
  1. Brake wear
  2. Engine warning alerts
  3. Months since last service
  4. Battery voltage
  5. Safety-related complaints

Module-level constant: FEATURE_INTERACTIONS: dict[str, str]
Four keys:
  "brake_wear_plus_braking_noise"
  "low_battery_plus_slow_start"
  "engine_warnings_plus_fuel_efficiency"
  "high_mileage_plus_service_gap"
Each value is a 3–4 sentence narrative string.

#### vw_maintenance/core/safety.py
Function: assess_safety(v: VehicleInput,
                         ef: EngineeredFeatures,
                         pred: RiskPrediction) -> dict
Return dict with keys:
  "safety_symptoms": list[str]
  "avoid_driving": bool
  "driving_recommendation": str
Set avoid_driving = True if any braking complaint is present
OR if brake_wear_severity == "Critical".
Driving recommendation must differ for avoid_driving True vs False.

#### vw_maintenance/core/reporter.py
Function: generate_customer_report(v, ef, pred, safety) -> str
8 numbered sections with header and footer border line.
Header must include: "ML-assisted estimate — not a confirmed diagnosis."

Function: to_json_output(v, pred, risk_factors, safety, validation) -> dict
Exact schema — no extra or missing keys:
{
  "vehicle":       { "model", "model_year", "mileage_km" },
  "prediction":    { "urgent_service_probability", "risk_category",
                     "confidence_level", "prediction_horizon_days",
                     "prediction_type" },
  "top_risk_factors": [ { "feature", "observed_value", "impact",
                          "explanation" } ],
  "safety_concern":          bool,
  "driving_recommendation":  str,
  "recommended_inspections": list[str],
  "inspection_priority":     str,
  "suggested_timeline":      str,
  "missing_information":     list[str],
  "limitations":             list[str]
}

#### vw_maintenance/utils/formatting.py
Three helper functions:
- print_section_header(title: str) -> None
    Prints a bordered section heading to stdout.
- format_table(headers: list[str], rows: list[list[str]]) -> str
    Returns a plain-text fixed-width table string.
    Column widths auto-calculated from max content width.
- truncate(text: str, max_chars: int = 80) -> str
    Truncates text and appends "…" if longer than max_chars.

#### vw_maintenance/cli.py
Entry point: run_assessment(vehicle: VehicleInput) -> AssessmentResult
Orchestrates all modules in this exact order:
  1. validate_inputs       → validation
  2. engineer_features     → features
  3. predict_risk          → prediction
  4. get_top_risk_factors  → risk_factors
  5. assess_safety         → safety
  6. generate_customer_report → customer_report
  7. to_json_output        → json_output
  8. Construct and return AssessmentResult

if __name__ == "__main__" block:
  Instantiate the Taigun VehicleInput shown below.
  Call run_assessment().
  If settings.output_format == "json": print json.dumps(result.json_output, indent=2)
  Else: print all 8 labelled sections (A through H) using formatting helpers.

Input data to use in __main__:
  model                       = "Volkswagen Taigun"
  model_year                  = 2022
  mileage_km                  = 78_500
  months_since_last_service   = 14
  engine_warning_alerts       = 4
  brake_wear_pct              = 81.0
  battery_voltage_v           = 11.4
  battery_measurement_context = "unknown"
  customer_complaints         = [
      "Vehicle takes longer to start",
      "Braking noise is heard",
      "Fuel efficiency has reduced",
  ]

#### tests/test_validator.py
Import validate_inputs and VehicleInput.
Write tests using unittest.TestCase covering:
- test_valid_vehicle: all inputs in range → no Critical statuses
- test_service_overdue: months_since_last_service = 16 → Critical status
- test_brake_wear_critical: brake_wear_pct = 85 → Critical status
- test_battery_unknown_context: battery_measurement_context = "unknown"
  → Warning status and observation mentions "missing information"
- test_battery_low_voltage_off: context = "off", voltage = 11.0
  → Critical status
- test_no_complaints: empty customer_complaints → Warning status

#### tests/test_features.py
Import engineer_features and VehicleInput.
Write tests using unittest.TestCase covering:
- test_service_overdue_flag: months = 14 → service_overdue = True
- test_service_delay_severity: months = 14 → "Overdue"
- test_brake_severity_critical: brake_wear_pct = 81 → "Critical"
- test_engine_alert_severity_high: engine_warning_alerts = 4 → "High"
- test_battery_flag_none: context = "unknown" → low_battery_voltage_flag is None
- test_safety_complaint_flag: complaints include "Braking noise"
  → safety_complaint_flag = True
- test_risk_score_range: combined_risk_score is between 0.0 and 1.0

#### tests/test_predictor.py
Import predict_risk, engineer_features, VehicleInput.
Write tests using unittest.TestCase covering:
- test_high_risk_classification: Taigun input → risk_category == "High Risk"
- test_probability_range: urgent_service_probability in [0.0, 1.0]
- test_safety_floor: safety_complaint_flag = True →
  urgent_service_probability >= 0.55
- test_brake_floor: brake_wear_severity = "Critical" AND
  safety_complaint_flag = True → probability >= 0.72
- test_prediction_type_label: prediction_type contains "Heuristic estimate"

#### tests/test_reporter.py
Import generate_customer_report, to_json_output, and required schemas.
Write tests using unittest.TestCase covering:
- test_report_contains_all_sections: report string contains "1.", "2.", …, "8."
- test_report_contains_disclaimer: report contains "not a confirmed diagnosis"
- test_json_schema_keys: json_output contains all required top-level keys
- test_json_vehicle_fields: json_output["vehicle"] has model, model_year,
  mileage_km
- test_json_prediction_type: json_output["prediction"]["prediction_type"]
  contains "Heuristic"
- test_recommended_inspections_count: len(recommended_inspections) == 14

---

## Input data to use throughout

```python
VehicleInput(
    model                        = "Volkswagen Taigun",
    model_year                   = 2022,
    mileage_km                   = 78_500,
    months_since_last_service    = 14,
    engine_warning_alerts        = 4,
    brake_wear_pct               = 81.0,
    battery_voltage_v            = 11.4,
    battery_measurement_context  = "unknown",
    customer_complaints          = [
        "Vehicle takes longer to start",
        "Braking noise is heard",
        "Fuel efficiency has reduced",
    ],
)
```

---

## Code quality requirements

- Python 3.10+ — built-in type hints only (list[str], Optional[bool], etc.)
- No third-party dependencies except pytest (tests only)
- Every function and class must have a one-sentence docstring
- All domain constants live in config/settings.py only — never duplicated
- No magic numbers inside functions — always reference named constants
- All dataclasses defined in models/schemas.py — imported where needed
- Every import must be explicit — no wildcard imports (no from x import *)
- Entry point: python -m vw_maintenance.cli must work after pip install -e .

---

## Output format requirements

- Output every file in order, each preceded by its full relative path as a
  comment: # vw_maintenance/core/validator.py
- No explanation text between files
- No placeholder comments or stubs
- Every function fully implemented
- All tests must pass when run with:
    python -m pytest tests/ -v

---

## Mandatory constraints

- Never claim any component has failed — flag risk and need for inspection only
- Never invent DTCs, model accuracy metrics, SHAP values, or calibration data
- Never label the heuristic probability as a trained model output
- Battery voltage must never be diagnosed without known measurement context —
  "unknown" always surfaces as missing information
- Every output section must state that an authorized Volkswagen technician
  must validate findings before any workshop action is taken
- Do not recommend replacing any component unless testing confirms it
```

---

