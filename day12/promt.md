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

Write a complete, production-structured Python script that implements an
ML-assisted vehicle service-risk assessment system for Volkswagen India.

The system must:
- Accept structured vehicle telemetry and customer complaint data as input
- Validate every input field
- Engineer derived features from raw inputs
- Compute a heuristic risk probability and classify it as Low / Medium / High
- Explain the top risk contributors
- Assess safety and generate a driving recommendation
- Produce a customer-friendly service report in plain language
- Output a machine-readable JSON result

All outputs must be clearly labelled as heuristic estimates — not trained
model predictions. An authorized Volkswagen technician must validate all
findings before any workshop action is taken.

---

## Input data to use

```python
VehicleInput(
    model                        = "Volkswagen Taigun",
    model_year                   = 2022,
    mileage_km                   = 78_500,
    months_since_last_service    = 14,
    engine_warning_alerts        = 4,
    brake_wear_pct               = 81.0,
    battery_voltage_v            = 11.4,
    battery_measurement_context  = "unknown",   # "off" | "cranking" | "running" | "unknown"
    customer_complaints          = [
        "Vehicle takes longer to start",
        "Braking noise is heard",
        "Fuel efficiency has reduced",
    ],
    current_year                 = 2025,
)
```

---

## Architecture requirements

Structure the code in exactly these 12 modules, in this order:

1. Data models
   - VehicleInput dataclass (all raw input fields)
   - ValidationResult dataclass (feature, input_value, status, observation)
   - EngineeredFeatures dataclass (all 10 derived features)
   - RiskPrediction dataclass (probability, category, confidence, horizon,
     prediction_type)
   - RiskFactor dataclass (rank, feature, observed_value, impact, direction,
     explanation)

2. Constants and thresholds
   - RISK_THRESHOLDS = {"low": 0.40, "high": 0.70}
   - NORMAL_ANNUAL_MILEAGE_KM = 15_000
   - MAX_SERVICE_INTERVAL_MONTHS = 12
   - BRAKE_WARN_PCT = 70, BRAKE_CRITICAL_PCT = 80
   - BATTERY_LOW_V_OFF = 12.0, BATTERY_LOW_V_RUNNING = 13.5
   - SAFETY_COMPLAINT_KEYWORDS list
   - FEATURE_WEIGHTS dict (must sum to 1.0):
       service_overdue   : 0.15
       brake_wear        : 0.25
       engine_alerts     : 0.20
       battery_voltage   : 0.15
       mileage_rate      : 0.10
       safety_complaints : 0.10
       complaint_count   : 0.05

3. validate_inputs(v: VehicleInput) -> list[ValidationResult]
   - model_year: flag if in the future or age > 20 years
   - mileage_km: flag if negative or > 300,000 km
   - months_since_last_service: flag if negative or > MAX_SERVICE_INTERVAL_MONTHS
   - engine_warning_alerts: categorize 0 / 1-2 / 3+ as Valid / Warning / Critical
   - brake_wear_pct: must be 0–100; flag Warning at >= 70%, Critical at >= 80%
   - battery_voltage_v: interpret based on battery_measurement_context:
       "off"      -> Critical if < 12.0 V
       "running"  -> Critical if < 13.5 V
       "cranking" -> Warning always (load test required)
       "unknown"  -> Warning with a note that context is missing and no
                     confirmed battery diagnosis can be made
   - customer_complaints: Warning if empty list

4. engineer_features(v: VehicleInput) -> EngineeredFeatures
   Derive all 10 features:
   - vehicle_age_years = current_year - model_year
   - avg_annual_mileage_km = mileage_km / max(vehicle_age_years, 1)
   - service_overdue (bool) = months_since_last_service > MAX_SERVICE_INTERVAL_MONTHS
   - service_delay_severity: "Normal" / "Overdue" (1-3 months late) /
     "Significantly Overdue" (> 3 months late)
   - brake_wear_severity: "Normal" / "Elevated" (>= 70%) / "Critical" (>= 80%)
   - engine_alert_severity: "None" / "Low" (1) / "Moderate" (2-3) / "High" (4+)
   - low_battery_voltage_flag: Optional[bool] — None when context is "unknown"
   - complaint_count = len(customer_complaints)
   - safety_complaint_flag: True if any complaint contains a keyword from
     SAFETY_COMPLAINT_KEYWORDS
   - combined_risk_score: weighted sum using FEATURE_WEIGHTS, each sub-score
     normalized to 0.0–1.0 before weighting

5. predict_risk(ef: EngineeredFeatures) -> RiskPrediction
   - base probability = combined_risk_score
   - apply safety floor: if safety_complaint_flag is True, prob >= 0.55
   - apply brake floor: if brake_wear_severity == "Critical" AND
     safety_complaint_flag is True, prob >= 0.72
   - clamp to [0.0, 1.0]
   - classify using RISK_THRESHOLDS
   - confidence: "Medium" if low_battery_voltage_flag is None, else "High"
     if prob > 0.80 or prob < 0.20, else "Medium"
   - prediction_type must always be:
     "Heuristic estimate — not a trained production model"

6. get_top_risk_factors(v, ef) -> list[RiskFactor]
   Return exactly 5 ranked RiskFactor objects:
   1. Brake wear
   2. Engine warning alerts
   3. Months since last service
   4. Battery voltage
   5. Safety-related complaints
   Each must include: rank, feature, observed_value, impact, direction,
   and a 2-sentence plain-English explanation.

7. FEATURE_INTERACTIONS dict (module-level constant, not a function)
   Four keys with multi-sentence narrative strings:
   - "brake_wear_plus_braking_noise"
   - "low_battery_plus_slow_start"
   - "engine_warnings_plus_fuel_efficiency"
   - "high_mileage_plus_service_gap"

8. assess_safety(v, ef, pred) -> dict
   Return a dict with keys:
   - "safety_symptoms": list[str] — one entry per identified safety signal
   - "avoid_driving": bool
   - "driving_recommendation": str (full sentence recommendation)
   Set avoid_driving = True if:
   - any braking complaint is present, OR
   - brake_wear_severity == "Critical"
   Driving recommendation must differ clearly between avoid_driving True and False.

9. INSPECTION_CHECKLIST (module-level list of dicts)
   Exactly 14 items, each with keys "system" and "action".
   Systems: Brake pads, Brake discs, Brake fluid, Battery condition,
   Battery charge, Alternator & charging system, Diagnostic trouble codes,
   Engine warning alerts, Fuel system, Air intake, Ignition system,
   Emission system, Tyres, Scheduled maintenance.

10. generate_customer_report(v, ef, pred, safety) -> str
    Return a formatted multi-line string with 8 numbered sections:
    1. Risk category
    2. Estimated service-risk probability (as a percentage)
    3. Major observations (bullet list)
    4. Why this risk level was assigned (2–3 sentences)
    5. Recommended inspections (priority bullet list)
    6. Should the vehicle be driven? (full paragraph)
    7. Priority and suggested timeline
    8. Limitations (bullet list)
    Include a header and footer border line.
    Include this disclaimer in the header:
    "This is an ML-assisted estimate, not a confirmed diagnosis."

11. to_json_output(v, pred, risk_factors, safety, validation) -> dict
    Return a dict matching this exact schema — no extra keys, no missing keys:
    {
      "vehicle": {
        "model": str,
        "model_year": int,
        "mileage_km": int
      },
      "prediction": {
        "urgent_service_probability": float,
        "risk_category": str,
        "confidence_level": str,
        "prediction_horizon_days": int,
        "prediction_type": str
      },
      "top_risk_factors": [
        {
          "feature": str,
          "observed_value": str,
          "impact": str,
          "explanation": str
        }
      ],
      "safety_concern": bool,
      "driving_recommendation": str,
      "recommended_inspections": list[str],
      "inspection_priority": str,
      "suggested_timeline": str,
      "missing_information": list[str],
      "limitations": list[str]
    }

12. run_assessment(vehicle: VehicleInput) -> None
    Orchestrate all modules in order and print these 8 labelled sections:
    A — Input data validation table
        Columns: Feature | Status | Observation
    B — Engineered features table
        Columns: Feature | Value | Impact
    C — ML risk prediction block (5 bullet lines)
    D — Top risk factors (ranked, with interaction narratives below)
    E — Safety risk assessment (symptoms + driving recommendation)
    F — Inspection recommendations (bullet list)
    G — Customer service-risk report (full formatted report)
    H — Machine-readable JSON (json.dumps with indent=2)

---

## Code quality requirements

- Python 3.10+ only — use built-in type hints (list[str], dict, Optional)
- No external dependencies — use only the standard library
  (json, math, datetime, dataclasses, typing)
- Every function must have a docstring (one sentence is enough)
- All constants must be defined at module level, never inside functions
- No magic numbers inside functions — reference the constants
- Use dataclasses for all data models — no plain dicts as return types
  except where explicitly specified above
- The entry point must be:
    if __name__ == "__main__":
        taigun = VehicleInput(...)
        run_assessment(taigun)

---

## Output format requirements

- Output the complete Python script only
- No explanation text before or after the code block
- No placeholder comments like "# add logic here"
- Every function must be fully implemented — no stubs, no pass statements
- The script must run without errors from:
    python vw_predictive_maintenance.py

---

## Mandatory constraints — enforce throughout

- Never claim any component has failed — only flag risk or need for inspection
- Never invent diagnostic trouble codes, model accuracy metrics, SHAP values,
  or calibration results
- Never present the heuristic probability as a trained model output
- Battery voltage must never be diagnosed without knowing the measurement
  context — when context is "unknown", always flag it as missing information
- Every output section must include a statement that an authorized Volkswagen
  technician must validate all findings before any action is taken
- Do not recommend replacing any component unless testing confirms it
