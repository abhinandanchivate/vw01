# Case Study: Evaluating Prompt Quality for an AI-Powered Volkswagen Warranty Claim Review Assistant

## Background

Volkswagen receives thousands of warranty claims every month from dealerships across multiple regions.

Warranty specialists currently review:

* Repair orders
* Technician findings
* Vehicle history
* Warranty policies
* Parts replaced
* Customer complaints

The review process is time-consuming and can lead to inconsistent decisions when different specialists interpret information differently.

Volkswagen plans to deploy a Generative AI assistant to help warranty analysts prepare preliminary warranty review summaries.

Before deployment, the AI Governance Team wants to compare multiple prompt designs to determine which prompt provides the most reliable, explainable, and policy-compliant results.

---

# Business Problem

Warranty analysts spend significant effort determining whether a repair appears consistent with warranty policy guidelines.

The AI assistant should:

* Summarize warranty claims
* Identify relevant warranty policy references
* Highlight missing information
* Flag potential review risks
* Recommend further validation steps

The assistant must NOT:

* Approve claims
* Reject claims
* Make legal decisions
* Assume warranty coverage
* Assume customer misuse
* Override technician findings

---

# Sample Warranty Claim

## Vehicle Information

Vehicle Model: Volkswagen Tiguan

Model Year: 2023

VIN: WVWZZZ12345678901

Mileage: 46,850 km

Warranty Status: Active

Warranty Expiry: 15-Dec-2027

---

## Customer Complaint

Customer reports:

* Engine warning lamp illuminated
* Intermittent loss of power while accelerating
* Increased fuel consumption

---

## Dealer Repair Order

Repair Order Number: RO-784512

Date Opened: 18-Jun-2026

Dealer:

Volkswagen Downtown Service Center

---

## Technician Findings

Technician reported:

* Diagnostic scan performed
* Multiple engine control module fault codes detected
* Turbocharger boost pressure outside expected range
* Further root-cause investigation recommended

---

## Parts Requested for Replacement

* Turbocharger Assembly
* Pressure Sensor
* Intake Hose

Estimated Claim Amount:

₹1,82,500

---

## Warranty Policy Extract

Policy Section 4.2:

Turbocharger replacement may be eligible under warranty when failure results from manufacturing defects and required maintenance schedules have been followed.

Policy Section 4.3:

Claims involving aftermarket modifications require additional review.

Policy Section 5.1:

Warranty approval requires supporting diagnostic evidence.

---

## Service History

Service 1:

Completed on schedule

Service 2:

Completed on schedule

Service 3:

Delayed by 7 months

---

## Missing Information

* Photographs of failed components
* Detailed diagnostic logs
* Evidence of aftermarket modifications
* Engineering review notes

---

# Business Objective

Volkswagen wants to determine which prompt design generates the most reliable warranty review report while ensuring compliance with internal warranty policies.

---

# Existing Prompt Versions

Three departments have independently created prompts.

---

## Version A – Basic Prompt

Created by a dealership operations team.

Characteristics:

* Requests a warranty summary
* No role definition
* No policy guidance
* No compliance instructions

Potential Concern:

The LLM may assume warranty eligibility without evidence.

---

## Version B – Structured Prompt

Created by the warranty operations department.

Characteristics:

* Defines reviewer role
* Requires summary sections
* Requests observations

Potential Concern:

May still allow unsupported conclusions.

---

## Version C – Enterprise Prompt

Created by Volkswagen AI Governance Team.

Characteristics:

* Explicit role assignment
* Policy-based reasoning
* Missing-information handling
* Compliance constraints
* Required justification for every observation

Potential Concern:

Prompt complexity may increase maintenance overhead.

---

# Tasks

## Task 1 – Prompt Review

Review all prompt versions and identify:

* Missing instructions
* Ambiguous wording
* Compliance gaps
* Risk of unsupported decisions

---

## Task 2 – Warranty-Specific Evaluation

Evaluate each prompt using the following criteria.

### Warranty Policy Alignment

Does the prompt force use of policy information?

### Evidence-Based Reasoning

Does the prompt require supporting evidence?

### Missing Information Handling

Can the prompt identify documentation gaps?

### Hallucination Prevention

Can the prompt prevent unsupported warranty decisions?

### Explainability

Can reviewers understand how conclusions were reached?

### Consistency

Will multiple analysts receive similar outputs?

### Compliance Readiness

Does the prompt support internal governance requirements?

### Risk Identification

Can the prompt identify claim-review risks?

### Auditability

Can decisions be traced back to policy references?

### Production Readiness

Can the prompt safely support real dealership operations?

---

## Task 3 – Scoring Framework

Evaluate every prompt using a score from 1–10.

| Score | Meaning    |
| ----- | ---------- |
| 1–2   | Poor       |
| 3–4   | Weak       |
| 5–6   | Acceptable |
| 7–8   | Good       |
| 9–10  | Excellent  |

---

## Task 4 – Governance Review

For each prompt identify:

### Business Risks

Examples:

* Incorrect claim recommendations
* Inconsistent claim handling
* Missing policy references

### Operational Risks

Examples:

* Increased analyst review effort
* Escalation volume

### Compliance Risks

Examples:

* Policy violations
* Unsupported recommendations

---

## Task 5 – Comparison Report

Prepare:

### Executive Summary

### Detailed Scoring Matrix

### Strengths

### Weaknesses

### Risks

### Governance Assessment

for each prompt.

---

## Task 6 – Prompt Ranking

Determine:

### Best Prompt

### Second Best Prompt

### Third Best Prompt

Provide evidence-based justification.

---

## Task 7 – Enterprise Standard Design

Create a recommended prompt standard for all future Volkswagen warranty-review assistants.

The standard should require:

* Role definition
* Warranty policy references
* Evidence-based analysis
* Missing-information identification
* Compliance validation
* Structured output
* Confidence assessment
* Audit trail generation

---

# Expected Deliverables

## Deliverable 1 – Prompt Evaluation Matrix

| Criteria                 | Version A | Version B | Version C |
| ------------------------ | --------- | --------- | --------- |
| Policy Alignment         | ?         | ?         | ?         |
| Evidence-Based Reasoning | ?         | ?         | ?         |
| Hallucination Prevention | ?         | ?         | ?         |
| Auditability             | ?         | ?         | ?         |
| Compliance Readiness     | ?         | ?         | ?         |
| Production Readiness     | ?         | ?         | ?         |

---

## Deliverable 2 – Warranty Governance Risk Report

Include:

* Business impact
* Financial impact
* Compliance impact
* Customer impact
* Mitigation recommendations

---

## Deliverable 3 – Executive Recommendation

Answer:

* Which prompt should be deployed?
* Why?
* What controls are required before production release?
* What prompt-engineering standards should Volkswagen adopt globally?

---

# Success Criteria

The selected prompt should:

* Support evidence-based warranty reviews
* Reduce unsupported conclusions
* Improve consistency across dealerships
* Reference warranty policies correctly
* Identify missing documentation
* Improve auditability
* Support governance and compliance reviews
* Be suitable for enterprise-scale deployment

---

# Learning Outcomes

Participants will learn how prompt engineering affects:

* Warranty claim processing
* Policy compliance
* AI governance
* Auditability
* Explainability
* Operational consistency
* Enterprise deployment readiness

