# Case Study: Selecting the Best Prompt for an AI-Powered Insurance Claim Assistant

## Background

A large insurance company plans to deploy a Generative AI assistant to help claims officers review motor insurance claims.

The assistant will analyze claim submissions and generate a preliminary claim assessment report for internal use.

The company has discovered that different teams are writing prompts differently, resulting in inconsistent outputs from the LLM.

Before deploying the AI solution, management wants to determine which prompt design approach produces the most reliable, explainable, and compliant results.

---

# Business Problem

Claims officers currently spend significant time reviewing claim forms, accident descriptions, repair estimates, and supporting documents.

The company wants the AI assistant to:

* Summarize claim information
* Identify missing information
* Highlight risk indicators
* Recommend next review actions

However, the assistant must:

* Not approve or reject claims automatically
* Not make legal decisions
* Not assume fraud without evidence
* Clearly distinguish facts from assumptions

Management wants to compare multiple prompt designs and identify the most production-ready approach.

---

# Sample Claim Information

### Policy Information

Policy Number: INS-2026-45891

Policy Type: Comprehensive Motor Insurance

Policy Start Date: 15-Jan-2026

Policy Status: Active

---

### Claim Information

Claim Number: CLM-2026-9032

Claim Date: 18-Jun-2026

Claim Amount Requested: ₹1,85,000

---

### Customer Statement

> While driving during heavy rain, the vehicle skidded and collided with a roadside barrier. The front bumper, left headlamp, and bonnet were damaged.

---

### Supporting Documents Submitted

* Claim form
* Vehicle photographs
* Repair estimate from authorized garage
* Driving license copy

---

### Missing Documents

* Police report
* Weather report
* Third-party witness statement

---

# Business Objective

Determine which prompt produces the best claim assessment report while maintaining compliance, consistency, and explainability.

---

# Existing Prompt Versions

Three departments have created different prompt versions.

---

## Version A – Minimal Prompt

Developed by a junior analyst.

Characteristics:

* Simple request
* No role definition
* No compliance instructions
* No output structure

Potential Concern:

The model may generate inconsistent assessments and unsupported conclusions.

---

## Version B – Structured Prompt

Developed by claims operations.

Characteristics:

* Role assignment
* Basic output format
* General guidance

Potential Concern:

May still allow subjective assumptions.

---

## Version C – Enterprise Prompt

Developed by AI Governance Team.

Characteristics:

* Explicit role definition
* Compliance requirements
* Required output sections
* Hallucination controls
* Risk-identification guidelines

Potential Concern:

Higher complexity and maintenance effort.

---

# Tasks

## Task 1 – Prompt Review

Review all three prompt versions.

Identify:

* Missing instructions
* Ambiguous language
* Compliance concerns
* Potential risks

---

## Task 2 – Prompt Evaluation

Evaluate each prompt using the following criteria.

### Clarity

Does the prompt clearly define the task?

### Context Quality

Does the prompt provide sufficient business context?

### Instruction Quality

Are expectations explicit?

### Output Consistency

Will outputs remain consistent across runs?

### Compliance Readiness

Does the prompt support insurance regulations and internal policies?

### Hallucination Prevention

Can the prompt reduce unsupported assumptions?

### Explainability

Will the output explain how conclusions were reached?

### Risk Awareness

Does the prompt identify missing evidence and uncertainties?

### Maintainability

Can future business rules be added easily?

### Production Readiness

Can the prompt be deployed safely in production?

---

## Task 3 – Scoring Framework

Use a scale of 1–10 for each criterion.

| Score | Meaning    |
| ----- | ---------- |
| 1–2   | Poor       |
| 3–4   | Weak       |
| 5–6   | Acceptable |
| 7–8   | Good       |
| 9–10  | Excellent  |

---

## Task 4 – Comparison Report

Prepare a report containing:

### Executive Summary

### Scoring Matrix

### Strengths

### Weaknesses

### Risks

### Overall Assessment

for each prompt version.

---

## Task 5 – Prompt Ranking

Rank:

1. Best Prompt
2. Second Best Prompt
3. Third Best Prompt

Provide justification based on measurable criteria.

---

## Task 6 – Improved Prompt Standard

Design a standard prompt template for all future insurance claim assistants.

The standard must include:

* Role definition
* Business context
* Compliance instructions
* Hallucination controls
* Output structure
* Validation requirements

---

# Expected Deliverables

## Deliverable 1 – Prompt Evaluation Matrix

Example:

| Criteria                 | Version A | Version B | Version C |
| ------------------------ | --------- | --------- | --------- |
| Clarity                  | ?         | ?         | ?         |
| Compliance               | ?         | ?         | ?         |
| Explainability           | ?         | ?         | ?         |
| Hallucination Prevention | ?         | ?         | ?         |
| Production Readiness     | ?         | ?         | ?         |

---

## Deliverable 2 – Risk Assessment Report

Identify:

* Risks introduced by each prompt
* Business impact
* Compliance impact
* Mitigation recommendations

---

## Deliverable 3 – Governance Recommendation

Answer:

* Which prompt should be deployed?
* Which prompt should be retired?
* What changes are required before production deployment?

---

# Success Criteria

The selected prompt should:

* Produce consistent outputs
* Avoid unsupported assumptions
* Highlight missing evidence
* Improve explainability
* Support regulatory compliance
* Reduce operational risk
* Be suitable for enterprise deployment

---

# Learning Outcomes

Participants will learn:

1. Why prompt quality affects business outcomes.
2. How prompt design impacts compliance and governance.
3. How to measure prompt effectiveness objectively.
4. How to compare prompts using scoring frameworks.
5. How to reduce hallucinations through structured instructions.
6. How enterprises establish prompt standards before production deployment.
7. How to evaluate AI systems beyond just "good" or "bad" answers.
