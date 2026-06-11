"""
LAB 1 - SUPERVISED LEARNING: Telecom Churn Classification
===========================================================
ML Masterclass · Module 2 & 3 companion lab

Goal: build, evaluate and interrogate a churn classifier the way you would
in production review — pipeline-enforced preprocessing (no leakage),
honest splits, metrics matched to the cost structure, and feature
importance for the explainability conversation.

Run:  python lab1_supervised_churn.py
Deps: numpy, pandas, scikit-learn  (matplotlib optional)
Time: < 60 seconds. All data is synthetic and generated locally.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, precision_recall_curve)

RNG = np.random.default_rng(42)

# ----------------------------------------------------------------------
# SECTION 1 — Synthetic data with a planted causal story
# ----------------------------------------------------------------------
# Churn is driven (noisily) by: month-to-month contracts, short tenure,
# network problems, support tickets, and price pressure. The model's job
# is to recover this structure from 12,000 subscribers.
# ----------------------------------------------------------------------

def make_subscribers(n=12_000):
    tenure = RNG.gamma(shape=2.0, scale=14, size=n).clip(1, 72)          # months
    contract = RNG.choice(["month_to_month", "one_year", "two_year"],
                          size=n, p=[0.55, 0.25, 0.20])
    monthly_charge = RNG.normal(680, 180, n).clip(199, 1999)             # INR
    data_gb = RNG.gamma(2.5, 8, n).clip(0.5, 200)
    drops_90d = RNG.poisson(2.0, n)                                      # call drops
    outage_min_90d = RNG.exponential(30, n).clip(0, 600)
    tickets_90d = RNG.poisson(0.8, n)
    payment = RNG.choice(["auto_debit", "card", "cash"], n, p=[0.4, 0.35, 0.25])
    usage_delta = RNG.normal(0, 0.25, n)             # 90d usage trend (− = declining)

    # latent churn propensity (the "truth" the model must approximate)
    z = (
        -1.8
        + 1.1 * (contract == "month_to_month")
        - 0.030 * tenure
        + 0.0012 * (monthly_charge - 680)
        + 0.18 * drops_90d
        + 0.004 * outage_min_90d
        + 0.35 * tickets_90d
        - 1.4 * usage_delta                 # declining usage -> higher churn
        + 0.30 * (payment == "cash")
        + RNG.normal(0, 0.35, n)            # irreducible noise
    )
    churn = (1 / (1 + np.exp(-z)) > RNG.uniform(0, 1, n)).astype(int)

    return pd.DataFrame({
        "tenure_months": tenure.round(1),
        "contract": contract,
        "monthly_charge": monthly_charge.round(0),
        "data_gb_month": data_gb.round(1),
        "call_drops_90d": drops_90d,
        "outage_minutes_90d": outage_min_90d.round(0),
        "support_tickets_90d": tickets_90d,
        "payment_method": payment,
        "usage_trend_90d": usage_delta.round(3),
        "churned_30d": churn,
    })

df = make_subscribers()
print("=" * 68)
print("SECTION 1 · Data")
print("=" * 68)
print(df.head(5).to_string(index=False))
print(f"\nRows: {len(df):,} | churn rate: {df.churned_30d.mean():.1%}")

# ----------------------------------------------------------------------
# SECTION 2 — Split FIRST, transform later (leakage discipline)
# ----------------------------------------------------------------------
X = df.drop(columns="churned_30d")
y = df["churned_30d"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
# NOTE: in production this would be a TIME-based split — train on months
# 1..N, test on month N+1 — because random splits leak the future.

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = ["contract", "payment_method"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# ----------------------------------------------------------------------
# SECTION 3 — Baseline first, then the contender
# ----------------------------------------------------------------------
baseline = Pipeline([("prep", preprocess),
                     ("clf", LogisticRegression(max_iter=1000))])
model = Pipeline([("prep", preprocess),
                  ("clf", GradientBoostingClassifier(
                      n_estimators=300, learning_rate=0.05,
                      max_depth=3, subsample=0.8, random_state=42))])

print("\n" + "=" * 68)
print("SECTION 3 · Cross-validated comparison (5-fold ROC-AUC on train)")
print("=" * 68)
for name, est in [("Logistic regression (baseline)", baseline),
                  ("Gradient boosting (contender)", model)]:
    scores = cross_val_score(est, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"{name:35s} AUC = {scores.mean():.3f} ± {scores.std():.3f}")

# ----------------------------------------------------------------------
# SECTION 4 — Fit, then evaluate ONCE on the held-out test set
# ----------------------------------------------------------------------
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
pred_default = (proba >= 0.5).astype(int)

print("\n" + "=" * 68)
print("SECTION 4 · Held-out test performance")
print("=" * 68)
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
print("\nClassification report @ threshold 0.50:")
print(classification_report(y_test, pred_default, digits=3))
print("Confusion matrix [ [TN FP] [FN TP] ]:")
print(confusion_matrix(y_test, pred_default))

# ----------------------------------------------------------------------
# SECTION 5 — The threshold is a BUSINESS decision
# ----------------------------------------------------------------------
# Cost model (per the case study): a missed churner forfeits ~Rs 8,160 of
# CLV; an unnecessary retention discount costs ~Rs 900. Sweep thresholds
# and pick the one minimizing expected cost — not the default 0.5.
COST_FN, COST_FP = 8160, 900
prec, rec, thr = precision_recall_curve(y_test, proba)

best_t, best_cost = 0.5, float("inf")
for t in np.arange(0.05, 0.95, 0.01):
    p = (proba >= t).astype(int)
    fn = ((p == 0) & (y_test == 1)).sum()
    fp = ((p == 1) & (y_test == 0)).sum()
    cost = fn * COST_FN + fp * COST_FP
    if cost < best_cost:
        best_t, best_cost = t, cost

print("=" * 68)
print("SECTION 5 · Cost-optimal threshold")
print("=" * 68)
print(f"Missed-churner cost Rs {COST_FN:,} | wasted-offer cost Rs {COST_FP:,}")
print(f"Cost-optimal threshold: {best_t:.2f}  (vs default 0.50)")
print(f"Expected cost at optimum: Rs {best_cost:,.0f} on the test cohort")
print("Lesson: with asymmetric costs the optimal threshold is far from 0.5 —")
print("here we intervene on many more customers because misses are 9x dearer.")

# ----------------------------------------------------------------------
# SECTION 6 — What is the model looking at? (explainability)
# ----------------------------------------------------------------------
gb = model.named_steps["clf"]
feat_names = (num_cols +
              model.named_steps["prep"].named_transformers_["cat"]
                   .get_feature_names_out(cat_cols).tolist())
imp = pd.Series(gb.feature_importances_, index=feat_names).sort_values(ascending=False)

print("\n" + "=" * 68)
print("SECTION 6 · Feature importance (impurity-based)")
print("=" * 68)
print(imp.head(8).round(3).to_string())
print("\nCaution: importances describe THIS model, not causality. For")
print("per-customer reason codes in production, use SHAP values instead.")

# ----------------------------------------------------------------------
# DISCUSSION (for the session)
# ----------------------------------------------------------------------
# 1. Which feature here would become a LEAK if computed at the wrong time?
#    (usage_trend_90d is fine at prediction time; 'retention_offer_made'
#     or 'disconnection_ticket_open' would not be.)
# 2. Re-run with test_size=0.5 — what happens to the AUC's stability?
# 3. Change COST_FP to 3000 (expensive offers). Where does the threshold go?
