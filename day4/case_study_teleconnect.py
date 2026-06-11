"""
CASE STUDY - TeleConnect: End-to-End Churn Solution
====================================================
ML Masterclass · Capstone (16:45-17:45)

This script operationalizes the full case: it simulates the subscriber
base, trains the churn model with a TIME-BASED split, calibrates
probabilities, applies the ECONOMIC decision layer (risk x CLV x offer
cost), and produces the agent action queue with reason codes — i.e.,
Stream 1 of the solution architecture, plus the hand-off point where
Stream 2 (the RAG copilot, Lab 4) takes over.

Run:  python case_study_teleconnect.py
Deps: numpy, pandas, scikit-learn.  Time: < 60 seconds. Synthetic data.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(2026)

# ======================================================================
# STEP 1 · Simulate 18 months of subscriber snapshots
# ======================================================================
# Unit of prediction: subscriber-month. Prediction moment: month start.
# Target: churned within the FOLLOWING 30 days. This framing decision
# is half the project — get it wrong and no algorithm saves you.
# ======================================================================

def simulate_month(n, month):
    tenure = RNG.gamma(2.0, 14, n).clip(1, 84)
    contract = RNG.choice(["m2m", "1yr", "2yr"], n, p=[0.55, 0.25, 0.20])
    charge = RNG.normal(680, 180, n).clip(199, 1999)
    drops = RNG.poisson(2.0, n)
    outage = RNG.exponential(30, n).clip(0, 600)
    tickets = RNG.poisson(0.8, n)
    usage_delta = RNG.normal(0, 0.25, n)
    competitor_intensity = 0.15 * np.sin(month / 3) + 0.1  # market seasonality

    z = (-1.8 + 1.1 * (contract == "m2m") - 0.03 * tenure
         + 0.0012 * (charge - 680) + 0.18 * drops + 0.004 * outage
         + 0.35 * tickets - 1.4 * usage_delta + competitor_intensity
         + RNG.normal(0, 0.35, n))
    churn = (1 / (1 + np.exp(-z)) > RNG.uniform(0, 1, n)).astype(int)

    return pd.DataFrame({
        "month": month, "tenure": tenure.round(1), "contract": contract,
        "monthly_charge": charge.round(0), "call_drops_90d": drops,
        "outage_min_90d": outage.round(0), "tickets_90d": tickets,
        "usage_trend_90d": usage_delta.round(3), "churned_next_30d": churn,
    })

frames = [simulate_month(4000, m) for m in range(1, 19)]
df = pd.concat(frames, ignore_index=True)
print("=" * 70)
print(f"STEP 1 · {len(df):,} subscriber-month rows over 18 months "
      f"(churn rate {df.churned_next_30d.mean():.1%})")
print("=" * 70)

# ======================================================================
# STEP 2 · TIME-BASED split (months 1-14 train, 15-18 test)
# ======================================================================
# A random split would mix the future into training — the leakage trap
# from Module 3. The time split mirrors how the model will actually live.
# ======================================================================
train = df[df.month <= 14]
test = df[df.month > 14]
features = ["tenure", "contract", "monthly_charge", "call_drops_90d",
            "outage_min_90d", "tickets_90d", "usage_trend_90d"]
target = "churned_next_30d"

prep = ColumnTransformer([
    ("num", StandardScaler(), [f for f in features if f != "contract"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["contract"]),
])

# ======================================================================
# STEP 3 · Train + CALIBRATE
# ======================================================================
# The decision layer multiplies probability by money, so "0.30" must
# truly mean 30%. Gradient boosting scores rank well but need
# calibration before you can do economics with them.
# ======================================================================
base = Pipeline([("prep", prep),
                 ("gb", GradientBoostingClassifier(
                     n_estimators=300, learning_rate=0.05, max_depth=3,
                     subsample=0.8, random_state=0))])
model = CalibratedClassifierCV(base, method="isotonic", cv=3)
model.fit(train[features], train[target])

proba = model.predict_proba(test[features])[:, 1]
auc = roc_auc_score(test[target], proba)
print(f"\nSTEP 3 · Time-split ROC-AUC: {auc:.3f} (calibrated probabilities)")

# ======================================================================
# STEP 4 · The decision layer: economics, not thresholds-by-habit
# ======================================================================
# Expected value of intervening on customer i:
#   EV_save = p_churn_i * SAVE_RATE * CLV_i  -  OFFER_COST_i
# Intervene when EV_save > 0, ranked by EV, capped by agent capacity.
# ======================================================================
SAVE_RATE = 0.35                    # pilot: offers save ~35% of true churners
AGENT_CAPACITY = 600                # calls/day the 400 agents can absorb

snap = test[test.month == test.month.max()].copy()
snap["p_churn"] = model.predict_proba(snap[features])[:, 1]
snap["clv"] = snap["monthly_charge"] * 12                      # 12-month CLV
snap["offer_cost"] = np.where(snap.p_churn > 0.5,
                              snap.monthly_charge * 0.20 * 6,  # 20% x 6 months
                              snap.monthly_charge * 0.10 * 6)  # 10% x 6 months
snap["ev_save"] = snap.p_churn * SAVE_RATE * snap.clv - snap.offer_cost

queue = (snap[snap.ev_save > 0]
         .sort_values("ev_save", ascending=False)
         .head(AGENT_CAPACITY))

print("\n" + "=" * 70)
print("STEP 4 · Decision layer output")
print("=" * 70)
print(f"Subscribers in month snapshot:        {len(snap):>7,}")
print(f"Positive-EV interventions available:  {(snap.ev_save > 0).sum():>7,}")
print(f"Queued for agents (capacity cap):     {len(queue):>7,}")
print(f"Expected net value of today's queue:  Rs {queue.ev_save.sum():>10,.0f}")

# Contrast with the old "blind offers" world:
blind = snap.sample(AGENT_CAPACITY, random_state=1)
blind_ev = (blind.p_churn * SAVE_RATE * blind.clv - blind.offer_cost).sum()
print(f"Same capacity, blind targeting:       Rs {blind_ev:>10,.0f}")
ratio = queue.ev_save.sum() / max(blind_ev, 1)
print(f"Model-targeted vs blind value ratio:  {ratio:>9.1f}x")

# ======================================================================
# STEP 5 · Reason codes: the explainability hand-off to agents
# ======================================================================
# Production would compute SHAP values per customer; we approximate with
# transparent rule-based reasons drawn from the same features, which is
# also a sound fallback pattern when SHAP latency is a constraint.
# ======================================================================
def reasons(row):
    out = []
    if row.contract == "m2m":        out.append("month-to-month contract")
    if row.usage_trend_90d < -0.15:  out.append("declining usage")
    if row.tickets_90d >= 2:         out.append("repeated support tickets")
    if row.call_drops_90d >= 4:      out.append("network quality issues")
    if row.monthly_charge > 900:     out.append("price-sensitive high bill")
    return "; ".join(out[:3]) or "composite risk factors"

queue["reason_codes"] = queue.apply(reasons, axis=1)

print("\n" + "=" * 70)
print("STEP 5 · Agent queue sample (top 8) — Stream 2 copilot picks up here")
print("=" * 70)
view = queue[["p_churn", "clv", "offer_cost", "ev_save", "reason_codes"]].head(8)
print(view.round({"p_churn": 2, "clv": 0, "offer_cost": 0, "ev_save": 0})
      .to_string(index=False))

print("""
HAND-OFF: each queued customer + reason codes goes to the retention
copilot (Lab 4), which retrieves the matching playbook section and
current offer policy, drafts talking points with citations, and a human
agent approves every offer. Classic ML decides WHO and WHY;
GenAI assists with HOW; humans stay accountable for the action.

DISCUSSION CHECKPOINTS (slide: 'The hard calls')
 1. Re-run with SAVE_RATE=0.15 (pessimistic). How does queue size change?
    What experiment would measure the TRUE save rate? (randomized holdout)
 2. The model will soon train on data contaminated by its own
    interventions. Sketch the uplift-modeling fix.
 3. CLV-ranked queues favour high-bill urban customers. Where would you
    document and review that fairness decision (Module 5)?
""")
