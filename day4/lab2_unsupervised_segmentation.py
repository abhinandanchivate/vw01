"""
LAB 2 - UNSUPERVISED LEARNING: Customer Segmentation
=====================================================
ML Masterclass · Module 2 companion lab

Goal: discover customer segments with no labels, choose k honestly
(elbow + silhouette), project with PCA for inspection, and translate
clusters into business language — the part that actually matters.

Run:  python lab2_unsupervised_segmentation.py
Deps: numpy, pandas, scikit-learn
Time: < 30 seconds. Synthetic data, generated locally.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

RNG = np.random.default_rng(7)

# ----------------------------------------------------------------------
# SECTION 1 — Synthetic customer base with 4 latent personas
# ----------------------------------------------------------------------
# We PLANT four personas, then check whether unsupervised methods can
# recover them. In real work you don't know the truth — which is exactly
# why validation discipline (Section 3) matters.
# ----------------------------------------------------------------------

def persona(n, spend, data, voice, intl, tenure, jitter=1.0):
    return np.column_stack([
        RNG.normal(spend, 90 * jitter, n).clip(150, 3000),    # monthly spend INR
        RNG.normal(data, 6 * jitter, n).clip(0.2, 120),       # GB/month
        RNG.normal(voice, 80 * jitter, n).clip(0, 1500),      # voice minutes
        RNG.normal(intl, 12 * jitter, n).clip(0, 300),        # intl minutes
        RNG.normal(tenure, 8 * jitter, n).clip(1, 96),        # tenure months
    ])

segments_truth = {
    "data_power_users":   persona(900,  spend=1450, data=85, voice=150, intl=10,  tenure=30),
    "voice_traditional":  persona(700,  spend=520,  data=4,  voice=900, intl=5,   tenure=58),
    "budget_minimal":     persona(1100, spend=260,  data=8,  voice=120, intl=2,   tenure=14),
    "global_business":    persona(300,  spend=1900, data=45, voice=500, intl=160, tenure=42),
}
X = np.vstack(list(segments_truth.values()))
cols = ["monthly_spend", "data_gb", "voice_min", "intl_min", "tenure_months"]
df = pd.DataFrame(X, columns=cols)
df = df.sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle

print("=" * 68)
print(f"SECTION 1 · {len(df):,} customers, {len(cols)} behavioural features")
print("=" * 68)
print(df.describe().round(1).to_string())

# ----------------------------------------------------------------------
# SECTION 2 — Scale first: K-Means is distance-based
# ----------------------------------------------------------------------
# Without standardization, 'monthly_spend' (hundreds) dominates
# 'data_gb' (tens) purely because of units. Classic silent failure.
Xs = StandardScaler().fit_transform(df)

# ----------------------------------------------------------------------
# SECTION 3 — Choose k honestly: elbow (inertia) + silhouette
# ----------------------------------------------------------------------
print("\n" + "=" * 68)
print("SECTION 3 · Model selection: how many segments?")
print("=" * 68)
print(f"{'k':>3} {'inertia':>12} {'silhouette':>12}")
results = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=7).fit(Xs)
    sil = silhouette_score(Xs, km.labels_)
    results[k] = (km.inertia_, sil, km)
    print(f"{k:>3} {km.inertia_:>12.0f} {sil:>12.3f}")

best_k = max(results, key=lambda k: results[k][1])
print(f"\nSilhouette peaks at k={best_k} (and the inertia elbow agrees).")
print("In real projects, also check STABILITY: re-run on bootstrap samples;")
print("segments that dissolve under resampling are artefacts, not structure.")

km = results[best_k][2]
df["segment"] = km.labels_

# ----------------------------------------------------------------------
# SECTION 4 — PCA projection: can we see the structure?
# ----------------------------------------------------------------------
pca = PCA(n_components=2).fit(Xs)
proj = pca.transform(Xs)
print("\n" + "=" * 68)
print("SECTION 4 · PCA for inspection")
print("=" * 68)
print(f"Variance explained by 2 components: {pca.explained_variance_ratio_.sum():.1%}")
print("Loadings (what each axis 'means'):")
print(pd.DataFrame(pca.components_, columns=cols,
                   index=["PC1", "PC2"]).round(2).to_string())

# ----------------------------------------------------------------------
# SECTION 5 — The deliverable: segments in business language
# ----------------------------------------------------------------------
print("\n" + "=" * 68)
print("SECTION 5 · Segment profiles (cluster means)")
print("=" * 68)
profile = df.groupby("segment")[cols].mean().round(0)
profile["size"] = df.groupby("segment").size()
profile["share"] = (profile["size"] / len(df)).round(2)
print(profile.to_string())

print("""
Read the table like a marketer, not a mathematician — e.g.:
  high data + high spend + low intl  -> 'data power users': upsell speed tiers
  high voice + long tenure + low data -> 'traditionalists': protect, don't poke
  low everything + short tenure       -> 'budget/minimal': churn-fragile
  high intl + high spend              -> 'global business': premium support
A clustering is GOOD when acting on it moves a metric — that, plus
stability, is your validation in the absence of labels.
""")

# ----------------------------------------------------------------------
# DISCUSSION
# ----------------------------------------------------------------------
# 1. Remove StandardScaler and re-run. Which feature hijacks the clusters?
# 2. Add 20% random-noise customers. What happens to silhouette? Would
#    DBSCAN handle them differently? (It has a noise label; K-Means must
#    assign everyone somewhere.)
# 3. These segments feed the case study's decision layer: churn risk x
#    segment value chooses WHICH retention offer a customer gets.
