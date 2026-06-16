# ================================================================
# VEHICLE URGENT-SERVICE RISK
# COMPLETE EXPLORATORY DATA ANALYSIS CASE STUDY
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# Reproducible random generator
RNG = np.random.default_rng(42)

# ================================================================
# STEP 1: CREATE A SYNTHETIC DATASET
# ================================================================

total_records = 3000

vehicle_age_years = RNG.integers(
    low=0,
    high=16,
    size=total_records
)

mileage_km = np.clip(
    vehicle_age_years * 9000
    + RNG.normal(12000, 15000, total_records),
    3000,
    180000
).astype(int)

avg_monthly_km = np.clip(
    RNG.normal(1400, 500, total_records),
    300,
    4000
).astype(int)

service_gap_months = np.clip(
    RNG.normal(
        6 + 0.25 * vehicle_age_years,
        3,
        total_records
    ),
    1,
    24
).round(1)

engine_alerts = RNG.poisson(
    0.35 + mileage_km / 70000
)

brake_wear_percent = np.clip(
    10
    + mileage_km / 1800
    + RNG.normal(0, 9, total_records),
    5,
    100
).round(1)

battery_voltage = np.clip(
    12.8
    - 0.055 * vehicle_age_years
    + RNG.normal(0, 0.25, total_records),
    10.8,
    13.2
).round(2)

battery_health = np.where(
    battery_voltage < 11.8,
    "Low",
    np.where(
        battery_voltage < 12.3,
        "Normal",
        "Good"
    )
)

warranty_claims = RNG.poisson(
    0.08 + vehicle_age_years / 10
)

customer_complaints = RNG.poisson(
    0.15 + engine_alerts * 0.22
)

driving_pattern = RNG.choice(
    ["City", "Highway", "Mixed"],
    size=total_records,
    p=[0.45, 0.25, 0.30]
)

# ------------------------------------------------
# Create the target variable
# ------------------------------------------------

risk_score = (
    -6
    + 0.000022 * mileage_km
    + 0.075 * service_gap_months
    + 0.35 * engine_alerts
    + 0.035 * brake_wear_percent
    + 0.55 * (battery_health == "Low")
    + 0.25 * (driving_pattern == "City")
    + 0.18 * customer_complaints
)

risk_probability = 1 / (1 + np.exp(-risk_score))

target = np.where(
    RNG.random(total_records) < risk_probability,
    "High Risk",
    "Low Risk"
)

# ------------------------------------------------
# Create DataFrame
# ------------------------------------------------

data = pd.DataFrame({
    "vehicle_id": [
        f"VW-{i:05d}"
        for i in range(1, total_records + 1)
    ],
    "vehicle_age_years": vehicle_age_years,
    "mileage_km": mileage_km,
    "avg_monthly_km": avg_monthly_km,
    "service_gap_months": service_gap_months,
    "engine_alerts": engine_alerts,
    "brake_wear_percent": brake_wear_percent,
    "battery_voltage": battery_voltage,
    "battery_health": battery_health,
    "warranty_claims": warranty_claims,
    "customer_complaints": customer_complaints,
    "driving_pattern": driving_pattern,
    "target": target
})

# ================================================================
# STEP 2: INTRODUCE SOME DATA-QUALITY PROBLEMS
# ================================================================

# Add 2% missing values to selected columns
columns_with_missing_values = [
    "service_gap_months",
    "brake_wear_percent",
    "battery_health",
    "driving_pattern"
]

for column in columns_with_missing_values:
    missing_indexes = RNG.choice(
        data.index,
        size=int(0.02 * total_records),
        replace=False
    )

    data.loc[missing_indexes, column] = np.nan

# Add five duplicate records
data = pd.concat(
    [data, data.iloc[:5]],
    ignore_index=True
)

# Add eight unusual mileage values
outlier_indexes = RNG.choice(
    data.index[:-5],
    size=8,
    replace=False
)

data.loc[outlier_indexes, "mileage_km"] = (
    data.loc[outlier_indexes, "mileage_km"] * 2.5
).astype(int)

print("=" * 70)
print("DATASET CREATED")
print("=" * 70)
print("Dataset shape:", data.shape)

# Save original dataset
original_file = Path.cwd() / "vehicle_service_eda_dataset.csv"
data.to_csv(original_file, index=False)

print("Original dataset saved to:")
print(original_file)

# ================================================================
# STEP 3: INITIAL DATA INSPECTION
# ================================================================

print("\n" + "=" * 70)
print("FIRST FIVE RECORDS")
print("=" * 70)
print(data.head())

print("\n" + "=" * 70)
print("LAST FIVE RECORDS")
print("=" * 70)
print(data.tail())

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)
print(data.columns.tolist())

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)
print(data.dtypes)

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)
data.info()

# ================================================================
# STEP 4: DESCRIPTIVE STATISTICS
# ================================================================

print("\n" + "=" * 70)
print("NUMERIC COLUMN STATISTICS")
print("=" * 70)

print(
    data.describe()
        .T
        .round(2)
)

print("\n" + "=" * 70)
print("CATEGORICAL COLUMN STATISTICS")
print("=" * 70)

print(
    data.describe(include="object")
        .T
)

# ================================================================
# STEP 5: CHECK DATA QUALITY
# ================================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_values = pd.DataFrame({
    "missing_count": data.isna().sum(),
    "missing_percentage":
        (data.isna().mean() * 100).round(2)
})

print(
    missing_values[
        missing_values["missing_count"] > 0
    ]
)

print("\nDuplicate rows:", data.duplicated().sum())

print(
    "\nDuplicate vehicle IDs:",
    data["vehicle_id"].duplicated().sum()
)

# Check unexpected categorical values
print("\nBattery-health values:")
print(data["battery_health"].value_counts(dropna=False))

print("\nDriving-pattern values:")
print(data["driving_pattern"].value_counts(dropna=False))

print("\nTarget values:")
print(data["target"].value_counts(dropna=False))

# ================================================================
# STEP 6: REMOVE DUPLICATE RECORDS
# ================================================================

cleaned_data = data.drop_duplicates().copy()

print("\nRows before duplicate removal:", len(data))
print("Rows after duplicate removal:", len(cleaned_data))
print("Rows removed:", len(data) - len(cleaned_data))

# ================================================================
# STEP 7: HANDLE MISSING VALUES
# ================================================================

numeric_missing_columns = [
    "service_gap_months",
    "brake_wear_percent"
]

categorical_missing_columns = [
    "battery_health",
    "driving_pattern"
]

# Median is resistant to extreme values
for column in numeric_missing_columns:
    median_value = cleaned_data[column].median()

    cleaned_data[column] = (
        cleaned_data[column]
        .fillna(median_value)
    )

    print(
        f"{column} filled using median:",
        median_value
    )

# Mode fills categorical columns
for column in categorical_missing_columns:
    mode_value = cleaned_data[column].mode()[0]

    cleaned_data[column] = (
        cleaned_data[column]
        .fillna(mode_value)
    )

    print(
        f"{column} filled using mode:",
        mode_value
    )

print(
    "\nRemaining missing values:",
    cleaned_data.isna().sum().sum()
)

# ================================================================
# STEP 8: TARGET-DISTRIBUTION ANALYSIS
# ================================================================

target_count = cleaned_data["target"].value_counts()

target_percentage = (
    cleaned_data["target"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

target_summary = pd.DataFrame({
    "count": target_count,
    "percentage": target_percentage
})

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)
print(target_summary)

plt.figure(figsize=(7, 5))

sns.countplot(
    data=cleaned_data,
    x="target"
)

plt.title("Distribution of Urgent-Service Risk")
plt.xlabel("Service-risk category")
plt.ylabel("Number of vehicles")
plt.tight_layout()
plt.show()

# ================================================================
# STEP 9: UNIVARIATE EDA
# Study one feature at a time
# ================================================================

# ------------------------------------------------
# 9.1 Mileage distribution
# ------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=cleaned_data,
    x="mileage_km",
    bins=30,
    kde=True
)

plt.title("Distribution of Vehicle Mileage")
plt.xlabel("Mileage in kilometres")
plt.ylabel("Number of vehicles")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 9.2 Mileage box plot
# ------------------------------------------------

plt.figure(figsize=(9, 4))

sns.boxplot(
    data=cleaned_data,
    x="mileage_km"
)

plt.title("Mileage Box Plot")
plt.xlabel("Mileage in kilometres")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 9.3 Brake-wear distribution
# ------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=cleaned_data,
    x="brake_wear_percent",
    bins=25,
    kde=True
)

plt.title("Distribution of Brake Wear")
plt.xlabel("Brake wear percentage")
plt.ylabel("Number of vehicles")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 9.4 Engine-alert distribution
# ------------------------------------------------

plt.figure(figsize=(9, 5))

sns.countplot(
    data=cleaned_data,
    x="engine_alerts"
)

plt.title("Distribution of Engine Alerts")
plt.xlabel("Number of engine alerts")
plt.ylabel("Number of vehicles")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 9.5 Battery-health distribution
# ------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=cleaned_data,
    x="battery_health",
    order=["Good", "Normal", "Low"]
)

plt.title("Battery-Health Distribution")
plt.xlabel("Battery health")
plt.ylabel("Number of vehicles")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 9.6 Driving-pattern distribution
# ------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=cleaned_data,
    x="driving_pattern"
)

plt.title("Driving-Pattern Distribution")
plt.xlabel("Driving pattern")
plt.ylabel("Number of vehicles")
plt.tight_layout()
plt.show()

# ================================================================
# STEP 10: BIVARIATE EDA
# Study relationships between two variables
# ================================================================

# ------------------------------------------------
# 10.1 Numeric features grouped by target
# ------------------------------------------------

numeric_target_summary = (
    cleaned_data.groupby("target")[
        [
            "mileage_km",
            "service_gap_months",
            "engine_alerts",
            "brake_wear_percent",
            "battery_voltage",
            "warranty_claims",
            "customer_complaints"
        ]
    ]
    .mean()
    .round(2)
)

print("\n" + "=" * 70)
print("AVERAGE FEATURE VALUES BY TARGET")
print("=" * 70)

print(numeric_target_summary)

# ------------------------------------------------
# 10.2 Mileage versus target
# ------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=cleaned_data,
    x="target",
    y="mileage_km"
)

plt.title("Mileage by Service-Risk Category")
plt.xlabel("Service-risk category")
plt.ylabel("Mileage in kilometres")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 10.3 Brake wear versus target
# ------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=cleaned_data,
    x="target",
    y="brake_wear_percent"
)

plt.title("Brake Wear by Service-Risk Category")
plt.xlabel("Service-risk category")
plt.ylabel("Brake wear percentage")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 10.4 Engine alerts versus target
# ------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=cleaned_data,
    x="target",
    y="engine_alerts"
)

plt.title("Engine Alerts by Service-Risk Category")
plt.xlabel("Service-risk category")
plt.ylabel("Number of engine alerts")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 10.5 Mileage versus brake wear
# ------------------------------------------------

sample_data = cleaned_data.sample(
    n=600,
    random_state=42
)

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=sample_data,
    x="mileage_km",
    y="brake_wear_percent",
    hue="target",
    alpha=0.7
)

plt.title("Mileage versus Brake Wear")
plt.xlabel("Mileage in kilometres")
plt.ylabel("Brake wear percentage")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 10.6 Battery health versus target
# ------------------------------------------------

battery_risk_table = pd.crosstab(
    cleaned_data["battery_health"],
    cleaned_data["target"],
    normalize="index"
).mul(100).round(2)

print("\nRisk percentage by battery health:")
print(battery_risk_table)

battery_high_risk = (
    battery_risk_table["High Risk"]
    .reindex(["Good", "Normal", "Low"])
)

battery_high_risk.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("High-Risk Percentage by Battery Health")
plt.xlabel("Battery health")
plt.ylabel("High-risk vehicles (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 10.7 Driving pattern versus target
# ------------------------------------------------

driving_risk_table = pd.crosstab(
    cleaned_data["driving_pattern"],
    cleaned_data["target"],
    normalize="index"
).mul(100).round(2)

print("\nRisk percentage by driving pattern:")
print(driving_risk_table)

driving_risk_table["High Risk"].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("High-Risk Percentage by Driving Pattern")
plt.xlabel("Driving pattern")
plt.ylabel("High-risk vehicles (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ================================================================
# STEP 11: MULTIVARIATE EDA
# Study multiple features together
# ================================================================

# Convert target into a numeric column for correlation
cleaned_data["target_num"] = cleaned_data["target"].map({
    "Low Risk": 0,
    "High Risk": 1
})

numeric_columns = [
    "vehicle_age_years",
    "mileage_km",
    "avg_monthly_km",
    "service_gap_months",
    "engine_alerts",
    "brake_wear_percent",
    "battery_voltage",
    "warranty_claims",
    "customer_complaints",
    "target_num"
]

correlation_matrix = (
    cleaned_data[numeric_columns]
    .corr()
    .round(2)
)

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

print(correlation_matrix)

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# Features most related to target
# ------------------------------------------------

target_correlations = (
    correlation_matrix["target_num"]
    .drop("target_num")
    .sort_values(ascending=False)
)

print("\nFeatures correlated with High Risk:")
print(target_correlations)

# ------------------------------------------------
# Pair plot using a smaller sample
# ------------------------------------------------

pairplot_data = cleaned_data.sample(
    n=500,
    random_state=42
)

sns.pairplot(
    data=pairplot_data,
    vars=[
        "mileage_km",
        "service_gap_months",
        "engine_alerts",
        "brake_wear_percent",
        "battery_voltage"
    ],
    hue="target",
    corner=True
)

plt.show()

# ================================================================
# STEP 12: OUTLIER DETECTION USING IQR
# ================================================================

Q1 = cleaned_data["mileage_km"].quantile(0.25)
Q3 = cleaned_data["mileage_km"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = max(
    0,
    Q1 - 1.5 * IQR
)

upper_bound = Q3 + 1.5 * IQR

cleaned_data["mileage_outlier"] = (
    (cleaned_data["mileage_km"] < lower_bound)
    | (cleaned_data["mileage_km"] > upper_bound)
)

mileage_outliers = cleaned_data[
    cleaned_data["mileage_outlier"]
]

print("\n" + "=" * 70)
print("MILEAGE OUTLIER ANALYSIS")
print("=" * 70)

print("Q1:", round(Q1, 2))
print("Q3:", round(Q3, 2))
print("IQR:", round(IQR, 2))
print("Lower bound:", round(lower_bound, 2))
print("Upper bound:", round(upper_bound, 2))
print("Number of mileage outliers:", len(mileage_outliers))

print(
    mileage_outliers[
        [
            "vehicle_id",
            "mileage_km",
            "vehicle_age_years",
            "target"
        ]
    ]
    .sort_values("mileage_km", ascending=False)
    .head(15)
)

# Do not immediately delete outliers.
# Preserve original mileage and create a capped feature.

cleaned_data["mileage_km_capped"] = (
    cleaned_data["mileage_km"]
    .clip(
        lower=lower_bound,
        upper=upper_bound
    )
)

# ================================================================
# STEP 13: FEATURE ENGINEERING
# ================================================================

# ------------------------------------------------
# Create mileage categories
# ------------------------------------------------

cleaned_data["mileage_category"] = pd.cut(
    cleaned_data["mileage_km_capped"],
    bins=[
        0,
        30000,
        60000,
        100000,
        np.inf
    ],
    labels=[
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

# ------------------------------------------------
# Create service-gap category
# ------------------------------------------------

cleaned_data["service_gap_category"] = pd.cut(
    cleaned_data["service_gap_months"],
    bins=[
        0,
        6,
        12,
        np.inf
    ],
    labels=[
        "On Time",
        "Delayed",
        "Highly Delayed"
    ]
)

# ------------------------------------------------
# Create maintenance warning count
# ------------------------------------------------

cleaned_data["maintenance_warning_count"] = (
    (cleaned_data["engine_alerts"] > 2).astype(int)
    + (cleaned_data["brake_wear_percent"] > 75).astype(int)
    + (cleaned_data["battery_health"] == "Low").astype(int)
    + (cleaned_data["service_gap_months"] > 12).astype(int)
)

print("\nFeature-engineering sample:")

print(
    cleaned_data[
        [
            "vehicle_id",
            "mileage_category",
            "service_gap_category",
            "maintenance_warning_count",
            "target"
        ]
    ].head(10)
)

# ------------------------------------------------
# Risk by warning count
# ------------------------------------------------

warning_risk = pd.crosstab(
    cleaned_data["maintenance_warning_count"],
    cleaned_data["target"],
    normalize="index"
).mul(100).round(2)

print("\nRisk percentage by warning count:")
print(warning_risk)

warning_risk["High Risk"].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("High-Risk Percentage by Maintenance Warnings")
plt.xlabel("Number of maintenance-warning conditions")
plt.ylabel("High-risk vehicles (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ================================================================
# STEP 14: CREATE ML-READY DATASET
# ================================================================

ml_data = cleaned_data.drop(
    columns=[
        "vehicle_id",
        "mileage_km",
        "mileage_outlier",
        "target_num"
    ]
).copy()

# Convert target into 0 and 1
ml_data["target"] = ml_data["target"].map({
    "Low Risk": 0,
    "High Risk": 1
})

# One-hot encode categorical columns
ml_data = pd.get_dummies(
    ml_data,
    columns=[
        "battery_health",
        "driving_pattern",
        "mileage_category",
        "service_gap_category"
    ],
    drop_first=True,
    dtype=int
)

print("\n" + "=" * 70)
print("ML-READY DATASET")
print("=" * 70)

print("Shape:", ml_data.shape)
print(ml_data.head())
print("\nRemaining missing values:", ml_data.isna().sum().sum())

# Save cleaned EDA dataset
cleaned_file = Path.cwd() / "vehicle_service_eda_cleaned.csv"
cleaned_data.to_csv(cleaned_file, index=False)

# Save ML-ready dataset
ml_file = Path.cwd() / "vehicle_service_ml_ready.csv"
ml_data.to_csv(ml_file, index=False)

print("\nCleaned EDA dataset saved to:")
print(cleaned_file)

print("\nML-ready dataset saved to:")
print(ml_file)

# ================================================================
# STEP 15: FINAL AUTOMATIC EDA SUMMARY
# ================================================================

high_risk_percentage = (
    cleaned_data["target"]
    .eq("High Risk")
    .mean()
    * 100
)

group_summary = (
    cleaned_data.groupby("target")[
        [
            "mileage_km",
            "service_gap_months",
            "engine_alerts",
            "brake_wear_percent",
            "battery_voltage",
            "warranty_claims",
            "customer_complaints"
        ]
    ]
    .mean()
    .round(2)
)

print("\n" + "=" * 70)
print("FINAL EDA SUMMARY")
print("=" * 70)

print(
    f"Total clean records: {len(cleaned_data)}"
)

print(
    f"High-risk vehicles: {high_risk_percentage:.2f}%"
)

print("\nAverage feature values by target:")
print(group_summary)

print("\nTop correlations with target:")
print(target_correlations.head(7))
