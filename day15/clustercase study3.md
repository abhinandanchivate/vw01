
# Case Study: Volkswagen Service Complaint Intelligence

## Problem

Volkswagen receives complaints from multiple channels:

```text
Call center
Service center notes
Mobile app feedback
Warranty claims
Roadside assistance notes
```

The business wants to automatically group complaints and identify:

```text
1. Common complaint categories
2. Urgent service clusters
3. Possible affected vehicle systems
4. Dealer-level issue patterns
5. Recommended service actions
```

---

# Sample Data

```python
complaints = [
    {
        "complaint_id": "C001",
        "model": "Taigun",
        "year": 2022,
        "mileage": 78500,
        "dealer": "Pune-East",
        "text": "Vehicle takes longer to start in the morning and battery voltage is low"
    },
    {
        "complaint_id": "C002",
        "model": "Taigun",
        "year": 2022,
        "mileage": 81200,
        "dealer": "Pune-East",
        "text": "Slow cranking during startup, customer says battery warning appeared"
    },
    {
        "complaint_id": "C003",
        "model": "Virtus",
        "year": 2023,
        "mileage": 41000,
        "dealer": "Mumbai-West",
        "text": "Engine warning light comes on during highway driving"
    },
    {
        "complaint_id": "C004",
        "model": "Virtus",
        "year": 2023,
        "mileage": 39500,
        "dealer": "Mumbai-West",
        "text": "Check engine light blinking and vehicle feels underpowered"
    },
    {
        "complaint_id": "C005",
        "model": "Kushaq",
        "year": 2021,
        "mileage": 92000,
        "dealer": "Pune-East",
        "text": "Brake noise from front wheels, brake pedal feels hard"
    },
    {
        "complaint_id": "C006",
        "model": "Kushaq",
        "year": 2021,
        "mileage": 88000,
        "dealer": "Pune-East",
        "text": "Squeaking sound while braking and vibration felt on pedal"
    },
    {
        "complaint_id": "C007",
        "model": "Taigun",
        "year": 2022,
        "mileage": 76000,
        "dealer": "Bangalore-North",
        "text": "Mileage dropped after last service and fuel consumption increased"
    },
    {
        "complaint_id": "C008",
        "model": "Taigun",
        "year": 2022,
        "mileage": 74000,
        "dealer": "Bangalore-North",
        "text": "Vehicle is consuming more fuel than usual in city traffic"
    },
    {
        "complaint_id": "C009",
        "model": "Virtus",
        "year": 2023,
        "mileage": 36000,
        "dealer": "Mumbai-West",
        "text": "AC cooling is poor and cabin takes too long to cool"
    },
    {
        "complaint_id": "C010",
        "model": "Virtus",
        "year": 2023,
        "mileage": 37000,
        "dealer": "Mumbai-West",
        "text": "Air conditioner not cooling properly during afternoon driving"
    }
]
```

---

# Complete Python Code

```python
import os
import json
import pandas as pd
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

complaints = [
    {
        "complaint_id": "C001",
        "model": "Taigun",
        "year": 2022,
        "mileage": 78500,
        "dealer": "Pune-East",
        "text": "Vehicle takes longer to start in the morning and battery voltage is low"
    },
    {
        "complaint_id": "C002",
        "model": "Taigun",
        "year": 2022,
        "mileage": 81200,
        "dealer": "Pune-East",
        "text": "Slow cranking during startup, customer says battery warning appeared"
    },
    {
        "complaint_id": "C003",
        "model": "Virtus",
        "year": 2023,
        "mileage": 41000,
        "dealer": "Mumbai-West",
        "text": "Engine warning light comes on during highway driving"
    },
    {
        "complaint_id": "C004",
        "model": "Virtus",
        "year": 2023,
        "mileage": 39500,
        "dealer": "Mumbai-West",
        "text": "Check engine light blinking and vehicle feels underpowered"
    },
    {
        "complaint_id": "C005",
        "model": "Kushaq",
        "year": 2021,
        "mileage": 92000,
        "dealer": "Pune-East",
        "text": "Brake noise from front wheels, brake pedal feels hard"
    },
    {
        "complaint_id": "C006",
        "model": "Kushaq",
        "year": 2021,
        "mileage": 88000,
        "dealer": "Pune-East",
        "text": "Squeaking sound while braking and vibration felt on pedal"
    },
    {
        "complaint_id": "C007",
        "model": "Taigun",
        "year": 2022,
        "mileage": 76000,
        "dealer": "Bangalore-North",
        "text": "Mileage dropped after last service and fuel consumption increased"
    },
    {
        "complaint_id": "C008",
        "model": "Taigun",
        "year": 2022,
        "mileage": 74000,
        "dealer": "Bangalore-North",
        "text": "Vehicle is consuming more fuel than usual in city traffic"
    },
    {
        "complaint_id": "C009",
        "model": "Virtus",
        "year": 2023,
        "mileage": 36000,
        "dealer": "Mumbai-West",
        "text": "AC cooling is poor and cabin takes too long to cool"
    },
    {
        "complaint_id": "C010",
        "model": "Virtus",
        "year": 2023,
        "mileage": 37000,
        "dealer": "Mumbai-West",
        "text": "Air conditioner not cooling properly during afternoon driving"
    }
]

df = pd.DataFrame(complaints)

# 1. Create rich text for embeddings
df["embedding_text"] = (
    "Model: " + df["model"] +
    ". Year: " + df["year"].astype(str) +
    ". Mileage: " + df["mileage"].astype(str) +
    ". Complaint: " + df["text"]
)

# 2. Generate OpenAI embeddings
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=df["embedding_text"].tolist()
)

text_embeddings = [item.embedding for item in response.data]

# 3. Add numerical mileage feature
scaler = StandardScaler()
mileage_scaled = scaler.fit_transform(df[["mileage"]])

# 4. Combine embedding + mileage feature
combined_features = []

for emb, mileage in zip(text_embeddings, mileage_scaled):
    combined_features.append(emb + mileage.tolist())

# 5. Cluster complaints
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(combined_features)

print(df[["complaint_id", "model", "dealer", "text", "cluster"]])

# 6. Ask ChatGPT to analyze each cluster
def analyze_cluster(cluster_id, records):
    prompt = f"""
You are a Volkswagen service analytics expert.

Analyze this complaint cluster:

{json.dumps(records, indent=2)}

Give output in JSON:

{{
  "cluster_name": "",
  "common_theme": "",
  "possible_vehicle_system": "",
  "risk_level": "Low / Medium / High",
  "dealer_pattern": "",
  "recommended_inspection": "",
  "business_action": ""
}}

Rules:
- Do not give confirmed mechanical diagnosis.
- Use words like may indicate, could be related to, requires inspection.
- Mention if complaints are concentrated by dealer, model, mileage, or year.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text


cluster_outputs = []

for cluster_id in sorted(df["cluster"].unique()):
    records = df[df["cluster"] == cluster_id].to_dict(orient="records")
    result = analyze_cluster(cluster_id, records)

    cluster_outputs.append({
        "cluster": cluster_id,
        "analysis": result
    })

for item in cluster_outputs:
    print("\n==============================")
    print("Cluster:", item["cluster"])
    print("==============================")
    print(item["analysis"])

# 7. Visualization
pca = PCA(n_components=2)
points = pca.fit_transform(combined_features)

df["x"] = points[:, 0]
df["y"] = points[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(df["x"], df["y"], c=df["cluster"])

for i, row in df.iterrows():
    plt.text(row["x"], row["y"], row["complaint_id"])

plt.title("Complex Complaint Clustering using OpenAI Embeddings")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
```

---

# What Makes This Complex?

This example does not use only complaint text.

It uses:

```text
Complaint text
Vehicle model
Model year
Mileage
Dealer location
LLM embeddings
Numerical feature scaling
KMeans clustering
ChatGPT-based cluster interpretation
```

---

# Expected Cluster Output

Example:

```json
{
  "cluster_name": "Battery and Starting Issues",
  "common_theme": "Customers report slow cranking and longer startup time.",
  "possible_vehicle_system": "Battery, starter motor, charging system",
  "risk_level": "High",
  "dealer_pattern": "Mostly seen in Pune-East dealer records",
  "recommended_inspection": "Battery health test, charging system test, starter motor inspection",
  "business_action": "Review battery-related complaints for Taigun vehicles above 75,000 km"
}
```

---

# Final Business Value

| Output                 | Business Use                         |
| ---------------------- | ------------------------------------ |
| Cluster Name           | Understand complaint category        |
| Risk Level             | Prioritize service queue             |
| Dealer Pattern         | Identify location-specific issues    |
| Model Pattern          | Identify vehicle-specific complaints |
| Recommended Inspection | Guide service advisors               |
| Business Action        | Improve customer service planning    |
