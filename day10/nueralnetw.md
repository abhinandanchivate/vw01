# Neural Network in Machine Learning — 5W1H

| Question   | Explanation                                                                                                                                                                                                           |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What?**  | A neural network is a machine-learning model made of connected processing units called **neurons**. It learns relationships between input features and an expected output.                                            |
| **Why?**   | It is used to identify complex and non-linear patterns that may be difficult for traditional algorithms to learn.                                                                                                     |
| **Who?**   | Data scientists, machine-learning engineers, AI engineers, researchers, analysts, and software developers use neural networks.                                                                                        |
| **When?**  | It is used when the dataset is sufficiently large, relationships are complex, and simpler algorithms do not provide satisfactory results.                                                                             |
| **Where?** | Neural networks are used in image recognition, speech recognition, fraud detection, medical diagnosis, predictive maintenance, recommendation systems, forecasting, and autonomous vehicles.                          |
| **How?**   | Data passes through an input layer, one or more hidden layers, and an output layer. The network calculates predictions, measures errors, and updates its weights using backpropagation and an optimization algorithm. |

## Basic Architecture

```text
Input Layer          Hidden Layer          Output Layer

Mileage ───────┐       ○
               ├─────► ○ ───────┐
Service Gap ───┤       ○         ├─────► Vehicle Risk
               ├─────► ○ ───────┘
Engine Alerts ─┤       ○
               │
Brake Wear ────┘
```

## Main Components

| Component               | Purpose                                  |
| ----------------------- | ---------------------------------------- |
| **Input layer**         | Receives feature values                  |
| **Hidden layers**       | Learn complex patterns                   |
| **Neuron**              | Performs a weighted calculation          |
| **Weights**             | Represent the importance of each input   |
| **Bias**                | Helps adjust the neuron’s output         |
| **Activation function** | Introduces non-linearity                 |
| **Output layer**        | Produces the final prediction            |
| **Loss function**       | Measures prediction error                |
| **Optimizer**           | Updates weights to reduce error          |
| **Backpropagation**     | Sends error backward through the network |

## How a Neuron Works

A neuron first calculates:

[
z = w_1x_1 + w_2x_2 + w_3x_3 + b
]

Where:

* (x) represents input values.
* (w) represents weights.
* (b) represents bias.
* (z) is the weighted result.

An activation function is then applied:

[
Output = Activation(z)
]

## Example: Vehicle Service-Risk Prediction

A vehicle company wants to predict whether a vehicle has a high breakdown risk.

### Input Features

| Feature             | Example value |
| ------------------- | ------------: |
| Mileage             |     85,000 km |
| Service gap         |     16 months |
| Engine alerts       |             6 |
| Brake wear          |           82% |
| Customer complaints |             4 |

### Neural Network Process

```text
Vehicle Information
        ↓
Input Layer
        ↓
Hidden Layer 1
Learns basic relationships
        ↓
Hidden Layer 2
Learns complex risk patterns
        ↓
Output Layer
        ↓
High-Risk Probability = 91%
```

### Output

```text
Predicted category: High Risk
Failure probability: 91%
```

## Neural Network Training Process

```text
1. Provide training data
        ↓
2. Initialize weights
        ↓
3. Perform forward propagation
        ↓
4. Generate prediction
        ↓
5. Calculate loss
        ↓
6. Perform backpropagation
        ↓
7. Update weights
        ↓
8. Repeat for multiple epochs
```

## Classification and Regression

| Problem Type               | Output-layer example | Common loss function      |
| -------------------------- | -------------------- | ------------------------- |
| Binary classification      | Sigmoid              | Binary cross-entropy      |
| Multi-class classification | Softmax              | Categorical cross-entropy |
| Regression                 | Linear neuron        | Mean squared error        |

## Simple Python Example

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. CREATE SAMPLE DATA
# --------------------------------------------------

data = pd.DataFrame({
    "mileage_km": [
        25000, 35000, 45000, 55000, 65000,
        75000, 85000, 95000, 105000, 120000
    ],
    "service_gap_months": [
        3, 4, 5, 7, 8,
        10, 13, 15, 18, 22
    ],
    "engine_alerts": [
        0, 1, 1, 2, 2,
        3, 5, 6, 7, 8
    ],
    "brake_wear_percent": [
        20, 28, 35, 42, 50,
        60, 72, 80, 88, 95
    ],
    "high_risk": [
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    ]
})


# --------------------------------------------------
# 2. SEPARATE INPUT AND OUTPUT
# --------------------------------------------------

X = data.drop("high_risk", axis=1)
y = data["high_risk"]


# --------------------------------------------------
# 3. SPLIT THE DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. SCALE THE FEATURES
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 5. CREATE THE NEURAL NETWORK
# --------------------------------------------------

model = MLPClassifier(
    hidden_layer_sizes=(8, 4),
    activation="relu",
    solver="adam",
    max_iter=2000,
    random_state=42
)


# --------------------------------------------------
# 6. TRAIN THE MODEL
# --------------------------------------------------

model.fit(X_train_scaled, y_train)


# --------------------------------------------------
# 7. TEST THE MODEL
# --------------------------------------------------

predictions = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))


# --------------------------------------------------
# 8. PREDICT A NEW VEHICLE
# --------------------------------------------------

new_vehicle = pd.DataFrame({
    "mileage_km": [90000],
    "service_gap_months": [16],
    "engine_alerts": [6],
    "brake_wear_percent": [85]
})

new_vehicle_scaled = scaler.transform(new_vehicle)

prediction = model.predict(new_vehicle_scaled)[0]
probability = model.predict_proba(new_vehicle_scaled)[0][1]

print("\nPredicted risk:", "High Risk" if prediction == 1 else "Low Risk")
print("High-risk probability:", round(probability, 4))
```

Here, `hidden_layer_sizes=(8, 4)` means the network contains:

```text
Input layer: 4 features
Hidden layer 1: 8 neurons
Hidden layer 2: 4 neurons
Output layer: Risk prediction
```
