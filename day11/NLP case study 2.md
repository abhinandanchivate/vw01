# Complex NLP Case Study: Automobile Service Complaint Intelligence System

## 1. Case Study Title

**AI-Based Vehicle Service Complaint Classification, Entity Extraction, Sentiment Analysis and Priority Routing**

---

## 2. Business Background

A large automobile manufacturer receives customer complaints from:

* Service-centre feedback forms
* Mobile application
* Emails
* Call-centre transcripts
* Social-media posts
* Dealer portals
* Vehicle diagnostic notes

The company receives more than **50,000 complaints every month**.

Complaints may relate to:

* Engine problems
* Brake failures
* Battery issues
* Service delays
* Incorrect billing
* Spare-part availability
* Warranty rejection
* Poor staff behaviour
* Repeated breakdowns
* Vehicle safety concerns

Currently, customer-service employees manually read each complaint and forward it to the appropriate department.

---

## 3. Business Problem

Manual complaint processing creates several challenges:

* Safety-critical complaints may not receive immediate attention.
* Similar complaints may be classified differently.
* Vehicle registration numbers and service IDs may be missed.
* Customer sentiment is not measured consistently.
* Complaints are sometimes sent to the wrong department.
* Repeated complaints from the same vehicle are difficult to identify.
* Management cannot easily identify emerging product defects.

The organization wants an NLP system that automatically understands each complaint and produces structured information.

---

# 4. Main Objective

Develop an NLP system that performs:

1. Text cleaning and normalization
2. Sentence and word tokenization
3. Lemmatization
4. Vehicle-specific Named Entity Recognition
5. Complaint-category classification
6. Sentiment analysis
7. Urgency prediction
8. Root-cause extraction
9. Duplicate complaint detection
10. Department routing
11. Recommended action generation

---

# 5. Sample Customer Complaints

| Complaint ID | Customer complaint                                                                                                                                       |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| V001         | My Tata Nexon suddenly lost power while driving on the Mumbai–Pune Expressway. The engine warning light appeared, and the vehicle stopped near Lonavala. |
| V002         | I submitted my Hyundai Creta at Sunrise Motors Pune on 15 June 2026, but the vehicle has still not been delivered. Service ID SER-98754.                 |
| V003         | The brakes of my Mahindra XUV700 stopped responding for a few seconds while driving at 80 km/h. This is extremely dangerous.                             |
| V004         | The service centre charged ₹18,500 for replacing the battery, although the vehicle is still under warranty.                                              |
| V005         | My vehicle MH12AB4587 has visited the workshop three times for the same engine-vibration problem, but the issue remains unresolved.                      |
| V006         | The technician was rude and refused to explain why my warranty claim was rejected.                                                                       |
| V007         | After the software update, the infotainment system keeps restarting every five minutes.                                                                  |
| V008         | I received my car after servicing, but the left door was scratched and the fuel level had reduced significantly.                                         |
| V009         | The electric vehicle battery range has dropped from 350 km to 210 km within six months.                                                                  |
| V010         | Smoke started coming from the engine compartment immediately after the scheduled service. Please arrange emergency assistance.                           |

---

# 6. Expected NLP Output

For complaint `V003`:

```text
The brakes of my Mahindra XUV700 stopped responding
for a few seconds while driving at 80 km/h.
This is extremely dangerous.
```

Expected structured output:

```json
{
  "complaint_id": "V003",
  "vehicle_brand": "Mahindra",
  "vehicle_model": "XUV700",
  "vehicle_component": "brakes",
  "speed": "80 km/h",
  "complaint_category": "Brake System Issue",
  "root_cause_area": "Braking System",
  "sentiment": "Highly Negative",
  "urgency": "Critical",
  "safety_risk": true,
  "department": "Vehicle Safety and Engineering",
  "recommended_action": "Contact customer immediately and arrange vehicle inspection"
}
```

---

# 7. NLP Architecture

```text
Customer complaint
        ↓
Text cleaning and normalization
        ↓
Sentence and word tokenization
        ↓
Lemmatization and POS tagging
        ↓
Vehicle-specific Named Entity Recognition
        ↓
Text vectorization or embeddings
        ↓
Complaint classification model
        ↓
Sentiment and urgency models
        ↓
Root-cause and safety-rule engine
        ↓
Department routing
        ↓
Structured report and dashboard
```

---

# 8. NLP Processing Stages

## Stage 1: Text Cleaning

The system must handle:

* Extra spaces
* Spelling mistakes
* Mixed uppercase and lowercase text
* Abbreviations
* Emojis
* Punctuation
* Hindi-English mixed messages
* Call-centre transcript noise

Example:

```text
brk not wrking properly!!! very dangerous 😡
```

Normalized form:

```text
brake not working properly very dangerous
```

---

## Stage 2: Tokenization

Input:

```text
The engine warning light appeared near Lonavala.
```

Tokens:

```text
The
engine
warning
light
appeared
near
Lonavala
.
```

Sentence tokenization must also correctly process long complaints containing multiple problems.

---

## Stage 3: Lemmatization

| Original word | Lemma   |
| ------------- | ------- |
| driving       | drive   |
| stopped       | stop    |
| responding    | respond |
| charged       | charge  |
| replacing     | replace |
| scratches     | scratch |

Lemmatization helps the system treat different grammatical forms as the same concept.

---

## Stage 4: Vehicle-Specific Named Entity Recognition

The NLP system must identify both general and automobile-specific entities.

| Entity type         | Example             |
| ------------------- | ------------------- |
| Vehicle brand       | Tata                |
| Vehicle model       | Nexon               |
| Registration number | MH12AB4587          |
| Service ID          | SER-98754           |
| Dealer              | Sunrise Motors Pune |
| Date                | 15 June 2026        |
| Location            | Lonavala            |
| Amount              | ₹18,500             |
| Vehicle component   | Brake               |
| Speed               | 80 km/h             |
| Mileage             | 72,000 km           |
| Warranty status     | Under warranty      |
| Software version    | v4.2.1              |

A standard NLP model may detect dates, locations and money, but custom training or rules are required for:

* Vehicle registration numbers
* Service IDs
* Vehicle components
* Diagnostic codes
* Software versions

---

# 9. Complaint Categories

| Category           | Example issue                                 |
| ------------------ | --------------------------------------------- |
| Engine Issue       | Power loss, overheating, vibration            |
| Brake System Issue | Brake failure, delayed response               |
| Battery Issue      | Battery drain, replacement problem            |
| EV Range Issue     | Range reduction, charging failure             |
| Transmission Issue | Gear shifting problem                         |
| Infotainment Issue | Screen restart, navigation failure            |
| Service Delay      | Vehicle not delivered on time                 |
| Billing Issue      | Excess or incorrect charges                   |
| Warranty Issue     | Warranty rejection                            |
| Body Damage        | Scratch or dent after servicing               |
| Staff Behaviour    | Rude or unhelpful employee                    |
| Repeated Complaint | Same issue occurs after multiple visits       |
| Safety Emergency   | Smoke, fire, brake failure or sudden shutdown |

---

# 10. Sentiment Analysis

The system should classify sentiment as:

* Positive
* Neutral
* Negative
* Highly Negative

Example:

```text
The service was completed on time and the staff was helpful.
```

Output:

```text
Positive
```

Example:

```text
The brakes failed while driving. This is extremely dangerous.
```

Output:

```text
Highly Negative
```

The system must understand negation.

```text
The service was not bad.
```

This should not automatically be classified as strongly negative merely because the word `bad` appears.

---

# 11. Urgency Classification

| Urgency  | Example                                                  |
| -------- | -------------------------------------------------------- |
| Critical | Brake failure, fire, smoke, accident risk                |
| High     | Engine shutdown, repeated breakdown, battery overheating |
| Medium   | Service delay, warranty issue, incorrect billing         |
| Low      | Minor infotainment issue, general enquiry                |

Critical keywords may include:

```text
fire
smoke
brake failed
accident
dangerous
vehicle stopped
emergency
overheating
```

Urgency should not depend only on keywords. Context is important.

Example:

```text
The fire extinguisher was replaced during service.
```

The presence of `fire` does not mean that the vehicle caught fire.

---

# 12. Root-Cause Extraction

The system must identify the likely root-cause area.

| Complaint text                | Root-cause area         |
| ----------------------------- | ----------------------- |
| Engine warning light appeared | Engine management       |
| Brake pedal became hard       | Braking system          |
| Battery range reduced         | EV battery system       |
| Screen restarts continuously  | Infotainment software   |
| Vehicle not delivered         | Service operations      |
| Warranty claim rejected       | Warranty administration |

---

# 13. Duplicate Complaint Detection

The system should detect complaints that describe the same issue using different words.

Complaint A:

```text
The engine vibrates when the car is idle.
```

Complaint B:

```text
Strong vibration is felt while the vehicle is stationary.
```

Although the words are different, sentence embeddings should show that they are semantically similar.

Possible output:

```text
Similarity score: 0.89
Possible duplicate: Yes
```

This can be implemented using:

* Sentence Transformers
* BERT embeddings
* Cosine similarity
* Vector database

---

# 14. Model Comparison

## TF-IDF with Logistic Regression

Suitable for:

* Initial classification baseline
* Fast training
* Explainable word importance
* Smaller datasets

Limitation:

* Weak understanding of context
* Cannot understand semantic similarity well

---

## Word2Vec or GloVe with LSTM

Suitable for:

* Sequential text classification
* Learning word relationships
* Medium-sized datasets

Limitation:

* Word vectors are static
* The same word receives the same vector in every context

Example:

```text
The battery is dead.
The phone battery lasts all day.
```

The meaning of `battery` changes with context, but static embeddings may not fully capture this.

---

## BiLSTM with Attention

Suitable for:

* Identifying important complaint words
* Understanding sequence information
* Improving long-text classification

Example attention focus:

```text
The [brakes] [stopped responding] while driving.
```

The model may assign higher attention weights to:

```text
brakes
stopped
responding
```

---

## BERT

Suitable for:

* Contextual text classification
* Sentiment analysis
* Complex complaint language
* Negation handling
* Semantic understanding

Example:

```text
The service centre did not resolve the engine issue.
```

BERT can understand that the complaint is negative because the issue remains unresolved.

---

# 15. Multi-Model Design

A complex implementation may use multiple NLP models.

| NLP task                       | Suggested approach                         |
| ------------------------------ | ------------------------------------------ |
| Tokenization and lemmatization | spaCy                                      |
| General NER                    | spaCy or transformer model                 |
| Vehicle-specific NER           | Custom spaCy NER or BERT NER               |
| Complaint classification       | Fine-tuned BERT                            |
| Sentiment analysis             | Fine-tuned transformer                     |
| Duplicate detection            | Sentence Transformer                       |
| Urgency detection              | BERT plus safety rules                     |
| Recommended action             | Rule engine or controlled generative model |

---

# 16. Training Dataset Structure

| Column         | Description             |
| -------------- | ----------------------- |
| complaint_id   | Unique complaint number |
| complaint_text | Original complaint      |
| vehicle_brand  | Manufacturer            |
| vehicle_model  | Vehicle model           |
| component      | Affected part           |
| category       | Complaint category      |
| sentiment      | Sentiment label         |
| urgency        | Priority label          |
| safety_risk    | Yes or No               |
| department     | Assigned department     |
| resolution     | Final action taken      |

Example:

```csv
complaint_id,complaint_text,vehicle_brand,vehicle_model,component,category,sentiment,urgency,safety_risk,department
V003,"The brakes stopped responding at 80 km/h",Mahindra,XUV700,Brake,Brake System Issue,Highly Negative,Critical,Yes,Vehicle Safety
```

---

# 17. Practical Task List

## Task 1: Data Preparation

* Load complaint data.
* Remove duplicate records.
* Handle missing complaints.
* Normalize text.
* Remove personally identifiable data where necessary.

## Task 2: Exploratory Text Analysis

Calculate:

* Most common complaint words
* Most frequent bigrams and trigrams
* Complaint count by category
* Sentiment distribution
* Critical complaint percentage
* Most frequently reported components

## Task 3: Linguistic Processing

Generate:

* Tokens
* Lemmas
* POS tags
* Dependency relationships
* Sentence boundaries
* Named entities

## Task 4: Custom Entity Recognition

Train or configure the model to detect:

* Vehicle registration number
* Service ID
* Diagnostic trouble code
* Vehicle component
* Dealer name
* Software version

## Task 5: Text Classification

Train models to predict:

```text
Engine Issue
Brake Issue
Battery Issue
Service Delay
Billing Issue
Warranty Issue
Safety Emergency
```

Compare:

* Naïve Bayes
* Logistic Regression
* SVM
* LSTM
* BERT

## Task 6: Sentiment Analysis

Predict:

```text
Positive
Neutral
Negative
Highly Negative
```

## Task 7: Urgency Classification

Predict:

```text
Low
Medium
High
Critical
```

## Task 8: Duplicate Detection

* Generate sentence embeddings.
* Calculate cosine similarity.
* Flag complaints above a selected threshold.

## Task 9: Department Routing

| Predicted category | Assigned department    |
| ------------------ | ---------------------- |
| Engine Issue       | Powertrain Engineering |
| Brake Issue        | Vehicle Safety         |
| Battery Issue      | Electrical Systems     |
| EV Range Issue     | EV Battery Team        |
| Service Delay      | Dealer Operations      |
| Billing Issue      | Accounts               |
| Warranty Issue     | Warranty Department    |
| Staff Behaviour    | Customer Experience    |

## Task 10: Reporting

Produce:

* Complaint-level output CSV
* Critical complaint report
* Dealer performance report
* Component failure dashboard
* Duplicate complaint report

---

# 18. Model Evaluation

## Classification metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

For critical safety complaints, **recall is especially important**.

Missing a genuine brake-failure complaint is more serious than incorrectly marking a normal complaint as critical.

## NER metrics

* Entity precision
* Entity recall
* Entity F1-score

## Duplicate detection metrics

* Precision at selected similarity threshold
* Recall of known duplicate pairs
* False-positive rate

---

# 19. Important Challenges

### Mixed-language complaints

```text
Car start nahi ho rahi and battery warning aa raha hai.
```

The system must handle Hindi-English mixed text.

### Spelling mistakes

```text
Break not workng properly.
```

Here, `Break` likely means `Brake`.

### Multiple problems in one complaint

```text
The engine is vibrating, the service was delayed,
and the staff charged an incorrect amount.
```

The complaint may need multiple labels:

```text
Engine Issue
Service Delay
Billing Issue
```

### Negation

```text
There is no problem with the brakes.
```

This must not be classified as a brake-failure complaint.

### Sarcasm

```text
Excellent service—I only had to visit the workshop four times.
```

Although `excellent` appears, the actual sentiment is negative.

---

# 20. Expected Final Output

| Complaint ID | Category               | Component       | Sentiment       | Urgency  | Safety risk | Department             |
| ------------ | ---------------------- | --------------- | --------------- | -------- | ----------- | ---------------------- |
| V001         | Engine Issue           | Engine          | Highly Negative | High     | Yes         | Powertrain Engineering |
| V002         | Service Delay          | General Service | Negative        | Medium   | No          | Dealer Operations      |
| V003         | Brake System Issue     | Brake           | Highly Negative | Critical | Yes         | Vehicle Safety         |
| V004         | Billing/Warranty Issue | Battery         | Negative        | Medium   | No          | Warranty and Accounts  |
| V005         | Repeated Engine Issue  | Engine          | Highly Negative | High     | Possible    | Quality Engineering    |
| V010         | Safety Emergency       | Engine          | Highly Negative | Critical | Yes         | Emergency Assistance   |

---

# 21. Final Problem Statement

Develop an end-to-end NLP system that reads automobile service complaints and automatically:

* Extracts vehicle and service information
* Identifies the affected component
* Classifies the complaint
* Predicts customer sentiment
* Determines urgency
* Detects safety risks
* Identifies duplicate complaints
* Routes complaints to the appropriate department
* Recommends the next business action

This case study combines:

```text
Tokenization
Lemmatization
NER
TF-IDF
Word embeddings
BERT
LSTM or GRU
Attention
Text classification
Sentiment analysis
Semantic similarity
Business-rule integration
```


# ============================================================
# AUTOMOBILE SERVICE COMPLAINT INTELLIGENCE SYSTEM
# ============================================================

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import spacy

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from transformers import pipeline


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
DUPLICATE_THRESHOLD = 0.65

OUTPUT_FOLDER = Path("outputs")
ARTIFACT_FOLDER = Path("artifacts")

OUTPUT_FOLDER.mkdir(exist_ok=True)
ARTIFACT_FOLDER.mkdir(exist_ok=True)


# ============================================================
# 2. SYNTHETIC CATEGORY-TRAINING DATA
# ============================================================

training_data = [
    # --------------------------------------------------------
    # Engine Issue
    # --------------------------------------------------------
    (
        "The engine loses power while driving on the highway.",
        "Engine Issue"
    ),
    (
        "The engine warning light appeared and the car stopped.",
        "Engine Issue"
    ),
    (
        "There is strong engine vibration while the vehicle is idle.",
        "Engine Issue"
    ),
    (
        "The engine overheats after driving for twenty minutes.",
        "Engine Issue"
    ),

    # --------------------------------------------------------
    # Brake System Issue
    # --------------------------------------------------------
    (
        "The brake pedal became hard and the vehicle did not stop.",
        "Brake System Issue"
    ),
    (
        "The brakes stopped responding while driving.",
        "Brake System Issue"
    ),
    (
        "The braking distance has suddenly increased.",
        "Brake System Issue"
    ),
    (
        "The brake warning light is on and the pedal feels soft.",
        "Brake System Issue"
    ),

    # --------------------------------------------------------
    # Battery or EV Issue
    # --------------------------------------------------------
    (
        "The electric vehicle battery range has reduced significantly.",
        "Battery or EV Issue"
    ),
    (
        "The car battery drains completely overnight.",
        "Battery or EV Issue"
    ),
    (
        "The vehicle does not start because the battery is weak.",
        "Battery or EV Issue"
    ),
    (
        "Charging stops after a few minutes at every charging station.",
        "Battery or EV Issue"
    ),

    # --------------------------------------------------------
    # Service Delay
    # --------------------------------------------------------
    (
        "The service centre has not delivered my vehicle on time.",
        "Service Delay"
    ),
    (
        "My car has been in the workshop for fifteen days.",
        "Service Delay"
    ),
    (
        "The dealer keeps postponing the promised delivery date.",
        "Service Delay"
    ),
    (
        "The vehicle is ready but the service centre is not releasing it.",
        "Service Delay"
    ),

    # --------------------------------------------------------
    # Billing or Warranty Issue
    # --------------------------------------------------------
    (
        "The service centre charged an incorrect amount.",
        "Billing or Warranty Issue"
    ),
    (
        "My warranty claim was rejected without an explanation.",
        "Billing or Warranty Issue"
    ),
    (
        "I was charged for a component covered under warranty.",
        "Billing or Warranty Issue"
    ),
    (
        "The final service bill contains additional labour charges.",
        "Billing or Warranty Issue"
    ),

    # --------------------------------------------------------
    # Infotainment Issue
    # --------------------------------------------------------
    (
        "The infotainment screen keeps restarting.",
        "Infotainment Issue"
    ),
    (
        "The navigation system does not display the correct location.",
        "Infotainment Issue"
    ),
    (
        "Bluetooth disconnects every few minutes.",
        "Infotainment Issue"
    ),
    (
        "The touchscreen stopped responding after the software update.",
        "Infotainment Issue"
    ),

    # --------------------------------------------------------
    # Body Damage
    # --------------------------------------------------------
    (
        "The service centre scratched the left door.",
        "Body Damage"
    ),
    (
        "I found a dent on the bumper after servicing.",
        "Body Damage"
    ),
    (
        "The vehicle paint was damaged inside the workshop.",
        "Body Damage"
    ),
    (
        "The side mirror was broken when the car was returned.",
        "Body Damage"
    ),

    # --------------------------------------------------------
    # Staff Behaviour
    # --------------------------------------------------------
    (
        "The service advisor was rude and unhelpful.",
        "Staff Behaviour"
    ),
    (
        "The technician refused to explain the repair.",
        "Staff Behaviour"
    ),
    (
        "The workshop manager behaved badly with the customer.",
        "Staff Behaviour"
    ),
    (
        "The dealer staff ignored my repeated calls.",
        "Staff Behaviour"
    )
]

training_df = pd.DataFrame(
    training_data,
    columns=[
        "complaint_text",
        "category"
    ]
)


# ============================================================
# 3. COMPLAINTS TO BE ANALYSED
# ============================================================

complaints = [
    {
        "complaint_id": "V001",
        "complaint_text": (
            "My Tata Nexon suddenly lost power while driving on the "
            "Mumbai-Pune Expressway. The engine warning light appeared, "
            "and the vehicle stopped near Lonavala."
        )
    },
    {
        "complaint_id": "V002",
        "complaint_text": (
            "I submitted my Hyundai Creta at Sunrise Motors Pune on "
            "15 June 2026, but the vehicle has still not been delivered. "
            "Service ID SER-98754."
        )
    },
    {
        "complaint_id": "V003",
        "complaint_text": (
            "The brakes of my Mahindra XUV700 stopped responding for a "
            "few seconds while driving at 80 km/h. "
            "This is extremely dangerous."
        )
    },
    {
        "complaint_id": "V004",
        "complaint_text": (
            "The service centre charged ₹18,500 for replacing the "
            "battery, although the vehicle is still under warranty."
        )
    },
    {
        "complaint_id": "V005",
        "complaint_text": (
            "My vehicle MH12AB4587 has visited the workshop three times "
            "for the same engine-vibration problem, but the issue "
            "remains unresolved."
        )
    },
    {
        "complaint_id": "V006",
        "complaint_text": (
            "The technician was rude and refused to explain why my "
            "warranty claim was rejected."
        )
    },
    {
        "complaint_id": "V007",
        "complaint_text": (
            "After software update v4.2.1, the infotainment system keeps "
            "restarting every five minutes."
        )
    },
    {
        "complaint_id": "V008",
        "complaint_text": (
            "I received my car after servicing, but the left door was "
            "scratched and the fuel level had reduced significantly."
        )
    },
    {
        "complaint_id": "V009",
        "complaint_text": (
            "The electric vehicle battery range has dropped from "
            "350 km to 210 km within six months."
        )
    },
    {
        "complaint_id": "V010",
        "complaint_text": (
            "Smoke started coming from the engine compartment immediately "
            "after the scheduled service. Please arrange emergency assistance."
        )
    },
    {
        "complaint_id": "V011",
        "complaint_text": (
            "The engine shakes strongly whenever my vehicle remains idle. "
            "The workshop has already inspected it three times."
        )
    }
]

complaint_df = pd.DataFrame(complaints)


# ============================================================
# 4. TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Standardizes punctuation and whitespace without removing
    important entities or changing the original meaning.
    """

    text = str(text)

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')

    text = re.sub(r"\s+", " ", text)

    return text.strip()


training_df["normalized_text"] = (
    training_df["complaint_text"]
    .apply(normalize_text)
)

complaint_df["normalized_text"] = (
    complaint_df["complaint_text"]
    .apply(normalize_text)
)


# ============================================================
# 5. BUILD TF-IDF CATEGORY CLASSIFIER
# ============================================================

X = training_df["normalized_text"]
y = training_df["category"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)


category_model = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]
)


category_model.fit(
    X_train,
    y_train
)


test_predictions = category_model.predict(X_test)

print("=" * 70)
print("CATEGORY CLASSIFIER EVALUATION")
print("=" * 70)

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)


# Retrain using the complete demonstration dataset
category_model.fit(X, y)


# Save the trained classification pipeline
joblib.dump(
    category_model,
    ARTIFACT_FOLDER / "complaint_category_model.joblib"
)


# ============================================================
# 6. LOAD SPACY NLP PIPELINE
# ============================================================

nlp = spacy.load("en_core_web_sm")


# Add custom automobile entities
entity_ruler = nlp.add_pipe(
    "entity_ruler",
    after="ner",
    config={
        "overwrite_ents": True
    }
)


vehicle_brands = [
    "Tata",
    "Mahindra",
    "Hyundai",
    "Maruti Suzuki",
    "Toyota",
    "Honda",
    "Kia",
    "Volkswagen",
    "Skoda",
    "MG",
    "Renault",
    "Nissan"
]


vehicle_models = [
    "Nexon",
    "XUV700",
    "Creta",
    "Seltos",
    "Harrier",
    "Safari",
    "Venue",
    "Thar",
    "Scorpio",
    "Hector",
    "Virtus",
    "Kushaq"
]


vehicle_components = [
    "engine",
    "brake",
    "brakes",
    "battery",
    "gearbox",
    "transmission",
    "infotainment system",
    "touchscreen",
    "door",
    "bumper",
    "fuel system",
    "steering",
    "airbag",
    "suspension"
]


dealers = [
    "Sunrise Motors Pune"
]


patterns = []


for brand in vehicle_brands:
    patterns.append({
        "label": "VEHICLE_BRAND",
        "pattern": brand
    })


for model in vehicle_models:
    patterns.append({
        "label": "VEHICLE_MODEL",
        "pattern": model
    })


for component in vehicle_components:
    patterns.append({
        "label": "VEHICLE_COMPONENT",
        "pattern": component
    })


for dealer in dealers:
    patterns.append({
        "label": "DEALER",
        "pattern": dealer
    })


entity_ruler.add_patterns(patterns)


# ============================================================
# 7. REGEX PATTERNS FOR DOMAIN ENTITIES
# ============================================================

REGEX_PATTERNS = {
    "REGISTRATION_NUMBER": re.compile(
        r"\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}\b",
        re.IGNORECASE
    ),

    "SERVICE_ID": re.compile(
        r"\bSER-\d+\b",
        re.IGNORECASE
    ),

    "DIAGNOSTIC_CODE": re.compile(
        r"\b(?:DTC-?)?[A-Z]\d{4}\b",
        re.IGNORECASE
    ),

    "SOFTWARE_VERSION": re.compile(
        r"\bv\d+(?:\.\d+)+\b",
        re.IGNORECASE
    ),

    "MONEY": re.compile(
        r"₹\s?\d+(?:,\d{3})*(?:\.\d+)?"
    ),

    "SPEED": re.compile(
        r"\b\d{1,3}\s*km/?h\b",
        re.IGNORECASE
    ),

    "DISTANCE_OR_RANGE": re.compile(
        r"\b\d+(?:,\d{3})?\s*km\b",
        re.IGNORECASE
    )
}


# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================

def unique_values(values):
    """
    Removes duplicates while retaining the original order.
    """

    return list(dict.fromkeys(values))


def extract_entities(text, doc):
    """
    Combines spaCy named entities with automobile-specific
    regular-expression entities.
    """

    entities = {}


    # General and custom spaCy entities
    for entity in doc.ents:

        label = entity.label_
        value = entity.text.strip()

        entities.setdefault(label, [])
        entities[label].append(value)


    # Domain-specific regular expressions
    for label, pattern in REGEX_PATTERNS.items():

        matches = pattern.findall(text)

        if matches:
            entities.setdefault(label, [])
            entities[label].extend(matches)


    # Remove repeated values
    for label in entities:
        entities[label] = unique_values(
            entities[label]
        )


    return entities


def create_linguistic_analysis(doc):
    """
    Creates token, lemma and POS details for one complaint.
    """

    analysis = []

    for token in doc:

        if token.is_space:
            continue

        analysis.append({
            "token": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dependency": token.dep_,
            "is_stopword": token.is_stop,
            "is_punctuation": token.is_punct,
            "entity_type": token.ent_type_
        })

    return analysis


def extract_important_lemmas(doc):
    """
    Keeps content words and removes stop words and punctuation.
    """

    lemmas = []

    for token in doc:

        if token.is_space:
            continue

        if token.is_stop:
            continue

        if token.is_punct:
            continue

        lemma = token.lemma_.lower().strip()

        if lemma:
            lemmas.append(lemma)

    return unique_values(lemmas)


def determine_urgency(text, category):
    """
    Assigns urgency using safety expressions and category rules.
    """

    normalized = text.lower()


    critical_phrases = [
        "brakes stopped responding",
        "brake failure",
        "brakes failed",
        "smoke started",
        "caught fire",
        "vehicle caught fire",
        "airbag did not deploy",
        "accident",
        "extremely dangerous",
        "emergency assistance",
        "lost control"
    ]


    high_phrases = [
        "lost power",
        "vehicle stopped",
        "engine stopped",
        "overheating",
        "same problem",
        "three times",
        "repeated breakdown",
        "remains unresolved",
        "battery overheating",
        "warning light"
    ]


    if any(
        phrase in normalized
        for phrase in critical_phrases
    ):
        return "Critical"


    if category == "Brake System Issue":
        return "High"


    if any(
        phrase in normalized
        for phrase in high_phrases
    ):
        return "High"


    if category in {
        "Engine Issue",
        "Battery or EV Issue"
    }:
        return "High"


    if category in {
        "Service Delay",
        "Billing or Warranty Issue",
        "Body Damage",
        "Staff Behaviour"
    }:
        return "Medium"


    return "Low"


def detect_safety_risk(text, category, urgency):
    """
    Flags potentially dangerous complaints.
    """

    safety_words = [
        "brake",
        "brakes",
        "smoke",
        "fire",
        "dangerous",
        "accident",
        "overheating",
        "lost control",
        "airbag",
        "emergency"
    ]

    normalized = text.lower()

    return (
        urgency == "Critical"
        or (
            category in {
                "Brake System Issue",
                "Engine Issue"
            }
            and any(
                word in normalized
                for word in safety_words
            )
        )
    )


CATEGORY_TO_ROOT_CAUSE = {
    "Engine Issue": "Powertrain or engine management",
    "Brake System Issue": "Braking system",
    "Battery or EV Issue": "Battery, charging or electrical system",
    "Service Delay": "Dealer or workshop operations",
    "Billing or Warranty Issue": "Billing or warranty administration",
    "Infotainment Issue": "Infotainment hardware or software",
    "Body Damage": "Workshop handling or body repair",
    "Staff Behaviour": "Customer service or employee conduct"
}


CATEGORY_TO_DEPARTMENT = {
    "Engine Issue": "Powertrain Engineering",
    "Brake System Issue": "Vehicle Safety Team",
    "Battery or EV Issue": "Electrical and EV Systems",
    "Service Delay": "Dealer Operations",
    "Billing or Warranty Issue": "Warranty and Accounts",
    "Infotainment Issue": "Connected Car and Infotainment",
    "Body Damage": "Workshop Quality Team",
    "Staff Behaviour": "Customer Experience Team"
}


def recommend_action(category, urgency):
    """
    Generates a controlled business recommendation.
    """

    if urgency == "Critical":
        return (
            "Contact the customer immediately, advise against driving "
            "the vehicle and arrange emergency inspection or towing."
        )


    recommendations = {
        "Engine Issue": (
            "Schedule priority diagnostics and inspect engine fault codes."
        ),

        "Brake System Issue": (
            "Arrange an immediate brake-system inspection and road-safety check."
        ),

        "Battery or EV Issue": (
            "Perform battery-health, charging and electrical diagnostics."
        ),

        "Service Delay": (
            "Contact the dealer and provide a confirmed delivery commitment."
        ),

        "Billing or Warranty Issue": (
            "Audit the invoice and verify warranty eligibility."
        ),

        "Infotainment Issue": (
            "Check software version, logs and available firmware updates."
        ),

        "Body Damage": (
            "Inspect the vehicle, review workshop records and arrange repair."
        ),

        "Staff Behaviour": (
            "Escalate the interaction to the customer-experience manager."
        )
    }


    return recommendations.get(
        category,
        "Assign the complaint for manual review."
    )


# ============================================================
# 9. PROCESS COMPLAINTS WITH SPACY
# ============================================================

texts = complaint_df["normalized_text"].tolist()


# nlp.pipe processes the complaints as a batch
processed_docs = list(
    nlp.pipe(
        texts,
        batch_size=16
    )
)


# ============================================================
# 10. CATEGORY PREDICTIONS
# ============================================================

category_predictions = category_model.predict(texts)

category_probabilities = (
    category_model.predict_proba(texts)
)

category_confidence = (
    category_probabilities.max(axis=1)
)


# ============================================================
# 11. TRANSFORMER SENTIMENT ANALYSIS
# ============================================================

sentiment_analyzer = pipeline(
    task="sentiment-analysis",
    model=(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
)


raw_sentiments = sentiment_analyzer(
    texts,
    truncation=True,
    batch_size=8
)


strong_negative_phrases = [
    "extremely dangerous",
    "smoke",
    "fire",
    "stopped responding",
    "remains unresolved",
    "emergency",
    "rejected",
    "rude"
]


def convert_sentiment(text, transformer_result):
    """
    Converts binary transformer sentiment into business labels.
    """

    label = transformer_result["label"].upper()
    score = float(transformer_result["score"])

    normalized = text.lower()


    if "NEGATIVE" in label:

        has_strong_negative_phrase = any(
            phrase in normalized
            for phrase in strong_negative_phrases
        )

        if has_strong_negative_phrase or score >= 0.98:
            return "Highly Negative"

        return "Negative"


    if score >= 0.80:
        return "Positive"

    return "Neutral"


# ============================================================
# 12. SEMANTIC DUPLICATE DETECTION
# ============================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


embeddings = embedding_model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=False
)


similarity_matrix = cosine_similarity(
    embeddings
)


# Prevent each record from matching itself
np.fill_diagonal(
    similarity_matrix,
    -1
)


duplicate_ids = []
duplicate_scores = []


for row_index in range(len(complaint_df)):

    best_match_index = int(
        similarity_matrix[row_index].argmax()
    )

    best_score = float(
        similarity_matrix[
            row_index,
            best_match_index
        ]
    )


    if best_score >= DUPLICATE_THRESHOLD:

        duplicate_ids.append(
            complaint_df.iloc[
                best_match_index
            ]["complaint_id"]
        )

    else:
        duplicate_ids.append(None)


    duplicate_scores.append(
        round(best_score, 4)
    )


# ============================================================
# 13. BUILD FINAL REPORT
# ============================================================

report_rows = []
token_rows = []


for index, complaint in complaint_df.iterrows():

    complaint_id = complaint["complaint_id"]
    original_text = complaint["complaint_text"]
    normalized_text = complaint["normalized_text"]

    doc = processed_docs[index]

    category = category_predictions[index]
    confidence = float(
        category_confidence[index]
    )

    sentiment = convert_sentiment(
        normalized_text,
        raw_sentiments[index]
    )

    urgency = determine_urgency(
        normalized_text,
        category
    )

    safety_risk = detect_safety_risk(
        normalized_text,
        category,
        urgency
    )

    entities = extract_entities(
        normalized_text,
        doc
    )

    important_lemmas = extract_important_lemmas(
        doc
    )

    sentences = [
        sentence.text.strip()
        for sentence in doc.sents
    ]

    department = CATEGORY_TO_DEPARTMENT.get(
        category,
        "Manual Review Team"
    )

    root_cause = CATEGORY_TO_ROOT_CAUSE.get(
        category,
        "Requires investigation"
    )

    action = recommend_action(
        category,
        urgency
    )


    report_rows.append({
        "complaint_id": complaint_id,
        "original_complaint": original_text,
        "sentence_count": len(sentences),
        "token_count": len([
            token
            for token in doc
            if not token.is_space
        ]),
        "sentences": json.dumps(
            sentences,
            ensure_ascii=False
        ),
        "important_lemmas": json.dumps(
            important_lemmas,
            ensure_ascii=False
        ),
        "entities": json.dumps(
            entities,
            ensure_ascii=False
        ),
        "predicted_category": category,
        "category_confidence": round(
            confidence,
            4
        ),
        "sentiment": sentiment,
        "urgency": urgency,
        "safety_risk": safety_risk,
        "root_cause_area": root_cause,
        "assigned_department": department,
        "recommended_action": action,
        "possible_duplicate_of": duplicate_ids[index],
        "duplicate_similarity": duplicate_scores[index]
    })


    # Detailed token-level report
    linguistic_analysis = create_linguistic_analysis(
        doc
    )

    for token_record in linguistic_analysis:

        token_rows.append({
            "complaint_id": complaint_id,
            **token_record
        })


report_df = pd.DataFrame(report_rows)

token_analysis_df = pd.DataFrame(token_rows)


# ============================================================
# 14. BUILD DUPLICATE-PAIR REPORT
# ============================================================

duplicate_pairs = []


for first_index in range(len(complaint_df)):

    for second_index in range(
        first_index + 1,
        len(complaint_df)
    ):

        score = float(
            similarity_matrix[
                first_index,
                second_index
            ]
        )

        if score >= DUPLICATE_THRESHOLD:

            duplicate_pairs.append({
                "complaint_id_1": (
                    complaint_df.iloc[
                        first_index
                    ]["complaint_id"]
                ),
                "complaint_id_2": (
                    complaint_df.iloc[
                        second_index
                    ]["complaint_id"]
                ),
                "similarity_score": round(
                    score,
                    4
                ),
                "possible_duplicate": True
            })


duplicate_df = pd.DataFrame(
    duplicate_pairs
)


# ============================================================
# 15. SAVE OUTPUT FILES
# ============================================================

report_path = (
    OUTPUT_FOLDER /
    "automobile_complaint_nlp_report.csv"
)

token_path = (
    OUTPUT_FOLDER /
    "automobile_token_analysis.csv"
)

duplicate_path = (
    OUTPUT_FOLDER /
    "duplicate_complaint_report.csv"
)


report_df.to_csv(
    report_path,
    index=False,
    encoding="utf-8-sig"
)

token_analysis_df.to_csv(
    token_path,
    index=False,
    encoding="utf-8-sig"
)

duplicate_df.to_csv(
    duplicate_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. DISPLAY IMPORTANT RESULTS
# ============================================================

display_columns = [
    "complaint_id",
    "predicted_category",
    "category_confidence",
    "sentiment",
    "urgency",
    "safety_risk",
    "assigned_department",
    "possible_duplicate_of",
    "duplicate_similarity"
]


print("\n" + "=" * 70)
print("FINAL NLP COMPLAINT REPORT")
print("=" * 70)

print(
    report_df[
        display_columns
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("CRITICAL COMPLAINTS")
print("=" * 70)

critical_complaints = report_df[
    report_df["urgency"] == "Critical"
]

if critical_complaints.empty:
    print("No critical complaints found.")
else:
    print(
        critical_complaints[
            [
                "complaint_id",
                "original_complaint",
                "predicted_category",
                "recommended_action"
            ]
        ].to_string(index=False)
    )


print("\n" + "=" * 70)
print("POSSIBLE DUPLICATES")
print("=" * 70)

if duplicate_df.empty:
    print("No duplicate pairs crossed the threshold.")
else:
    print(
        duplicate_df.to_string(
            index=False
        )
    )


print("\nFiles created:")

print(report_path)
print(token_path)
print(duplicate_path)

print(
    ARTIFACT_FOLDER /
    "complaint_category_model.joblib"
)
