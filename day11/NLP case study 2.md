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
