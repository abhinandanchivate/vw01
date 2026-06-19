# Complex Naïve Bayes Case Study

## Intelligent Automobile Customer-Complaint Classification and Escalation

## 1. Clear Problem Statement

A large automobile manufacturer receives thousands of customer complaints every day through:

* Emails
* Mobile applications
* Service-centre notes
* Website forms
* Call-centre transcripts
* Social-media messages

Customers describe problems using unstructured text, such as:

> “The engine temperature warning keeps appearing and smoke is coming from the bonnet.”

> “I was charged twice for the same service.”

> “The service centre has not delivered my vehicle even after ten days.”

Currently, support executives manually read every complaint and assign it to a department. This creates several problems:

* Complaints are assigned to the wrong department.
* Safety-critical complaints may not be escalated immediately.
* Manual classification takes too much time.
* Different executives classify similar complaints differently.
* Fraud and billing issues may be mixed with normal service requests.
* Customers experience delayed responses.

The company wants to develop a **Naïve Bayes-based text classification system** that automatically classifies each complaint into one of the following categories:

| Complaint category       | Responsible department    |
| ------------------------ | ------------------------- |
| `General Service`        | Service operations        |
| `Engine and Mechanical`  | Mechanical diagnostics    |
| `Electrical and Battery` | Electrical diagnostics    |
| `Billing and Payment`    | Finance department        |
| `Warranty Claim`         | Warranty department       |
| `Delivery Delay`         | Workshop operations       |
| `Safety Critical`        | Emergency escalation team |

The system must give special importance to the **Safety Critical** category because failing to identify a safety-related complaint could lead to accidents, legal action and damage to the company’s reputation.

---

# 2. Business Question

> Based on the customer’s complaint text and available complaint information, which department should handle the complaint, and does it require urgent escalation?

---

# 3. Machine-Learning Problem

This is a **supervised multi-class text-classification problem**.

The target variable is:

```text
complaint_category
```

Possible classes:

```text
General Service
Engine and Mechanical
Electrical and Battery
Billing and Payment
Warranty Claim
Delivery Delay
Safety Critical
```

---

# 4. Business Objectives

The system should help the organization:

* Automatically route complaints
* Reduce manual classification work
* Identify safety complaints immediately
* Reduce average response time
* Improve department-level workload planning
* Reduce incorrect complaint transfers
* Improve customer satisfaction
* Track common vehicle problems
* Identify emerging product-quality issues

---

# 5. Why Naïve Bayes Is Suitable

Naïve Bayes is particularly effective for text classification because complaint documents contain thousands of possible words.

For example, the following words may indicate different classes:

| Words or phrases                               | Likely category        |
| ---------------------------------------------- | ---------------------- |
| engine, overheating, smoke, vibration          | Engine and Mechanical  |
| battery, charging, wiring, headlight           | Electrical and Battery |
| invoice, charged, payment, refund              | Billing and Payment    |
| warranty, replacement, claim rejected          | Warranty Claim         |
| delivery, pending, delayed, promised date      | Delivery Delay         |
| fire, brake failure, accident, steering locked | Safety Critical        |

Naïve Bayes calculates the probability of every complaint category after observing the words in the complaint.

genui{"probability_statistics_learning_block":{"type_id":"BAYES_THEOREM"}}

For complaint classification:

* **A** represents a complaint category.
* **B** represents the words found in the complaint.
* The model calculates the probability of each category given those words.
* The class with the highest probability becomes the prediction.

---

# 6. Why It Is Called “Naïve”

The model assumes that features are conditionally independent within a class.

Suppose a complaint contains:

```text
brake
failure
highway
accident
```

Naïve Bayes treats the contribution of each word separately when calculating the probability of the `Safety Critical` category.

In real language, words are not completely independent. However, this simplified assumption often works very well for:

* Email classification
* Spam detection
* Sentiment analysis
* Document classification
* Complaint routing
* News classification

---

# 7. Prediction Unit

Each dataset row represents:

> One customer complaint submitted through one communication channel.

Example:

| complaint_id | complaint_text                                         | channel     | category               |
| ------------ | ------------------------------------------------------ | ----------- | ---------------------- |
| CMP1001      | Battery is not charging and the vehicle does not start | Mobile App  | Electrical and Battery |
| CMP1002      | Brake stopped responding while driving                 | Call Centre | Safety Critical        |
| CMP1003      | I was charged twice for the same repair                | Email       | Billing and Payment    |

---

# 8. Dataset Description

The organization has approximately **60,000 historical complaint records** collected over three years.

## Input Features

| Feature                   | Type        | Description                  |
| ------------------------- | ----------- | ---------------------------- |
| `complaint_id`            | Identifier  | Unique complaint number      |
| `customer_id`             | Identifier  | Unique customer number       |
| `vehicle_model`           | Categorical | Vehicle model                |
| `vehicle_age_years`       | Numerical   | Vehicle age                  |
| `mileage_km`              | Numerical   | Current mileage              |
| `complaint_text`          | Text        | Customer’s complaint         |
| `subject_line`            | Text        | Email or ticket subject      |
| `communication_channel`   | Categorical | Email, App, Web, Call Centre |
| `previous_complaints`     | Numerical   | Historical complaint count   |
| `days_since_last_service` | Numerical   | Days since last service      |
| `vehicle_under_warranty`  | Categorical | Yes or No                    |
| `customer_region`         | Categorical | Customer location            |
| `service_centre`          | Categorical | Associated centre            |
| `complaint_category`      | Target      | Assigned complaint class     |

---

# 9. Example Records

| Complaint text                                         | Channel     | Warranty | Target category        |
| ------------------------------------------------------ | ----------- | -------- | ---------------------- |
| Engine makes a loud knocking sound during acceleration | App         | No       | Engine and Mechanical  |
| Battery drains overnight and vehicle does not start    | Email       | Yes      | Electrical and Battery |
| Invoice includes parts that were never replaced        | Web         | No       | Billing and Payment    |
| Vehicle delivery has been pending for twelve days      | Call Centre | Yes      | Delivery Delay         |
| Brake pedal failed while driving on the highway        | Call Centre | Yes      | Safety Critical        |
| Warranty replacement request was rejected              | Email       | Yes      | Warranty Claim         |

---

# 10. Complexity of the Case Study

## 10.1 Unstructured Complaint Text

Customers may describe the same problem differently.

Examples:

```text
The car does not start.
Vehicle ignition is not responding.
Battery appears completely dead.
Nothing happens when I press the start button.
```

All four complaints may belong to the same category even though they use different words.

---

## 10.2 Spelling Errors

Complaint messages may contain mistakes such as:

```text
batery
engne
break fail
waranty
delivary
```

Text cleaning must handle common spelling variations where practical.

---

## 10.3 Multiple Languages

Customers may submit complaints in:

* English
* Hindi
* Marathi
* Mixed-language text
* Romanized regional language

Example:

```text
Car start nahi ho rahi and battery warning aa raha hai.
```

The initial model may focus on English and Romanized text, while unsupported languages can be routed to manual review.

---

## 10.4 Class Imbalance

Most complaints may concern general service, while safety complaints are relatively rare.

Example distribution:

| Category               | Records | Percentage |
| ---------------------- | ------: | ---------: |
| General Service        |  21,000 |        35% |
| Engine and Mechanical  |  12,000 |        20% |
| Electrical and Battery |   9,000 |        15% |
| Billing and Payment    |   7,200 |        12% |
| Warranty Claim         |   5,400 |         9% |
| Delivery Delay         |   4,200 |         7% |
| Safety Critical        |   1,200 |         2% |

A model could achieve high accuracy while missing safety-critical complaints. Therefore, class-specific recall is essential.

---

## 10.5 Similar Categories

Some complaints may overlap.

Example:

> “The battery failed, and the service centre rejected the replacement because the warranty expired.”

This complaint contains information related to:

* Electrical and Battery
* Warranty Claim

The business must define a priority rule. For example:

> Classify the complaint according to the primary customer request.

In this case, the complaint could be classified as `Warranty Claim`.

---

## 10.6 High-Dimensional Data

After converting complaint text into numerical features, the dataset may contain tens of thousands of word columns.

Example vocabulary:

```text
engine
battery
brake
refund
invoice
delivery
warranty
smoke
overheating
replacement
```

Naïve Bayes handles high-dimensional sparse text data efficiently.

---

## 10.7 Data Leakage

Certain fields may directly reveal the target.

Examples:

```text
assigned_department
final_resolution
escalation_team
closed_by_department
```

Such columns must not be used during training because they are available only after the complaint has already been classified.

---

# 11. Recommended Naïve Bayes Algorithm

For word-count or TF-IDF features, the recommended algorithm is:

```python
MultinomialNB
```

It is suitable for:

* Word frequencies
* Document counts
* TF-IDF values
* Sparse text matrices
* Multi-class classification

Other variants include:

| Algorithm       | Suitable data                           |
| --------------- | --------------------------------------- |
| `MultinomialNB` | Word counts and TF-IDF values           |
| `BernoulliNB`   | Word present or absent                  |
| `GaussianNB`    | Continuous numerical features           |
| `ComplementNB`  | Imbalanced text-classification problems |
| `CategoricalNB` | Discrete categorical features           |

Because the classes are imbalanced, the project should compare:

```python
MultinomialNB
```

and:

```python
ComplementNB
```

---

# 12. Text-Preprocessing Requirements

The complaint text should be processed using the following workflow:

```text
Raw complaint
      ↓
Convert to lowercase
      ↓
Remove URLs and special characters
      ↓
Handle blank complaints
      ↓
Combine subject and complaint body
      ↓
Convert text into TF-IDF features
      ↓
Train Naïve Bayes model
```

Example:

```text
Original complaint:
"URGENT!! My BRAKES failed on Highway-48."

Cleaned text:
"urgent my brakes failed on highway"
```

Excessive text cleaning should be avoided because words such as `not`, `no` and `never` may change meaning.

Example:

```text
Battery is working.
Battery is not working.
```

Removing `not` would incorrectly make both sentences appear similar.

---

# 13. Feature Extraction Using TF-IDF

Machine-learning models cannot directly understand text.

TF-IDF converts each complaint into a numerical representation.

Example complaints:

```text
Complaint 1: battery not charging
Complaint 2: engine overheating warning
Complaint 3: battery replacement required
```

Possible feature table:

| Complaint   | battery | charging | engine | overheating | replacement |
| ----------- | ------: | -------: | -----: | ----------: | ----------: |
| Complaint 1 |    0.62 |     0.78 |   0.00 |        0.00 |        0.00 |
| Complaint 2 |    0.00 |     0.00 |   0.57 |        0.69 |        0.00 |
| Complaint 3 |    0.61 |     0.00 |   0.00 |        0.00 |        0.79 |

Frequently occurring generic words receive lower importance, while useful category-specific words receive greater importance.

---

# 14. TF-IDF Configuration

A suitable starting configuration is:

```python
TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    max_features=30000,
    sublinear_tf=True
)
```

## Parameter Meaning

| Parameter              | Purpose                                                |
| ---------------------- | ------------------------------------------------------ |
| `lowercase=True`       | Converts text to lowercase                             |
| `stop_words="english"` | Removes common English words                           |
| `ngram_range=(1,2)`    | Uses individual words and two-word phrases             |
| `min_df=3`             | Ignores words appearing in fewer than three complaints |
| `max_df=0.95`          | Ignores extremely common words                         |
| `max_features=30000`   | Limits vocabulary size                                 |
| `sublinear_tf=True`    | Reduces the influence of repeated words                |

Two-word phrases are important because:

```text
brake failure
engine noise
payment refund
delivery delay
battery warning
```

may be more informative than individual words.

---

# 15. Model Pipeline

```text
Historical complaints
        ↓
Remove duplicate complaints
        ↓
Handle missing complaint text
        ↓
Combine subject and description
        ↓
Train-test split
        ↓
TF-IDF Vectorization
        ↓
Multinomial Naïve Bayes
        ↓
Probability prediction
        ↓
Complaint category
        ↓
Urgency and department assignment
```

---

# 16. Initial Model Configuration

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(
    alpha=1.0
)
```

The `alpha` parameter applies smoothing.

Without smoothing, a word that was never observed in one class could make the complete probability for that class become zero.

---

# 17. Laplace Smoothing

Suppose the word `fire` did not appear in any historical `Billing and Payment` complaint.

Without smoothing:

```text
P(fire | Billing and Payment) = 0
```

This zero probability may completely eliminate the billing class.

Smoothing adds a small value to word counts so that unseen words do not create zero-probability problems.

|       Alpha | Behaviour                                          |
| ----------: | -------------------------------------------------- |
| Small alpha | Model relies strongly on observed word frequencies |
|       `1.0` | Standard Laplace smoothing                         |
| Large alpha | Produces smoother, less extreme probabilities      |

---

# 18. Hyperparameter-Tuning Requirements

The team should tune both the vectorizer and the Naïve Bayes model.

```python
parameter_grid = {
    "tfidf__ngram_range": [
        (1, 1),
        (1, 2)
    ],
    "tfidf__min_df": [
        2,
        3,
        5
    ],
    "tfidf__max_features": [
        10000,
        20000,
        30000
    ],
    "model__alpha": [
        0.01,
        0.1,
        0.5,
        1.0,
        2.0
    ]
}
```

Recommended cross-validation metric:

```python
scoring="f1_macro"
```

Macro F1 gives equal importance to every complaint category.

---

# 19. Baseline and Comparison Models

The project should compare:

| Model                     | Purpose                                 |
| ------------------------- | --------------------------------------- |
| Majority-class classifier | Minimum baseline                        |
| Multinomial Naïve Bayes   | Primary text model                      |
| Complement Naïve Bayes    | Imbalanced text model                   |
| Logistic Regression       | Strong linear baseline                  |
| Linear SVM                | Alternative high-dimensional classifier |

The final model should not be selected only because it has the highest accuracy. Safety-class performance must also be considered.

---

# 20. Evaluation Metrics

The following metrics are required:

| Metric                 | Purpose                                                               |
| ---------------------- | --------------------------------------------------------------------- |
| Accuracy               | Overall correct classifications                                       |
| Precision              | Correctness of category predictions                                   |
| Recall                 | Ability to identify all complaints in a category                      |
| F1-score               | Balance of precision and recall                                       |
| Macro F1               | Equal importance to all classes                                       |
| Weighted F1            | Performance adjusted by class size                                    |
| Confusion matrix       | Exact category-level errors                                           |
| Safety-critical recall | Percentage of safety complaints detected                              |
| Top-2 accuracy         | Whether the correct category appears in the two highest probabilities |

---

# 21. Most Important Metric

The most important metric is:

```text
Recall for Safety Critical complaints
```

Suppose:

```text
Actual Safety Critical complaints = 200
Correctly identified = 184
Missed = 16
```

Then:

[
\text{Safety Recall}=\frac{184}{200}=0.92
]

The model identifies 92% of safety-critical complaints.

A missed safety complaint is more serious than incorrectly escalating a normal complaint.

---

# 22. Error-Cost Matrix

| Actual category | Predicted category    | Business impact                     | Cost           |
| --------------- | --------------------- | ----------------------------------- | -------------- |
| Safety Critical | General Service       | Dangerous delay                     | Extremely high |
| Safety Critical | Engine and Mechanical | Safety escalation may be missed     | High           |
| Billing         | Warranty              | Complaint transferred unnecessarily | Medium         |
| Delivery Delay  | General Service       | Delayed resolution                  | Medium         |
| General Service | Safety Critical       | Unnecessary escalation              | Low to medium  |

The primary error to reduce is:

```text
Safety Critical → Non-safety category
```

---

# 23. Example Confusion Matrix

| Actual / Predicted | General | Engine | Electrical | Billing | Warranty | Delivery | Safety |
| ------------------ | ------: | -----: | ---------: | ------: | -------: | -------: | -----: |
| General            |   3,760 |    110 |         80 |      60 |       50 |      120 |     20 |
| Engine             |     130 |  2,090 |         90 |      15 |       30 |       20 |     25 |
| Electrical         |      85 |    100 |      1,550 |      10 |       20 |       15 |     20 |
| Billing            |      55 |     10 |         10 |   1,270 |       75 |       15 |      5 |
| Warranty           |      70 |     20 |         25 |      90 |      840 |       30 |      5 |
| Delivery           |      90 |     15 |         10 |      25 |       35 |      655 |     10 |
| Safety             |       8 |     15 |          7 |       2 |        1 |        3 |    204 |

Important observation:

* 204 safety complaints were correctly identified.
* 36 safety complaints were assigned to another category.
* The eight Safety-to-General errors need urgent investigation.

---

# 24. Prediction Output

For every new complaint, the system should generate:

| Output field             | Description                        |
| ------------------------ | ---------------------------------- |
| `complaint_id`           | Complaint identifier               |
| `predicted_category`     | Highest-probability class          |
| `predicted_department`   | Department receiving the complaint |
| `prediction_confidence`  | Highest class probability          |
| `safety_probability`     | Probability of Safety Critical     |
| `manual_review_required` | Yes or No                          |
| `recommended_action`     | Business action                    |
| `prediction_timestamp`   | Processing time                    |

---

# 25. Example Prediction

## New complaint

```text
While driving at 90 km per hour, the steering became locked and
I narrowly avoided an accident.
```

Possible model output:

| Category              | Probability |
| --------------------- | ----------: |
| Safety Critical       |        0.91 |
| Engine and Mechanical |        0.06 |
| General Service       |        0.02 |
| Other categories      |        0.01 |

Prediction:

```text
Predicted category: Safety Critical
Confidence: 91%
```

Recommended action:

```text
Immediately escalate to the emergency response team.
Contact the customer within 10 minutes.
Advise the customer not to drive the vehicle.
Arrange towing support.
Notify the regional safety manager.
```

---

# 26. Business Decision Rules

| Condition                                 | Action                          |
| ----------------------------------------- | ------------------------------- |
| Safety probability ≥ 0.70                 | Immediate emergency escalation  |
| Safety probability between 0.40 and 0.70  | Safety-team manual review       |
| Highest category probability ≥ 0.75       | Automatically assign department |
| Highest probability between 0.50 and 0.75 | Assign and flag for review      |
| Highest probability below 0.50            | Send for manual classification  |
| Blank or extremely short complaint        | Request additional information  |

A complaint may be escalated even when Safety Critical is not the highest class.

Example:

```text
Engine and Mechanical probability = 0.45
Safety Critical probability = 0.42
```

Because the safety probability is significant, the complaint should still receive manual safety review.

---

# 27. Model Success Criteria

| Metric                                  | Minimum target |
| --------------------------------------- | -------------: |
| Overall accuracy                        |            85% |
| Macro F1-score                          |           0.80 |
| Weighted F1-score                       |           0.86 |
| Safety-critical recall                  |            92% |
| Safety-critical precision               |            75% |
| Top-2 accuracy                          |            94% |
| Automatic-routing rate                  |            75% |
| Incorrect department transfer reduction |            30% |

The most important acceptance requirement is:

```text
At least 92% of historical Safety Critical complaints
must be correctly identified.
```

---

# 28. Error-Analysis Requirements

The data-science team must examine:

* Safety complaints predicted as General Service
* Billing complaints predicted as Warranty
* Battery complaints predicted as Engine problems
* Complaints with low prediction confidence
* Complaints containing multiple issues
* Short complaints with fewer than five words
* Mixed-language complaints
* Complaints containing spelling mistakes
* Performance by communication channel
* Performance by vehicle model
* Performance by region

Example error table:

| Complaint             | Actual   | Predicted | Probable reason           |
| --------------------- | -------- | --------- | ------------------------- |
| Car stopped suddenly  | Safety   | General   | Too little information    |
| Replacement refused   | Warranty | General   | Missing warranty keywords |
| Amount deducted again | Billing  | General   | Uncommon billing wording  |

---

# 29. Explainability Requirement

Naïve Bayes can show words strongly associated with each category.

Example:

| Category        | Important words                                       |
| --------------- | ----------------------------------------------------- |
| Safety Critical | fire, brake failure, accident, smoke, steering locked |
| Billing         | invoice, refund, payment, charged, amount             |
| Electrical      | battery, charging, wiring, headlight, fuse            |
| Delivery        | delayed, pending, delivery, promised, workshop        |
| Warranty        | warranty, claim, replacement, coverage, rejected      |

This helps business teams understand why the model is producing certain predictions.

---

# 30. Project Tasks

## Task 1: Data Understanding

* Load the complaint dataset.
* Display shape and column information.
* Check class distribution.
* Identify missing complaint text.
* Check duplicate complaints.
* Examine average complaint length.
* Identify unsupported languages.

## Task 2: Exploratory Analysis

* Display the most common words.
* Compare complaint length by category.
* Generate category-wise word frequencies.
* Analyze complaints by communication channel.
* Study safety complaints separately.
* Detect class imbalance.

## Task 3: Data Cleaning

* Combine subject and complaint body.
* Convert text to lowercase.
* Remove URLs and unnecessary characters.
* Preserve meaningful negation words.
* Handle missing text.
* Remove exact duplicates.
* Remove target-leaking columns.

## Task 4: Dataset Splitting

* Separate training and testing data.
* Preserve category distribution using stratification.
* Ensure duplicate complaints do not occur in both sets.

## Task 5: Feature Extraction

Use TF-IDF with:

* Unigrams
* Bigrams
* Minimum document frequency
* Maximum vocabulary size
* Sublinear term frequency

## Task 6: Baseline Model

Train:

```python
MultinomialNB(alpha=1.0)
```

## Task 7: Hyperparameter Tuning

Tune:

* `alpha`
* `ngram_range`
* `min_df`
* `max_features`
* MultinomialNB versus ComplementNB

## Task 8: Evaluation

Calculate:

* Accuracy
* Precision
* Recall
* F1-score
* Macro F1
* Weighted F1
* Confusion matrix
* Safety Critical recall
* Top-2 accuracy

## Task 9: Error Analysis

Analyze:

* Missed safety complaints
* Low-confidence predictions
* Multi-topic complaints
* Spelling and language problems
* Frequently confused categories

## Task 10: New Complaint Prediction

For new complaints:

* Predict category
* Produce class probabilities
* Assign department
* Determine escalation level
* Flag uncertain complaints
* Save predictions to CSV

## Task 11: Save the Pipeline

Save the complete model containing:

```text
TF-IDF vectorizer
        +
Naïve Bayes classifier
```

using Joblib.

---

# 31. Expected Project Structure

```text
naive_bayes_complaint_project/
│
├── data/
│   ├── automobile_complaints.csv
│   └── new_complaints.csv
│
├── src/
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── error_analysis.py
│   └── predict_complaints.py
│
├── artifacts/
│   └── complaint_classifier_pipeline.joblib
│
├── outputs/
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   ├── model_comparison.csv
│   ├── missed_safety_complaints.csv
│   └── complaint_predictions.csv
│
├── reports/
│   └── naive_bayes_case_study_report.pdf
│
└── requirements.txt
```

---

# 32. Final Case Study Question

> Develop an end-to-end Naïve Bayes machine-learning solution that automatically classifies automobile customer complaints into General Service, Engine and Mechanical, Electrical and Battery, Billing and Payment, Warranty Claim, Delivery Delay and Safety Critical categories. The solution must process unstructured complaint text, handle class imbalance, use TF-IDF features, compare Multinomial Naïve Bayes and Complement Naïve Bayes, generate category probabilities, identify uncertain complaints and prioritize recall for Safety Critical complaints.

The complexity comes from:

* Unstructured text
* Thousands of text features
* Similar complaint categories
* Imbalanced classes
* Multi-topic complaints
* Mixed-language text
* Spelling variations
* High-cost safety-class errors
* Confidence-based routing
* Department and escalation recommendations
