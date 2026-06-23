# Case Study: Customer Sentiment and Complaint Classification Using LSTM, GRU and Attention

## 1. Case Study Title

**Automobile Service Feedback Classification and Sentiment Analysis**

## 2. Business Background

A large automobile company receives customer feedback from:

* Service-centre surveys
* Emails
* Mobile applications
* Call-centre transcripts
* Dealer websites
* Social-media messages

The company receives approximately **20,000 feedback messages every month**.

Customers may discuss multiple topics, including:

* Service quality
* Vehicle repair
* Billing
* Staff behaviour
* Delivery delays
* Warranty claims
* Safety issues

The organization currently depends on employees to read and classify each message manually.

---

## 3. Business Problem

Manual feedback analysis creates several challenges:

* Negative feedback is not identified quickly.
* Safety-related complaints may be overlooked.
* Messages are sometimes assigned to the wrong department.
* Long customer messages are difficult to analyse.
* Important words may appear at different positions in a sentence.
* The same issue may be expressed using different words.
* Simple keyword systems may misunderstand negation.

For example:

```text
The service was not good.
```

A keyword system may detect `good` and incorrectly mark the message as positive.

Another example:

```text
The initial service was slow, but the technician resolved the issue perfectly.
```

The message contains both negative and positive expressions. The model must understand the complete sequence.

---

## 4. Project Objective

Develop and compare three deep-learning models:

1. **LSTM**
2. **GRU**
3. **Bidirectional LSTM with Attention**

The models must perform two NLP tasks:

### Task A: Sentiment analysis

Classify each customer message as:

```text
Positive
Neutral
Negative
```

### Task B: Complaint classification

Classify each message into a business category:

```text
Service Quality
Repair Issue
Billing Issue
Staff Behaviour
Delivery Delay
Warranty Issue
Safety Issue
General Feedback
```

---

## 5. Sample Dataset

| Feedback ID | Customer message                                                               | Sentiment | Complaint category |
| ----------- | ------------------------------------------------------------------------------ | --------- | ------------------ |
| F001        | The technician explained the repair clearly and completed the service on time. | Positive  | Service Quality    |
| F002        | My vehicle was kept at the workshop for ten days without any update.           | Negative  | Delivery Delay     |
| F003        | The final bill included labour charges that were never discussed.              | Negative  | Billing Issue      |
| F004        | The service advisor was polite, but the engine problem was not resolved.       | Negative  | Repair Issue       |
| F005        | The waiting area was clean and the staff were helpful.                         | Positive  | Service Quality    |
| F006        | My warranty claim is still under review.                                       | Neutral   | Warranty Issue     |
| F007        | The brakes stopped responding one day after servicing.                         | Negative  | Safety Issue       |
| F008        | The technician was rude and refused to answer my questions.                    | Negative  | Staff Behaviour    |
| F009        | The battery was replaced successfully and the vehicle now starts normally.     | Positive  | Repair Issue       |
| F010        | The car was delivered today after the scheduled service.                       | Neutral   | Delivery Delay     |
| F011        | The bill was slightly higher, but the service quality was excellent.           | Positive  | Billing Issue      |
| F012        | The vehicle still vibrates even after three workshop visits.                   | Negative  | Repair Issue       |
| F013        | The service centre accepted my warranty claim without any difficulty.          | Positive  | Warranty Issue     |
| F014        | Smoke appeared from the engine compartment after the repair.                   | Negative  | Safety Issue       |
| F015        | I visited the service centre for a routine inspection.                         | Neutral   | General Feedback   |

---

## 6. Dataset Structure

The complete dataset should contain at least **10,000–20,000 records**.

| Column               | Description                   |
| -------------------- | ----------------------------- |
| `feedback_id`        | Unique feedback identifier    |
| `customer_message`   | Original customer feedback    |
| `sentiment`          | Positive, Neutral or Negative |
| `complaint_category` | Business complaint class      |
| `vehicle_model`      | Vehicle model                 |
| `service_centre`     | Service-centre name           |
| `feedback_date`      | Date of feedback              |
| `resolution_status`  | Open, In Progress or Resolved |

---

## 7. Sentiment Label Distribution

The dataset should be checked for class imbalance.

Example distribution:

| Sentiment | Number of records |
| --------- | ----------------: |
| Positive  |             5,500 |
| Neutral   |             2,500 |
| Negative  |             7,000 |
| **Total** |        **15,000** |

The learner must determine whether class-balancing techniques are required.

---

## 8. Complaint-Category Distribution

| Complaint category | Number of records |
| ------------------ | ----------------: |
| Service Quality    |             2,800 |
| Repair Issue       |             3,500 |
| Billing Issue      |             1,700 |
| Staff Behaviour    |             1,200 |
| Delivery Delay     |             1,800 |
| Warranty Issue     |             1,400 |
| Safety Issue       |               900 |
| General Feedback   |             1,700 |

Safety complaints may have fewer records but greater business importance.

---

# 9. NLP Processing Requirements

## 9.1 Text Cleaning

The system must handle:

* Uppercase and lowercase text
* Extra spaces
* Repeated punctuation
* HTML characters
* URLs
* Email addresses
* Emojis
* Spelling mistakes
* Abbreviations
* Mixed-language text

Example:

```text
SERVICE WAS TOO SLOW!!! 😡
```

Possible normalized form:

```text
service was too slow angry
```

---

## 9.2 Tokenization

Input:

```text
The service advisor was polite, but the engine issue was not resolved.
```

Expected tokens:

```text
The
service
advisor
was
polite
,
but
the
engine
issue
was
not
resolved
.
```

---

## 9.3 Vocabulary Creation

Every important word should receive a numeric token ID.

Example:

| Word       | Token ID |
| ---------- | -------: |
| service    |       15 |
| technician |       42 |
| engine     |       58 |
| delayed    |       91 |
| excellent  |      105 |
| refund     |      128 |

The learner must decide:

* Maximum vocabulary size
* Treatment of unknown words
* Whether stop words should be retained
* Whether punctuation should be removed

Negation words such as `not`, `never` and `without` should generally be preserved.

---

## 9.4 Sequence Conversion

Example message:

```text
The service was excellent.
```

Possible numeric sequence:

```text
[5, 15, 12, 105]
```

Different messages will have different lengths.

---

## 9.5 Padding and Truncation

Suppose the maximum sequence length is `100`.

A short message:

```text
[5, 15, 12, 105]
```

After padding:

```text
[5, 15, 12, 105, 0, 0, 0, ...]
```

A message longer than 100 tokens must be truncated.

The learner must decide whether to use:

```text
Pre-padding or post-padding
Pre-truncation or post-truncation
```

---

# 10. Model 1: LSTM

The first model should use:

```text
Input sequence
      ↓
Embedding layer
      ↓
LSTM layer
      ↓
Dropout
      ↓
Dense layer
      ↓
Prediction
```

## LSTM objective

The LSTM should learn how earlier words affect later words.

Example:

```text
The repair was not successful.
```

The model should remember `not` while processing `successful`.

## Suggested architecture

```text
Vocabulary input
      ↓
Embedding: 128 dimensions
      ↓
LSTM: 64 or 128 units
      ↓
Dropout
      ↓
Dense hidden layer
      ↓
Softmax output
```

---

# 11. Model 2: GRU

The second model should replace the LSTM layer with a GRU layer.

```text
Input sequence
      ↓
Embedding layer
      ↓
GRU layer
      ↓
Dropout
      ↓
Dense layer
      ↓
Prediction
```

The objective is to compare whether GRU can achieve similar performance with fewer parameters or faster training.

---

# 12. Model 3: Bidirectional LSTM with Attention

The third model should use:

```text
Input sequence
      ↓
Embedding
      ↓
Bidirectional LSTM
      ↓
Attention mechanism
      ↓
Context vector
      ↓
Dense layer
      ↓
Prediction
```

## Attention requirement

The attention mechanism should assign greater importance to relevant words.

Example:

```text
The technician was polite, but the repair was completely unsuccessful.
```

Possible attention focus:

| Word or phrase          | Expected importance |
| ----------------------- | ------------------: |
| technician              |                 Low |
| polite                  |              Medium |
| but                     |                High |
| repair                  |                High |
| completely unsuccessful |           Very high |

The actual weights must be learned by the model.

---

# 13. Multi-Output Requirement

The system should ideally generate two predictions from the same customer message.

```text
Customer message
       ↓
Shared embedding and sequence representation
       ↓
 ┌──────────────────┬─────────────────────┐
 ↓                  ↓
Sentiment output    Complaint-category output
```

Example input:

```text
The brakes failed after servicing and the staff did not respond.
```

Expected output:

```json
{
  "sentiment": "Negative",
  "complaint_category": "Safety Issue"
}
```

A more advanced system may assign multiple complaint categories:

```json
{
  "sentiment": "Negative",
  "complaint_categories": [
    "Safety Issue",
    "Staff Behaviour"
  ]
}
```

---

# 14. Complex Test Messages

## Test message 1: Negation

```text
The repair was not satisfactory.
```

Expected sentiment:

```text
Negative
```

---

## Test message 2: Contrast

```text
The staff was helpful, but the vehicle problem remained unresolved.
```

Expected output:

```text
Sentiment: Negative
Category: Repair Issue
```

---

## Test message 3: Mixed sentiment

```text
The service was delayed, although the final repair quality was excellent.
```

Possible output:

```text
Sentiment: Positive or Neutral
Categories: Delivery Delay and Repair Issue
```

---

## Test message 4: Safety complaint

```text
The brakes stopped working while I was driving at 80 km/h.
```

Expected output:

```text
Sentiment: Negative
Category: Safety Issue
Priority: Critical
```

---

## Test message 5: Neutral update

```text
My vehicle is currently undergoing a warranty inspection.
```

Expected output:

```text
Sentiment: Neutral
Category: Warranty Issue
```

---

## Test message 6: Sarcasm

```text
Excellent service—I only had to visit the workshop five times.
```

Expected sentiment:

```text
Negative
```

This is difficult because the message contains the positive word `excellent` but expresses dissatisfaction.

---

## Test message 7: Long-distance dependency

```text
Although the technician initially promised that the vehicle would be delivered on Monday and repeatedly assured me that all repairs were completed, the car was still not ready when I visited the service centre on Friday.
```

Expected output:

```text
Sentiment: Negative
Category: Delivery Delay
```

---

# 15. Practical Task List

## Task 1: Prepare the dataset

* Load customer-feedback records.
* Remove duplicate messages.
* Handle missing text.
* Check sentiment distribution.
* Check complaint-category distribution.

## Task 2: Clean the text

* Normalize case.
* Remove unnecessary symbols.
* Preserve negation words.
* Process emojis.
* Handle contractions.

## Task 3: Tokenize the messages

* Create word tokens.
* Build the vocabulary.
* Convert words into integer sequences.
* Identify out-of-vocabulary words.

## Task 4: Pad the sequences

* Select maximum sequence length.
* Apply padding.
* Apply truncation.
* Compare message-length distributions.

## Task 5: Build the LSTM model

* Configure the embedding layer.
* Add LSTM units.
* Add dropout.
* Configure sentiment output.
* Train and validate the model.

## Task 6: Build the GRU model

* Use the same dataset split.
* Use comparable embedding and hidden dimensions.
* Measure training time and accuracy.

## Task 7: Build the attention model

* Use a Bidirectional LSTM.
* Return outputs for all token positions.
* Apply attention.
* Generate a context vector.
* Perform classification.

## Task 8: Perform sentiment classification

Predict:

```text
Positive
Neutral
Negative
```

## Task 9: Perform complaint classification

Predict:

```text
Service Quality
Repair Issue
Billing Issue
Staff Behaviour
Delivery Delay
Warranty Issue
Safety Issue
General Feedback
```

## Task 10: Compare all models

Compare:

* Validation accuracy
* Test accuracy
* Precision
* Recall
* F1-score
* Training time
* Number of parameters
* Performance on long reviews
* Performance on negation
* Performance on mixed sentiment

## Task 11: Visualize attention weights

Display important words for selected messages.

Example:

```text
The brakes stopped responding after the service.
```

Possible visualization:

```text
The        0.02
brakes     0.31
stopped    0.22
responding 0.28
after      0.03
service    0.10
```

## Task 12: Analyse incorrect predictions

Identify errors caused by:

* Sarcasm
* Spelling mistakes
* Mixed languages
* Multiple complaint categories
* Very long messages
* Limited vocabulary
* Incorrect labels

---

# 16. Model Evaluation

## Sentiment-analysis metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

## Complaint-classification metrics

* Macro F1-score
* Weighted F1-score
* Per-category recall
* Confusion matrix

## Safety-specific metric

The most important measurement for `Safety Issue` is:

```text
Recall
```

A low recall means genuine safety complaints are being missed.

---

# 17. Expected Comparison Report

| Model                 |    Test accuracy |    Training time |       Parameters | Key observation                   |
| --------------------- | ---------------: | ---------------: | ---------------: | --------------------------------- |
| LSTM                  | To be calculated | To be calculated | To be calculated | Strong sequence memory            |
| GRU                   | To be calculated | To be calculated | To be calculated | Potentially faster and smaller    |
| BiLSTM with Attention | To be calculated | To be calculated | To be calculated | Better focus on important phrases |

The learner should not assume that the attention model will always be the best. The result must be determined experimentally.

---

# 18. Expected Business Output

| Feedback ID | Sentiment | Category        | Confidence | Priority | Assigned team       |
| ----------- | --------- | --------------- | ---------: | -------- | ------------------- |
| F001        | Positive  | Service Quality |       0.94 | Low      | Customer Experience |
| F002        | Negative  | Delivery Delay  |       0.91 | Medium   | Dealer Operations   |
| F003        | Negative  | Billing Issue   |       0.89 | Medium   | Accounts            |
| F007        | Negative  | Safety Issue    |       0.98 | Critical | Vehicle Safety      |
| F013        | Positive  | Warranty Issue  |       0.93 | Low      | Warranty Team       |

---

# 19. Expected Deliverables

The learner must submit:

1. Customer-feedback dataset
2. Text-cleaning report
3. Tokenized and padded data
4. LSTM model
5. GRU model
6. Bidirectional LSTM with attention model
7. Sentiment predictions
8. Complaint-category predictions
9. Model-comparison report
10. Confusion matrices
11. Attention-weight analysis
12. Incorrect-prediction analysis
13. Final business-routing report

---

# 20. Final Problem Statement

Develop a deep-learning NLP system that analyses automobile service feedback and predicts:

* Customer sentiment
* Complaint category
* Complaint priority
* Responsible department

Train and compare **LSTM, GRU and Bidirectional LSTM with Attention** using the same dataset and evaluation process.

The final system must correctly handle:

* Negation
* Long customer messages
* Mixed sentiment
* Multiple complaint topics
* Safety-related language
* Important word identification through attention

The business must be able to determine which model provides the best balance of accuracy, recall, interpretability and training cost.
