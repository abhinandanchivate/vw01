# Case Study: NLP-Based Banking Complaint Analysis

## 1. Case Study Title

**Automatic Analysis of Customer Banking Complaints Using NLP**

## 2. Business Background

A private bank receives thousands of customer complaints every day through:

* Emails
* Mobile banking application
* Website complaint forms
* Chat support
* Call-centre transcripts
* Social-media messages

The complaints are related to failed transactions, ATM withdrawals, credit cards, loans, account access, unauthorized transactions and refund delays.

Currently, customer-support employees manually read each complaint and forward it to the appropriate department.

## 3. Business Problem

Manual complaint processing creates several challenges:

* Complaints take too long to reach the correct department.
* Important transaction details may be missed.
* High-risk fraud complaints may not be prioritized.
* Similar complaints may be assigned different categories.
* Customer names, transaction IDs, amounts and dates are manually extracted.
* Increasing complaint volume requires additional employees.

The bank wants to develop an NLP system that can understand customer complaints and extract useful information automatically.

## 4. Objective

The NLP system should process each complaint and perform:

* Sentence tokenization
* Word tokenization
* Lemmatization
* Part-of-speech tagging
* Stop-word identification
* Named Entity Recognition
* Complaint keyword detection
* Complaint-category identification
* Urgency detection

## 5. Sample Customer Complaints

| Complaint ID | Customer complaint                                                                                               |
| ------------ | ---------------------------------------------------------------------------------------------------------------- |
| B001         | ₹15,000 was debited from my account on 18 June 2026, but the ATM did not dispense cash.                          |
| B002         | I transferred ₹25,500 to Rajesh Patil using UPI, but the transaction failed and the money has not been refunded. |
| B003         | My credit card ending in 4582 was charged twice at Amazon yesterday.                                             |
| B004         | Someone made an unauthorized transaction of ₹8,999 from my account. Please block my card immediately.            |
| B005         | I applied for a home loan in Pune last month, but I have not received any update.                                |
| B006         | My internet-banking account has been locked after three unsuccessful login attempts.                             |
| B007         | Transaction ID TXN-984562 shows successful, but the receiver has not received the amount.                        |
| B008         | The EMI for my car loan was deducted two times this month.                                                       |
| B009         | I requested a refund from Flipkart seven days ago, but it is still not credited.                                 |
| B010         | Please update my mobile number from 9876543210 to 9988776655.                                                    |

## 6. NLP Processing Requirements

### 6.1 Sentence Tokenization

The system must divide a complaint into individual sentences.

Example:

```text
My card was charged twice. Please refund the additional amount.
```

Expected sentence separation:

```text
Sentence 1: My card was charged twice.
Sentence 2: Please refund the additional amount.
```

### 6.2 Word Tokenization

The system must divide each sentence into meaningful tokens.

Example:

```text
The ATM did not dispense ₹15,000.
```

Expected tokens:

```text
The
ATM
did
not
dispense
₹15,000
.
```

### 6.3 Lemmatization

Words must be converted into their meaningful base forms.

| Original word | Expected lemma |
| ------------- | -------------- |
| transferred   | transfer       |
| failed        | fail           |
| charged       | charge         |
| received      | receive        |
| deducted      | deduct         |
| transactions  | transaction    |

### 6.4 Part-of-Speech Tagging

The system should identify the grammatical role of every token.

Example:

```text
The ATM failed yesterday.
```

| Token     | Possible part of speech |
| --------- | ----------------------- |
| The       | Determiner              |
| ATM       | Noun                    |
| failed    | Verb                    |
| yesterday | Adverb                  |
| .         | Punctuation             |

### 6.5 Named Entity Recognition

The system should identify entities such as:

* Customer names
* Organizations
* Locations
* Dates
* Amounts
* Phone numbers
* Card numbers
* Transaction IDs

Example:

```text
I transferred ₹25,500 to Rajesh Patil on 18 June 2026.
```

Expected entities:

| Entity       | Entity type |
| ------------ | ----------- |
| ₹25,500      | Money       |
| Rajesh Patil | Person      |
| 18 June 2026 | Date        |

### 6.6 Business-Specific Entity Extraction

The standard NLP model may not identify all banking entities. The system should also detect:

| Entity type         | Example      |
| ------------------- | ------------ |
| Transaction ID      | TXN-984562   |
| Card digits         | 4582         |
| Account number      | ACC-567890   |
| UPI ID              | customer@upi |
| Mobile number       | 9876543210   |
| Loan application ID | LOAN-45982   |

## 7. Complaint Categories

The system should assign each complaint to one of the following categories:

| Category                  | Description                                          |
| ------------------------- | ---------------------------------------------------- |
| ATM Issue                 | Cash not dispensed, partial cash or ATM failure      |
| UPI Transaction           | Failed, pending or incorrect UPI transaction         |
| Credit Card Issue         | Duplicate charge, declined card or incorrect billing |
| Unauthorized Transaction  | Suspected fraudulent transaction                     |
| Loan Enquiry              | Home, car or personal-loan complaint                 |
| Account Access            | Login failure, locked account or password issue      |
| Refund Delay              | Refund initiated but not received                    |
| EMI Issue                 | Duplicate or incorrect EMI deduction                 |
| Customer Details Update   | Mobile number, address or email update               |
| General Banking Complaint | Complaint that does not match another category       |

## 8. Urgency Levels

Each complaint must be assigned an urgency level.

| Urgency  | Conditions                                                       |
| -------- | ---------------------------------------------------------------- |
| Critical | Unauthorized transaction, stolen card or suspected fraud         |
| High     | Large failed transaction, duplicate deduction or account blocked |
| Medium   | Refund delay, loan update or failed payment                      |
| Low      | Personal-information update or general enquiry                   |

## 9. Important Keywords

The NLP system should identify meaningful complaint words.

### ATM-related keywords

```text
ATM
cash
dispense
withdrawal
debited
```

### Fraud-related keywords

```text
unauthorized
fraud
unknown
stolen
block
immediately
```

### Refund-related keywords

```text
refund
credited
pending
reversed
not received
```

### Card-related keywords

```text
credit card
debit card
charged
blocked
declined
```

### Loan-related keywords

```text
loan
EMI
application
approval
interest
```

## 10. Task List

### Task 1: Load the Complaint Data

Create or load the complaint records into a structured dataset.

Required columns:

```text
complaint_id
customer_complaint
```

### Task 2: Apply an NLP Pipeline

Process each complaint using an NLP library such as spaCy.

The pipeline should support:

* Tokenization
* Sentence detection
* Lemmatization
* Part-of-speech tagging
* Named Entity Recognition

### Task 3: Generate Token Information

For every token, capture:

| Field          | Description                     |
| -------------- | ------------------------------- |
| Token          | Original word                   |
| Lemma          | Base form                       |
| Part of speech | Grammatical role                |
| Stop word      | Whether the token is common     |
| Punctuation    | Whether it is punctuation       |
| Entity type    | Entity category, when available |

### Task 4: Extract Named Entities

Identify:

* People
* Organizations
* Locations
* Dates
* Money values

### Task 5: Extract Banking Entities

Create additional rules to identify:

* Transaction IDs
* Card digits
* Account numbers
* UPI IDs
* Mobile numbers
* Loan application IDs

### Task 6: Detect Complaint Keywords

Identify words or phrases that indicate the main issue.

Example:

```text
debited
ATM
did not dispense
refund
unauthorized
charged twice
```

### Task 7: Assign Complaint Category

Use the extracted words, lemmas and entities to determine the complaint category.

### Task 8: Determine Urgency

Assign the complaint an urgency level using the complaint language and identified entities.

### Task 9: Prepare Structured Output

The final output should contain:

| Output field       | Description                                      |
| ------------------ | ------------------------------------------------ |
| Complaint ID       | Unique complaint number                          |
| Original complaint | Original customer message                        |
| Sentence count     | Number of sentences                              |
| Token count        | Number of meaningful tokens                      |
| Tokens             | Extracted tokens                                 |
| Lemmas             | Base forms                                       |
| Named entities     | Person, organization, location, date and money   |
| Banking entities   | Transaction IDs, card digits and account numbers |
| Keywords           | Important complaint terms                        |
| Complaint category | Assigned business category                       |
| Urgency            | Critical, high, medium or low                    |

## 11. Additional Test Complaints

```text
My debit card was used for a ₹12,499 transaction that I did not authorize.
```

```text
UPI transaction TXN-776542 is pending for more than 48 hours.
```

```text
The ATM deducted ₹10,000 from my account but gave me only ₹5,000.
```

```text
Please block my card ending in 7845 because I lost it yesterday.
```

```text
My home-loan application LOAN-45982 has been under review since 1 June 2026.
```

```text
I made a payment to abcstore@upi, but the merchant did not receive it.
```

```text
My account was charged three EMIs instead of one EMI this month.
```

```text
Please change my registered email address to customer@example.com.
```

## 12. Expected Business Workflow

```text
Customer complaint
        ↓
NLP text processing
        ↓
Tokenization and lemmatization
        ↓
Named Entity Recognition
        ↓
Banking entity extraction
        ↓
Keyword identification
        ↓
Complaint classification
        ↓
Urgency determination
        ↓
Routing to the correct department
```

## 13. Department Routing

| Complaint category       | Assigned department             |
| ------------------------ | ------------------------------- |
| ATM Issue                | ATM Operations Team             |
| UPI Transaction          | Digital Payments Team           |
| Credit Card Issue        | Card Services Team              |
| Unauthorized Transaction | Fraud Investigation Team        |
| Loan Enquiry             | Loan Processing Team            |
| Account Access           | Internet Banking Support        |
| Refund Delay             | Transaction Reconciliation Team |
| EMI Issue                | Loan Servicing Team             |
| Customer Details Update  | Customer Service Team           |

## 14. Expected Deliverables

The learner must prepare:

1. Banking complaint dataset
2. NLP token-analysis output
3. Sentence and word tokens
4. Lemmas and parts of speech
5. Named-entity report
6. Banking-specific entity report
7. Complaint keywords
8. Complaint-category output
9. Urgency-level output
10. Department-routing report

## 15. Evaluation Criteria

| Evaluation area           | Expected result                                       |
| ------------------------- | ----------------------------------------------------- |
| Tokenization              | Words and sentences are separated correctly           |
| Lemmatization             | Words are converted into correct base forms           |
| POS tagging               | Grammatical roles are identified                      |
| Entity extraction         | Names, amounts, locations and dates are detected      |
| Banking entity extraction | Transaction IDs and banking identifiers are preserved |
| Category identification   | Complaints are assigned to suitable categories        |
| Urgency detection         | High-risk complaints are prioritized                  |
| Department routing        | Complaints reach the correct team                     |
| Output quality            | Final output is structured and readable               |

## 16. Final Problem Statement

Develop an NLP-based banking complaint-analysis system that processes unstructured customer messages, extracts linguistic and banking information, identifies the complaint category, determines urgency and routes each complaint to the appropriate banking department.

The system must preserve important details such as transaction IDs, monetary amounts, dates, card digits, account numbers and customer contact information.
