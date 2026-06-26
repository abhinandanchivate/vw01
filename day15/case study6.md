# Case Study: Intelligent Banking Fraud Detection Using LLM Embeddings and Clustering

## 1. Business Background

A multinational bank operates across multiple countries and serves millions of retail and corporate customers. Every day, thousands of fraud-related support tickets are generated from various customer interaction channels.

These fraud reports originate from:

* Mobile Banking Application
* Internet Banking Portal
* ATM Network
* Credit Card Processing System
* Branch Offices
* Customer Care Center
* WhatsApp Banking
* Email Support
* SMS Fraud Reporting Portal

Each ticket is manually reviewed by fraud analysts before assigning it to an investigation team.

As the number of fraud reports continues to grow, manual classification has become time-consuming, inconsistent, and expensive.

---

# 2. Business Problem

The fraud investigation team receives nearly **25,000 fraud-related support tickets every week**.

Several operational challenges have emerged:

* Similar fraud incidents are assigned different categories by different analysts.
* Duplicate investigations are created for the same fraud pattern.
* New fraud trends are identified only after hundreds of customer complaints.
* Analysts spend significant time manually reviewing ticket descriptions.
* Existing rule-based categorization relies heavily on keywords and fails when customers describe the same fraud differently.

For example, the following complaints describe nearly identical incidents:

> Money debited through UPI without approval.

> Unauthorized UPI transaction detected.

> Someone transferred money using my UPI account.

> Unknown UPI payment happened at midnight.

Although these complaints represent the same fraud type, they are often classified differently because they use different wording.

---

# 3. Business Objective

Develop an AI-driven fraud intelligence system capable of automatically discovering hidden fraud patterns without predefined labels.

The system should:

* Group similar fraud complaints automatically.
* Detect emerging fraud trends.
* Identify duplicate fraud reports.
* Separate unusual or isolated fraud incidents.
* Help investigators prioritize high-risk fraud clusters.
* Generate dashboards for fraud operations teams.

---

# 4. Available Data

Each fraud ticket contains both structured and unstructured information.

| Attribute          | Example                               |
| ------------------ | ------------------------------------- |
| Ticket ID          | T001                                  |
| Product            | UPI                                   |
| Channel            | Mobile App                            |
| Priority           | High                                  |
| Transaction Amount | ₹45,000                               |
| Customer Complaint | Unauthorized UPI transfer at midnight |

The organization has accumulated over **3 million historical fraud cases** covering:

* UPI Fraud
* Credit Card Fraud
* ATM Fraud
* Internet Banking Fraud
* KYC Phishing
* Loan Scams
* Cheque Fraud
* Forex Fraud

---

# 5. Why Traditional Keyword Search Fails

Traditional approaches use:

* Keyword Matching
* Regular Expressions
* Rule-Based Classification
* Static Fraud Dictionaries

Example:

Complaint 1

> Unauthorized UPI transaction

Complaint 2

> Money transferred without approval

Complaint 3

> Someone used my UPI account

Although the wording differs, all describe the same fraud category.

Traditional algorithms treat them as unrelated because they compare words rather than meaning.

---

# 6. Proposed AI Solution

Instead of comparing keywords, every complaint is converted into a numerical embedding using a transformer-based language model.

Each complaint becomes a high-dimensional vector.

Example

Complaint

> Unauthorized UPI transaction

↓

Embedding Vector

```
[0.18, -0.44, 0.91, ..., 384 values]
```

Complaints with similar meanings are positioned close together in vector space.

---

# 7. Feature Engineering

Unlike traditional clustering, the solution combines multiple business features.

### Unstructured Features

* Customer complaint description

### Structured Features

* Banking Product
* Transaction Channel
* Priority
* Transaction Amount

The structured attributes are encoded and combined with the text embeddings to produce a richer feature representation for clustering.

---

# 8. Clustering Objectives

Three clustering algorithms are evaluated.

## K-Means

Business Question

> What are the major fraud categories in the banking system?

Expected Output

* UPI Fraud
* Credit Card Fraud
* ATM Fraud
* Internet Banking Fraud
* KYC Phishing
* Loan Scam

---

## DBSCAN

Business Question

> Which fraud cases are unusual or previously unseen?

Expected Output

* Automatically detects dense fraud patterns.
* Identifies rare fraud cases.
* Marks isolated complaints as outliers.

Example

```
Locker room light not working
```

This ticket is unrelated to fraud and should be detected as noise.

---

## Hierarchical Clustering

Business Question

> How are different fraud categories related?

Expected Hierarchy

```
Banking Fraud

│

├── Payment Fraud

│      ├── UPI

│      ├── Credit Card

│      └── ATM

│

├── Digital Banking Fraud

│      ├── Internet Banking

│      ├── Mobile Banking

│      └── Password Theft

│

├── Identity Fraud

│      ├── KYC

│      ├── Phishing

│      └── Fake Documents

│

└── Financial Scam

       ├── Loan Scam

       ├── Investment Scam

       └── Forex Fraud
```

---

# 9. Machine Learning Workflow

The fraud analysis pipeline consists of the following stages:

1. Load banking fraud tickets.
2. Generate sentence embeddings from complaint descriptions.
3. Normalize embedding vectors.
4. Encode structured business attributes.
5. Combine text embeddings with structured features.
6. Apply K-Means clustering.
7. Apply DBSCAN clustering.
8. Apply Hierarchical clustering.
9. Evaluate clustering quality using Silhouette Score.
10. Visualize clusters using PCA.

---

# 10. Expected Business Output

The clustering process groups similar fraud incidents.

Example

| Cluster   | Business Category                 |
| --------- | --------------------------------- |
| Cluster 0 | UPI Unauthorized Transactions     |
| Cluster 1 | Credit Card Fraud                 |
| Cluster 2 | Internet Banking Account Takeover |
| Cluster 3 | ATM Withdrawal Fraud              |
| Cluster 4 | KYC Phishing                      |
| Cluster 5 | Loan Scam                         |

DBSCAN additionally identifies rare fraud incidents that do not belong to any major group.

---

# 11. Business Benefits

Implementing this AI-based fraud clustering solution provides several advantages.

### Faster Investigation

Investigators review clusters instead of thousands of individual tickets.

### Duplicate Detection

Repeated fraud reports are automatically grouped.

### Fraud Trend Discovery

Emerging fraud campaigns become visible much earlier.

### Better Prioritization

High-volume fraud categories receive immediate attention.

### Reduced Manual Effort

Analysts spend less time categorizing complaints manually.

### Improved Decision Making

Fraud management teams obtain meaningful insights into customer-reported fraud patterns.

---

# 12. Learning Outcomes

By completing this case study, learners will understand:

* How transformer-based sentence embeddings capture semantic meaning.
* How structured and unstructured data can be combined into a single feature set.
* The differences between K-Means, DBSCAN, and Hierarchical Clustering.
* How high-dimensional vectors are clustered.
* How dimensionality reduction techniques such as PCA help visualize high-dimensional data.
* How clustering can solve real-world fraud detection problems without requiring labeled datasets.

---

# 13. Conclusion

This case study demonstrates a practical enterprise application of unsupervised machine learning using transformer embeddings and clustering algorithms.

Unlike traditional rule-based fraud detection systems, the proposed solution understands the semantic meaning of customer complaints, groups similar fraud reports automatically, identifies unusual fraud cases, and provides valuable business intelligence for fraud operations.

The same architecture can be extended to domains such as insurance claim analysis, healthcare incident management, telecom customer complaints, cybersecurity event analysis, and enterprise IT service management.
