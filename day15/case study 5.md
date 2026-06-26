#  Case Study: Intelligent Root Cause Discovery from Multi-Source Enterprise IT Incidents Using LLM Embeddings

## 1. Background

A global technology company with over **120,000 employees** operates across **45 countries**. Every day, thousands of IT incidents are reported through different enterprise systems.

These incidents originate from multiple channels:

* ServiceNow
* Microsoft Teams Help Bot
* Outlook Email Support
* Jira Service Management
* Internal Chatbot
* Walk-in Service Desk
* Monitoring Alerts
* Infrastructure Logs

Each team categorizes incidents manually, resulting in inconsistent classifications and duplicate tickets.

For example, the following incidents may describe the same underlying issue:

```text
Unable to connect to VPN after password reset.

VPN authentication keeps failing.

Remote access stopped working after MFA enrollment.

Cannot establish VPN connection.

Cisco AnyConnect keeps disconnecting.
```

Although these incidents describe the same problem, they are often assigned different categories by different support engineers.

---

# 2. Business Problem

The organization faces several operational challenges:

* More than **18,000 incident tickets** are created every week.
* Multiple support engineers assign different categories for similar issues.
* Duplicate tickets increase incident resolution time.
* Root causes are identified only after hundreds of users report similar issues.
* Global IT leadership lacks visibility into emerging technology problems.
* Traditional keyword-based categorization fails because employees describe the same issue using different language.

As a result:

* Mean Time to Resolution (MTTR) increases.
* SLA violations become frequent.
* Engineering teams spend significant effort manually reviewing tickets.

---

# 3. Business Objective

Develop an AI-powered incident intelligence platform capable of automatically discovering hidden patterns in enterprise incident data.

The platform should:

* Group semantically similar incidents.
* Detect emerging enterprise-wide problems.
* Identify duplicate incidents.
* Separate unique issues from noisy or isolated tickets.
* Discover hierarchical relationships among incident categories.
* Provide business-friendly explanations for every discovered cluster.

---

# 4. Available Data

Each incident contains structured and unstructured information.

| Attribute            | Example                                           |
| -------------------- | ------------------------------------------------- |
| Incident ID          | INC00045218                                       |
| Region               | APAC                                              |
| Country              | India                                             |
| Business Unit        | Finance                                           |
| Department           | Payroll                                           |
| Priority             | High                                              |
| Device Type          | Windows Laptop                                    |
| Operating System     | Windows 11                                        |
| Application          | Outlook                                           |
| Assignment Group     | Messaging Team                                    |
| Incident Description | Outlook freezes whenever shared mailbox is opened |
| Resolution Notes     | Available after closure                           |

The organization has approximately:

* 2.5 million historical incidents
* 350 unique support applications
* 120 support teams
* Incidents written in English, French, German and Spanish

---

# 5. Existing Solution Limitations

The current categorization system relies on:

* Keyword matching
* Regular expressions
* Manual tagging
* Rule-based automation

Example:

```text
VPN authentication failed

VPN login failed

Remote access unavailable

Cisco AnyConnect disconnected

Unable to establish VPN tunnel
```

Because the wording differs, traditional NLP approaches classify them as unrelated issues.

The existing solution cannot understand semantic similarity.

---

# 6. Proposed AI Solution

The organization decides to leverage Large Language Models to understand the semantic meaning of each incident instead of relying only on keywords.

Each incident description is converted into a high-dimensional embedding vector.

Example:

```text
Unable to connect to VPN
```

↓

```text
[0.271,
-0.614,
0.933,
...
1536 dimensions]
```

Incidents with similar meanings are located close to one another in vector space, even when different terminology is used.

---

# 7. Clustering Objectives

Three different clustering techniques will be evaluated.

## K-Means

Business Question

> How many major enterprise incident categories exist?

Expected Outcome

* VPN
* Outlook
* SAP
* Teams
* Printer
* Battery
* Azure Virtual Desktop
* OneDrive

---

## DBSCAN

Business Question

> Which incidents are isolated, rare or anomalous?

Expected Outcome

* Detect uncommon incidents
* Ignore noisy tickets
* Discover naturally occurring clusters
* Identify one-off security incidents

---

## Hierarchical Clustering

Business Question

> How are enterprise incidents related to one another?

Expected Outcome

```text
Enterprise IT

│

├── Infrastructure

│      ├── VPN

│      ├── Network

│      └── DNS

│

├── Collaboration

│      ├── Outlook

│      ├── Teams

│      └── Exchange

│

├── Enterprise Applications

│      ├── SAP

│      ├── Oracle

│      └── Salesforce

│

└── Hardware

       ├── Laptop

       ├── Battery

       └── Printer
```

---

# 8. Role of the LLM

The LLM is **not** responsible for clustering directly.

Instead, it performs three key functions:

### Step 1 — Embedding Generation

Every incident description is converted into a semantic vector.

Example:

```text
VPN authentication failed
```

↓

Embedding

```text
[1536-dimensional vector]
```

---

### Step 2 — Semantic Understanding

Unlike TF-IDF or Bag-of-Words, embeddings capture meaning.

Example:

```text
VPN authentication failed

Unable to connect to corporate VPN

Remote access unavailable

Cisco VPN disconnected
```

These incidents have very different wording but represent the same concept.

Their embeddings are located close together in vector space.

---

### Step 3 — Cluster Interpretation

Once clustering algorithms generate groups, the LLM analyzes each cluster to produce:

* Cluster name
* Common business issue
* Possible technology involved
* Incident priority
* Business impact
* Recommended engineering action

For example,

instead of returning

```text
Cluster 4
```

the LLM generates

```text
Cluster Name

Enterprise VPN Connectivity Issues

Summary

Most incidents indicate authentication failures or unstable VPN sessions affecting remote employees.

Business Impact

Remote employees cannot securely access enterprise systems.

Recommended Action

Review VPN gateway health, authentication servers, MFA configuration and certificate expiration.
```

---

# 9. Expected Business Deliverables

The AI platform should produce:

### Cluster Dashboard

| Cluster              | Incident Count | Priority |
| -------------------- | -------------- | -------- |
| VPN Connectivity     | 4,520          | High     |
| Outlook Issues       | 3,860          | High     |
| SAP Authentication   | 2,240          | Medium   |
| Teams Audio Problems | 1,930          | Medium   |
| Printer Connectivity | 870            | Low      |

---

### Executive Summary

Automatically generated by the LLM.

Example:

> "Approximately 41% of all enterprise incidents during the last seven days relate to authentication and remote connectivity. Outlook-related incidents have increased by 23% compared to the previous week. Most VPN incidents originate from APAC regions following the latest MFA rollout."

---

### Engineering Recommendations

Examples include:

* Increase VPN gateway capacity.
* Review recent authentication policy changes.
* Roll back problematic Outlook update.
* Notify impacted users proactively.
* Create knowledge-base articles for recurring issues.

---

# 10. Success Criteria

The solution will be considered successful if it achieves the following:

* Automatically groups semantically similar incidents.
* Reduces duplicate ticket analysis.
* Detects emerging enterprise issues before manual escalation.
* Identifies rare or anomalous incidents using density-based clustering.
* Reveals hierarchical relationships among support domains.
* Produces human-readable business insights using an LLM.
* Improves incident triage, reporting, and operational decision-making.

---

