"""
LAB 4 - GENERATIVE AI: A Mini-RAG Pipeline From First Principles
=================================================================
ML Masterclass · Module 6 companion lab

Goal: demystify Retrieval-Augmented Generation by building every box of
the architecture slide in ~120 lines: chunking -> indexing -> retrieval
-> prompt assembly -> (optional) LLM call. TF-IDF stands in for neural
embeddings; the ARCHITECTURE is identical, which is the point.

Run:  python lab4_rag_mini.py
Deps: scikit-learn, numpy. Runs fully offline.
      (Optional Section 6 shows where a real LLM API call plugs in.)
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# SECTION 1 — The knowledge base (stand-in for your wikis/policies/PDFs)
# ----------------------------------------------------------------------
DOCUMENTS = {
    "retention_policy.md": """
        Retention Offer Policy (v3, effective April 2026).
        Agents may offer a discount of up to 20% on the monthly plan for
        customers flagged HIGH churn risk, for a maximum of 6 months.
        MEDIUM risk customers qualify for a 10% discount or a free OTT
        bundle, never both. Offers above these limits require team-lead
        approval recorded in CRM. Customers in the first 90 days of a
        contract are not eligible for retention discounts.
    """,
    "plan_catalogue.md": """
        Plan Catalogue 2026. Fiber Basic: 50 Mbps, Rs 499/month.
        Fiber Plus: 200 Mbps, Rs 799/month, includes router rental.
        Fiber Pro: 1 Gbps, Rs 1499/month, includes static IP and
        priority support. All fiber plans include unlimited data with
        fair-use at 3.3 TB. Mobile postpaid add-on: Rs 299 for 50 GB.
    """,
    "outage_compensation.md": """
        Outage Compensation Rules. For unplanned outages exceeding 24
        continuous hours, customers are entitled to a pro-rata credit of
        2 days per 24-hour block, applied automatically within one
        billing cycle. Planned maintenance announced 48 hours in advance
        is not compensable. Repeated outages (3+ in 30 days) escalate to
        the network quality desk and qualify the customer for one free
        month of router rental.
    """,
    "save_playbook.md": """
        Save-Conversation Playbook. Step 1: acknowledge the stated
        reason for leaving without arguing. Step 2: review the
        customer's last 90 days - outages, tickets, bill changes - and
        name them proactively. Step 3: match the offer to the driver:
        price-driven churn gets a discount, quality-driven churn gets a
        service credit plus a network review, competitor-driven churn
        gets a plan upgrade comparison. Never lead with the maximum
        discount. Close by confirming the resolution in writing.
    """,
}

# ----------------------------------------------------------------------
# SECTION 2 — Chunking: the most underrated dial in RAG
# ----------------------------------------------------------------------
def chunk(text, max_words=60):
    """Split on sentences, pack into ~max_words chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    chunks, cur = [], []
    for s in sentences:
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= max_words:
            chunks.append(" ".join(cur)); cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks

corpus, meta = [], []
for doc_id, text in DOCUMENTS.items():
    for i, ch in enumerate(chunk(text)):
        corpus.append(ch)
        meta.append(f"{doc_id}#chunk{i}")

print("=" * 68)
print(f"SECTION 2 · Indexed {len(corpus)} chunks from {len(DOCUMENTS)} documents")
print("=" * 68)

# ----------------------------------------------------------------------
# SECTION 3 — Embed + index (TF-IDF here; swap for neural embeddings
#             + a vector DB in production. Same interface: text -> vector.)
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
index = vectorizer.fit_transform(corpus)            # the "vector store"

def retrieve(query, k=3):
    """The R in RAG: rank chunks by similarity to the query."""
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, index).ravel()
    top = np.argsort(sims)[::-1][:k]
    return [(meta[i], corpus[i], float(sims[i])) for i in top]

# ----------------------------------------------------------------------
# SECTION 4 — Prompt assembly: retrieved chunks become CONTEXT
# ----------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a retention-desk assistant for TeleConnect.
Answer the agent's question using ONLY the context below. Cite the
source id for every fact. If the context does not contain the answer,
say so explicitly - do not guess.

CONTEXT:
{context}

AGENT QUESTION: {question}

ANSWER (with citations):"""

def build_prompt(question, k=3):
    hits = retrieve(question, k)
    context = "\n".join(f"[{src}] {text}" for src, text, _ in hits)
    return PROMPT_TEMPLATE.format(context=context, question=question), hits

# ----------------------------------------------------------------------
# SECTION 5 — End-to-end dry run (retrieval quality is visible)
# ----------------------------------------------------------------------
QUESTIONS = [
    "What discount can I give a high churn-risk customer?",
    "Customer had a 30-hour outage last week - what do they get?",
    "Can a customer who joined last month get a retention discount?",
    "What is the price of the 1 Gbps plan?",
]

for q in QUESTIONS:
    print("\n" + "-" * 68)
    print("QUESTION:", q)
    prompt, hits = build_prompt(q)
    print("RETRIEVED:")
    for src, _, score in hits:
        print(f"   {score:.3f}  {src}")
    # The prompt below is what you would send to the LLM:
    # print(prompt)

print("""
\nNotice the failure-analysis discipline this enables: if the final
answer is wrong, FIRST check 'did we retrieve the right chunk?'
(scores above). Most RAG defects are retrieval defects - chunking,
query phrasing, missing reranking - not model defects.
""")

# ----------------------------------------------------------------------
# SECTION 6 — Where the real LLM plugs in (reference, not executed)
# ----------------------------------------------------------------------
# In production, replace the print with an API call, e.g.:
#
#   import anthropic
#   client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
#   msg = client.messages.create(
#       model="claude-sonnet-4-5",
#       max_tokens=500,
#       messages=[{"role": "user", "content": prompt}],
#   )
#   print(msg.content[0].text)
#
# Production hardening checklist (Module 6 risk register):
#   [ ] Treat retrieved text as DATA, never as instructions (injection)
#   [ ] Enforce output contract (citations present? offer within policy?)
#   [ ] Permission filter at retrieval time (who may see which docs)
#   [ ] Golden-set eval: 50 questions with approved answers, run on
#       every model/prompt/index change
#   [ ] Log retrievals + answers for audit (with retention policy)
#
# DISCUSSION
# 1. Ask: "What's the maximum discount?" without specifying risk tier.
#    Does retrieval surface enough context for a SAFE answer?
# 2. Break it on purpose: set max_words=15 (over-chunking). Which
#    questions start failing, and why?
# 3. The policy document changes weekly. Why does this make RAG the
#    right adaptation choice over fine-tuning here?
