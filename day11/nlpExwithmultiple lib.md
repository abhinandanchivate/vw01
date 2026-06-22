# Why Use spaCy Instead of Other NLP Libraries?

**spaCy is not always better than every other NLP library.** It is especially suitable when you need a complete, customizable NLP pipeline for a business application.

For the banking-complaint case study, spaCy can perform tokenization, sentence detection, lemmatization, POS tagging, dependency parsing and NER through one `Doc` object. Its pipeline can also combine trained models with business rules using components such as `EntityRuler`. ([spaCy][1])

---

# 1. Practical Business Requirement

Consider this complaint:

```text
₹15,000 was debited from my account on 18 June 2026,
but the ATM did not dispense cash.
Transaction ID TXN-984562 is still pending.
```

The system needs to:

1. Break the complaint into sentences and words.
2. Convert `debited` into the lemma `debit`.
3. identify nouns, verbs and other grammatical roles.
4. Extract the amount and date.
5. Preserve the transaction ID.
6. Detect the phrase `ATM did not dispense cash`.
7. Categorize the complaint as `ATM Issue`.
8. Route it to the ATM Operations Team.

---

# 2. Practical Comparison

| Requirement              | NLTK                                    | spaCy                                     | Hugging Face Transformers                            | Stanza                            | scikit-learn                        |
| ------------------------ | --------------------------------------- | ----------------------------------------- | ---------------------------------------------------- | --------------------------------- | ----------------------------------- |
| Tokenization             | Yes                                     | Yes                                       | Uses model-specific subword tokenization             | Yes                               | Basic vectorizer tokenization       |
| Sentence splitting       | Yes                                     | Yes                                       | Not its primary purpose                              | Yes                               | No linguistic sentence pipeline     |
| Lemmatization            | Available separately                    | Built into pipelines                      | Usually not exposed as a linguistic lemma            | Yes                               | No                                  |
| POS tagging              | Available separately                    | Built in                                  | Possible with an appropriate model                   | Yes                               | No                                  |
| Dependency parsing       | Available through separate tools/models | Built in                                  | Requires a suitable model                            | Strong focus                      | No                                  |
| Named Entity Recognition | Available                               | Built in                                  | Strong contextual NER models                         | Available                         | No                                  |
| Business rules           | Manual functions and regex              | `Matcher`, `PhraseMatcher`, `EntityRuler` | Usually additional Python logic                      | Additional logic required         | Additional logic required           |
| Text classification      | Traditional approaches                  | Built-in `textcat` components             | Major strength                                       | Not the primary focus             | Strong classical ML baseline        |
| Context understanding    | Limited in traditional pipeline         | Moderate; transformers can be integrated  | Strong                                               | Neural linguistic analysis        | Based mainly on supplied features   |
| Production pipeline      | Requires more assembly                  | Major strength                            | Good, but often heavier                              | More research/linguistic-oriented | Good for classical ML pipelines     |
| Best practical use       | Learning and individual NLP operations  | Business information extraction           | Advanced classification and contextual understanding | Multilingual linguistic analysis  | TF-IDF and classical classification |

NLTK provides tokenizers, taggers, corpora and linguistic resources and is excellent for learning individual NLP operations. Hugging Face pipelines provide simple access to transformer tasks such as NER, sentiment analysis and text classification. Stanza provides neural linguistic pipelines across many languages, while scikit-learn provides vectorizers such as `TfidfVectorizer` for classical document classification. ([nltk.org][2])

---

# 3. Practical Example with NLTK

## Installation

```python
!pip install nltk
```

## Code

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("wordnet")

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer


text = """
₹15,000 was debited from my account on 18 June 2026,
but the ATM did not dispense cash.
Transaction ID TXN-984562 is still pending.
"""

# Sentence tokenization
sentences = sent_tokenize(text)

# Word tokenization
tokens = word_tokenize(text)

# POS tagging
pos_tags = nltk.pos_tag(tokens)

# Lemmatization
lemmatizer = WordNetLemmatizer()

lemmas = [
    lemmatizer.lemmatize(token.lower())
    for token in tokens
]

print("Sentences:")
print(sentences)

print("\nTokens:")
print(tokens)

print("\nPOS tags:")
print(pos_tags)

print("\nLemmas:")
print(lemmas)
```

## Practical observation

NLTK performs the operations, but you usually call separate functions for:

```text
Tokenization
POS tagging
Lemmatization
Chunking or NER
```

You also need to manage downloaded resources and connect the outputs yourself.

NLTK is therefore very useful for:

* Teaching NLP concepts
* Understanding individual algorithms
* Research experiments
* Custom preprocessing
* Stemming and corpus exploration

NLTK’s recommended `word_tokenize` combines a Treebank tokenizer with sentence tokenization, and the toolkit provides separate tagging and information-extraction facilities. ([nltk.org][3])

---

# 4. The Same Example with spaCy

## Installation

```python
!pip install spacy
!python -m spacy download en_core_web_sm
```

## Code

```python
import spacy
import pandas as pd


# Load the trained English pipeline
nlp = spacy.load("en_core_web_sm")


text = """
₹15,000 was debited from my account on 18 June 2026,
but the ATM did not dispense cash.
Transaction ID TXN-984562 is still pending.
"""


# Process the complete text through one pipeline
doc = nlp(text)


# Sentence tokenization
print("Sentences:")

for sentence in doc.sents:
    print(sentence.text.strip())


# Token-level analysis
token_data = []

for token in doc:
    if not token.is_space:
        token_data.append({
            "token": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dependency": token.dep_,
            "is_stopword": token.is_stop,
            "entity_type": token.ent_type_
        })


token_df = pd.DataFrame(token_data)

print("\nToken analysis:")
print(token_df.to_string(index=False))


# Named entities
print("\nNamed entities:")

for entity in doc.ents:
    print(
        entity.text,
        "->",
        entity.label_
    )
```

## Key difference

One call:

```python
doc = nlp(text)
```

can produce access to:

```text
Tokens
Sentences
Lemmas
POS tags
Dependencies
Named entities
```

spaCy stores these annotations in connected `Doc`, `Token` and `Span` objects instead of returning unrelated lists. Its trained pipelines can contain taggers, parsers, lemmatizers and entity recognizers. ([spaCy][1])

---

# 5. Adding Banking Rules in spaCy

A general English model may not know that `ATM`, `UPI`, `TXN-984562` and `cash not dispensed` are banking-specific entities.

spaCy lets us combine trained NLP predictions with business rules.

```python
import spacy
from spacy.matcher import Matcher


nlp = spacy.load("en_core_web_sm")

matcher = Matcher(nlp.vocab)


# ATM cash-dispensation problem
atm_issue_pattern = [
    {"LOWER": "atm"},
    {"OP": "*", "IS_PUNCT": False},
    {"LEMMA": "dispense", "OP": "+"},
    {"LOWER": "cash"}
]


# Unauthorized transaction
fraud_pattern = [
    {"LOWER": {"IN": [
        "unauthorized",
        "fraudulent",
        "unknown"
    ]}},
    {"LEMMA": "transaction"}
]


# Refund delay
refund_pattern = [
    {"LEMMA": "refund"},
    {"OP": "*"},
    {"LOWER": {"IN": [
        "pending",
        "delayed",
        "credited",
        "received"
    ]}}
]


matcher.add(
    "ATM_ISSUE",
    [atm_issue_pattern]
)

matcher.add(
    "FRAUD_ISSUE",
    [fraud_pattern]
)

matcher.add(
    "REFUND_DELAY",
    [refund_pattern]
)


text = """
₹15,000 was debited from my account,
but the ATM did not dispense cash.
"""

doc = nlp(text)

matches = matcher(doc)

for match_id, start, end in matches:

    rule_name = nlp.vocab.strings[match_id]
    matched_text = doc[start:end].text

    print(
        rule_name,
        "->",
        matched_text
    )
```

Possible result:

```text
ATM_ISSUE -> ATM did not dispense cash
```

The important advantage is that a spaCy rule can use linguistic properties such as:

```python
{"LEMMA": "dispense"}
```

Therefore, the rule can match variations such as:

```text
dispense
dispensed
dispensing
```

This is more language-aware than searching only for the exact string `"dispense"`.

spaCy’s matcher can work with token text, lemma, POS, entity labels and other token attributes. It can also combine rule-based recognition with statistical NER. ([spaCy][4])

---

# 6. Using EntityRuler for Banking Entities

```python
import spacy


nlp = spacy.load("en_core_web_sm")


# Add the business entity component
ruler = nlp.add_pipe(
    "entity_ruler",
    before="ner"
)


patterns = [
    {
        "label": "BANKING_CHANNEL",
        "pattern": "ATM"
    },
    {
        "label": "BANKING_CHANNEL",
        "pattern": "UPI"
    },
    {
        "label": "BANKING_PRODUCT",
        "pattern": "credit card"
    },
    {
        "label": "BANKING_PRODUCT",
        "pattern": "home loan"
    },
    {
        "label": "ISSUE",
        "pattern": [
            {"LOWER": "cash"},
            {"LOWER": "not"},
            {"LEMMA": "dispense"}
        ]
    }
]


ruler.add_patterns(patterns)


text = """
The ATM did not dispense cash.
My home loan application is also pending.
"""

doc = nlp(text)


for entity in doc.ents:
    print(
        entity.text,
        "->",
        entity.label_
    )
```

Expected business entities:

```text
ATM -> BANKING_CHANNEL
home loan -> BANKING_PRODUCT
```

The `EntityRuler` can add exact phrases or token-based patterns to named entities and can operate alongside the trained entity recognizer. ([spaCy][4])

---

# 7. Hugging Face Transformers Comparison

Hugging Face is more suitable when the main requirement is contextual classification, sentiment analysis, summarization, question answering or transformer-based NER. Its pipeline API abstracts much of the inference code. ([Hugging Face][5])

## Example: sentiment analysis

```python
!pip install transformers torch
```

```python
from transformers import pipeline


sentiment_model = pipeline(
    "sentiment-analysis"
)


text = """
₹15,000 was debited from my account,
but the ATM did not dispense cash.
"""


result = sentiment_model(text)

print(result)
```

Possible output:

```python
[
    {
        "label": "NEGATIVE",
        "score": 0.99
    }
]
```

## Advantage

The transformer understands the context better than a keyword-only rule.

For example:

```text
The ATM worked correctly.
```

and:

```text
The ATM did not work correctly.
```

have similar words but opposite meanings. A contextual model is more capable of distinguishing such differences.

## Limitation for this case study

A generic sentiment model may tell us:

```text
NEGATIVE
```

but it may not directly provide:

```text
Transaction ID: TXN-984562
Department: ATM Operations
Banking category: Cash not dispensed
```

You still need domain-specific training, extraction rules or another pipeline around it.

---

# 8. Stanza Comparison

Stanza provides neural processing for sentence splitting, tokenization, lemmatization, POS tagging, morphological analysis, dependency parsing and NER. It has a strong multilingual and Universal Dependencies focus. ([stanfordnlp.github.io][6])

## Example

```python
!pip install stanza
```

```python
import stanza


stanza.download("en")

nlp = stanza.Pipeline(
    "en",
    processors=(
        "tokenize,"
        "pos,"
        "lemma,"
        "depparse,"
        "ner"
    )
)


text = """
₹15,000 was debited from my account,
but the ATM did not dispense cash.
"""


doc = nlp(text)


for sentence in doc.sentences:

    for word in sentence.words:

        print({
            "word": word.text,
            "lemma": word.lemma,
            "pos": word.upos,
            "dependency": word.deprel
        })


print("\nEntities:")

for entity in doc.ents:
    print(
        entity.text,
        entity.type
    )
```

## When Stanza may be better

Choose Stanza when:

* Linguistic analysis is the central requirement.
* Universal Dependencies annotations are important.
* You are researching syntax or morphology.
* You require its supported multilingual models.
* Consistency of linguistic annotation across languages is important.

Stanza is a strong alternative; spaCy is not automatically more accurate for every language or task.

---

# 9. scikit-learn Comparison

scikit-learn is useful when the objective is:

```text
Text → TF-IDF vectors → Classification model
```

It does not replace a complete linguistic NLP pipeline.

## Practical classification example

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


training_texts = [
    "ATM did not dispense cash",
    "cash was not received from ATM",
    "UPI transfer failed",
    "UPI transaction is pending",
    "someone used my card without permission",
    "unauthorized card transaction",
    "refund has not been credited",
    "refund is still pending"
]


training_labels = [
    "ATM Issue",
    "ATM Issue",
    "UPI Issue",
    "UPI Issue",
    "Fraud Issue",
    "Fraud Issue",
    "Refund Delay",
    "Refund Delay"
]


model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])


model.fit(
    training_texts,
    training_labels
)


new_complaint = [
    "The ATM deducted money but gave no cash"
]


prediction = model.predict(
    new_complaint
)


print(prediction[0])
```

Possible output:

```text
ATM Issue
```

`TfidfVectorizer` converts text into weighted numeric features suitable for classical classifiers. It is often a strong, fast baseline for text classification, but it does not itself provide lemmas, dependencies or general-purpose named entities. ([Scikit-learn][7])

---

# 10. Practical Decision Matrix

| Project requirement                                 | Recommended tool |
| --------------------------------------------------- | ---------------- |
| Teach tokenization, stemming and basic NLP concepts | NLTK             |
| Build an end-to-end business extraction pipeline    | spaCy            |
| Extract custom banking terms using rules            | spaCy            |
| Perform POS, lemma, dependency and NER together     | spaCy or Stanza  |
| Perform advanced contextual sentiment analysis      | Hugging Face     |
| Fine-tune BERT for complaint classification         | Hugging Face     |
| Build a quick TF-IDF classification baseline        | scikit-learn     |
| Perform detailed multilingual linguistic research   | Stanza           |
| Combine rules with statistical NER                  | spaCy            |
| Process large batches using configurable components | spaCy            |
| Generate text or summarize complaints               | Hugging Face     |

spaCy’s `nlp.pipe()` can process documents in batches, and unnecessary pipeline components can be disabled to reduce processing work. ([spaCy][1])

---

# 11. Recommended Architecture for the Banking Case Study

The strongest practical solution is not necessarily to select only one library.

```text
Customer complaint
        ↓
spaCy
Tokenization, lemma, POS, NER and banking rules
        ↓
Custom extraction
Amount, transaction ID, account/card information
        ↓
Hugging Face transformer
Contextual complaint classification and urgency
        ↓
Business routing rules
        ↓
ATM / UPI / Fraud / Loan / Refund department
```

## Why use a hybrid approach?

### spaCy handles

* Linguistic preprocessing
* Sentence and word boundaries
* Lemmatization
* POS tagging
* Entity extraction
* Business-specific rules
* Structured outputs

### Hugging Face handles

* Contextual classification
* Complex sentiment
* Transformer-based NER
* Fine-tuned domain models
* Difficult language variations

### scikit-learn provides

* A quick baseline
* TF-IDF features
* Interpretable traditional models
* Low-compute classification

---

# 12. Final Conclusion

Use **spaCy** for the banking case study because the requirement is not only sentiment prediction. The application must extract structured details, apply banking rules and create a maintainable processing pipeline.

```text
Choose spaCy when:
Linguistic analysis + custom rules + production pipeline are required.

Choose NLTK when:
Learning and experimenting with individual NLP operations.

Choose Hugging Face when:
Contextual understanding and transformer accuracy are the priority.

Choose Stanza when:
Detailed multilingual linguistic analysis is the priority.

Choose scikit-learn when:
You need a simple TF-IDF-based classification baseline.
```

For a real banking complaint system, a practical recommendation is:

```text
spaCy + custom rules + a fine-tuned transformer classifier
```

rather than spaCy alone.

[1]: https://spacy.io/usage/processing-pipelines?utm_source=chatgpt.com "Language Processing Pipelines"
[2]: https://www.nltk.org/?utm_source=chatgpt.com "NLTK :: Natural Language Toolkit"
[3]: https://www.nltk.org/api/nltk.tokenize.word_tokenize.html?utm_source=chatgpt.com "nltk.tokenize.word_tokenize"
[4]: https://spacy.io/usage/rule-based-matching?utm_source=chatgpt.com "Rule-based matching · spaCy Usage Documentation"
[5]: https://huggingface.co/docs/transformers/main_classes/pipelines?utm_source=chatgpt.com "Pipelines"
[6]: https://stanfordnlp.github.io/stanza/?utm_source=chatgpt.com "Overview - Stanza"
[7]: https://scikit-learn.org/stable/modules/feature_extraction.html?utm_source=chatgpt.com "8.2. Feature extraction"
