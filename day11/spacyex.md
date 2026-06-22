!pip install nltk

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
