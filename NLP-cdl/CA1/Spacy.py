import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Sample random tech-related news text
text = ("Apple has announced the release of its latest iPhone model, "
        "featuring an advanced AI-powered camera system and improved battery life. "
        "The launch event took place in California, attracting a global audience.")

# Process the text
doc = nlp(text)

# Tokenization
tokens = [token.text for token in doc]
print("1. Tokens:")
print(tokens)

# Stop-word removal
non_stop_tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
print("\n2. After Stop-word Removal:")
print(non_stop_tokens)

# Lemmatization (excluding stopwords & punctuation)
lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
print("\n3. Lemmatized Words:")
print(lemmas)


# token.text	Raw tokens	['Apple', 'has', 'announced', ...]
# token.is_stop	Identifies non-informative words	Removes "has", "the", "in", etc.
# token.is_alpha	Keeps only actual words (no symbols or digits)
# token.lemma_	Lemmatizes to base form	"featuring" → "feature"

