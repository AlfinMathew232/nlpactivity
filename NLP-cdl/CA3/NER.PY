# CADL3: Named Entity Recognition
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")

# Example dataset: sports news
text = """
Lionel Messi scored two goals for Inter Miami in a thrilling match against LA Galaxy.
The FIFA World Cup will be hosted in Canada, Mexico, and the United States in 2026.
Cristiano Ronaldo continues his career with Al Nassr in Saudi Arabia.
"""

doc = nlp(text)

# Extract named entities
print("🔹 Named Entities")
for ent in doc.ents:
    print(ent.text, ent.label_)

# Structured Information (Persons and Organizations)
entities = [(ent.text, ent.label_) for ent in doc.ents]
df = pd.DataFrame(entities, columns=["Entity", "Type"])
print("\nStructured Info:\n", df)
