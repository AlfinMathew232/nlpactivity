import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Download required resources (only once)
nltk.download('stopwords')
nltk.download('punkt')

# Example corpus: Travel and Tourism
texts = [
    "Paris is famous for the Eiffel Tower and the Louvre Museum.",
    "Tokyo offers modern technology, sushi, and beautiful temples.",
    "New York City is known for Times Square and Central Park.",
    "Tourists love exploring the beaches and nightlife of Barcelona."
]

stop_words = set(stopwords.words('english'))

# Preprocess
processed_texts = []
for doc in texts:
    tokens = word_tokenize(doc.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    processed_texts.append(tokens)

# Create dictionary & corpus
dictionary = corpora.Dictionary(processed_texts)
corpus = [dictionary.doc2bow(text) for text in processed_texts]

# LDA Model
lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# Print topics
for idx, topic in lda_model.print_topics():
    print("Topic: {} \nWords: {}".format(idx, topic))

# Visualization
vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, 'lda_visualization.html')
print("\nInteractive LDA visualization saved as lda_visualization.html")
