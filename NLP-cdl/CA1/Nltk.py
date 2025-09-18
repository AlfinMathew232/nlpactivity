import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Input text
text = ("Manchester United secured a dramatic victory in the Premier League, "
        "thanks to a last-minute goal by their star striker. Fans celebrated "
        "wildly in the stadium and on social media after the thrilling win.")

# 1. Tokenization: break into words
tokens = word_tokenize(text)
print("1. Tokens:")
print(tokens)

# 2. Stop-word removal: remove common useless words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
print("\n2. After Stop-word Removal:")
print(filtered_tokens)

# 3. Stemming: reduce words to base root (may be crude)
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
print("\n3. Stemmed Words:")
print(stemmed_words)

# 4. Lemmatization: reduce to dictionary form (cleaner than stemming)
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\n4. Lemmatized Words:")
print(lemmatized_words)
