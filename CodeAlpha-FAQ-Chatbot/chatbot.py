import json
import string
import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
# Download NLTK Resources (First Time Only)
# ======================================================

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ======================================================
# Project Paths
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ======================================================
# Load Files
# ======================================================

with open(os.path.join(DATA_DIR, "faq.json"), "r", encoding="utf-8") as f:
    faqs = json.load(f)

with open(os.path.join(DATA_DIR, "greetings.json"), "r", encoding="utf-8") as f:
    greetings = json.load(f)

with open(os.path.join(DATA_DIR, "synonyms.json"), "r", encoding="utf-8") as f:
    synonyms = json.load(f)

# ======================================================
# NLP Setup
# ======================================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ======================================================
# Preprocessing
# ======================================================

def preprocess(text):

    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    tokens = word_tokenize(text)

    words = []

    for word in tokens:

        if word not in stop_words:

            word = lemmatizer.lemmatize(word)

            words.append(word)

    return words


# ======================================================
# Replace Synonyms
# ======================================================

def normalize(tokens):

    normalized = []

    for word in tokens:

        replaced = False

        for key, values in synonyms.items():

            if word == key:

                normalized.append(key)
                replaced = True
                break

            if word in values:

                normalized.append(key)
                replaced = True
                break

        if not replaced:

            normalized.append(word)

    return normalized


# ======================================================
# Greeting Detection
# ======================================================

def check_greetings(user_text):

    text = user_text.lower()

    for word in greetings["greetings"]["patterns"]:

        if word in text:
            return greetings["greetings"]["response"]

    for word in greetings["thanks"]["patterns"]:

        if word in text:
            return greetings["thanks"]["response"]

    for word in greetings["goodbye"]["patterns"]:

        if word in text:
            return greetings["goodbye"]["response"]

    return None


# ======================================================
# Build TF-IDF
# ======================================================

questions = []

for item in faqs:

    q = preprocess(item["question"])

    q = normalize(q)

    questions.append(" ".join(q))

vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(questions)

# ======================================================
# Main Chat Function
# ======================================================

def get_response(user_question):

    greeting = check_greetings(user_question)

    if greeting:

        return greeting, 1.0

    tokens = preprocess(user_question)

    tokens = normalize(tokens)

    processed = " ".join(tokens)

    user_vector = vectorizer.transform([processed])

    similarity = cosine_similarity(
        user_vector,
        question_vectors
    )

    best = similarity.argmax()

    confidence = similarity[0][best]

    if confidence < 0.25:

        return (
            "❌ Sorry, I couldn't understand your question.\n\nTry asking about:\n\n• Internship\n• Certificates\n• GitHub\n• Submission\n• Tasks",
            confidence
        )

    return (
        faqs[best]["answer"],
        confidence
    )


# ======================================================
# Terminal Test
# ======================================================

if __name__ == "__main__":

    print("=" * 60)
    print("🤖 CodeAlpha AI Internship Assistant")
    print("=" * 60)

    while True:

        user = input("\nYou : ")

        if user.lower() == "exit":

            print("\nBot : Goodbye 👋")
            break

        answer, score = get_response(user)

        print("\nBot :", answer)

        print("Confidence :", round(score * 100, 2), "%")