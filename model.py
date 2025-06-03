# Step 1: Import libraries
import pandas as pd
import numpy as np
import string
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

fake_news=pd.read_csv("Fake.csv")
true_news=pd.read_csv("True.csv")
fake_news["class"]=0
true_news["class"]=1
news_df = pd.concat([fake_news, true_news], ignore_index=True)

X = news_df['text']
y = news_df['class']

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectors = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)


# Save the model
with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
