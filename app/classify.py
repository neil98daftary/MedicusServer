# Bag of Words and Logistic Regression

from __future__ import print_function
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from django.conf import settings
from .preprocess import preprocess
from .train import train

def classify(sentence, to_train):
    Main_Path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(Main_Path)

    if to_train == 1:
        train_decision()
    else:
        if not os.path.isfile("vectorizer.pkl") or not os.path.isfile("model.pkl"):
            print(app.root_path, file=sys.stderr)
            if not os.path.isfile("vectorizer.pkl"):
                if os.path.isfile("model.pkl"):
                    os.remove("model.pkl")
            if not os.path.isfile("model.pkl"):
                if os.path.isfile("vectorizer.pkl"):
                    os.remove("vectorizer.pkl")
            train()

    corpus = []
    corpus.append(preprocess(sentence))
    X = CountVectorizer(vocabulary = joblib.load("vectorizer.pkl"))
    R = X.transform(corpus).todense()
    model = joblib.load("model.pkl")
    rating = model.predict(R)[-1]
    return rating



def train_decision():
    if os.path.isfile("vectorizer.pkl"):
        os.remove("vectorizer.pkl")
    if os.path.isfile("model.pkl"):
        os.remove("model.pkl")
    train()


# classify("It was a very bad movie", 0)
