# For training only

import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from django.conf import settings
from .preprocess import preprocess

def train():
    Main_Path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(Main_Path)

    data = pd.read_csv("imdb_labelled.txt", delimiter="\t", header = None, names=["Review", "Label"])

    reviews = data["Review"].values.tolist()
    labels = data["Label"].values.tolist()

    corpus = []

    for i in range(0, len(reviews)):
        corpus.append(preprocess(reviews[i]))

    cv = CountVectorizer(max_features = 1500)
    X = cv.fit_transform(corpus).toarray()
    joblib.dump(cv.vocabulary_, "vectorizer.pkl")

    X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size = 0.2)

    model = LogisticRegression()
    # model.fit(X, Y_train)
    model.fit(X, labels)
    joblib.dump(model, "model.pkl")

    # print accuracy_score(Y_test, model.predict(X_test))
