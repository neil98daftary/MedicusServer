# Preprocessing

import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess(x):
    y = re.sub('[^a-zA-Z]', ' ', x)
    y = y.lower()
    y = y.split()
    ps = PorterStemmer()
    y = [ps.stem(word) for word in y if not word in stop_words]
    y = " ".join(y)
    return y
