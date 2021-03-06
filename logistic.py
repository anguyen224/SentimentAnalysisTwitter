import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from sklearn import preprocessing
from sklearn import metrics
import re
import pickle
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from autocorrect import Speller

nltkstop = set(stopwords.words('english'))
stop = ['an', 'and', 'a', 'for', 'was', 'we','they', 'do', 'of', 'am',
        'who', 'as', 'from', 'had', 'in', 'those']
combined_pat = "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # Tokenize and join together to remove unneccessary white spaces

    words = tok.tokenize(lower_case)    
    #words = tok.tokenize(lower_case)
    #words = [spell(w) for w in words2]

    #words = [w for w in words if w not in stopwords.words('english')]
    return (" ".join(words)).lower().strip()

df = pd.read_csv('./datasets/training80k.csv', names=['label', 'sentence'], sep='\t', engine='python')
tok = WordPunctTokenizer()

X = df['sentence'].values
y_train = df['label'].values

df_test = pd.read_csv('./datasets/test20k.csv', names=['label', 'sentence'], sep='\t', engine='python')


X_val, X_test, y_val, y_test = train_test_split(df_test['sentence'].values, df_test['label'].values,random_state = 9, test_size=0.5)

spell = Speller(fast=True)

X_train = [tweet_cleaner(t) for t in X]
X_test = [tweet_cleaner(t) for t in X_test]
X_val = [tweet_cleaner(t) for t in X_val]

tfidf = TfidfVectorizer(max_df = 0.08, ngram_range = (1,2)) #Can add stopwords here
X = tfidf.fit_transform(X_train)
le = preprocessing.LabelEncoder()
le.fit(y_train)
target_labels = le.classes_
train_y = le.transform(y_train)

cls = LogisticRegression(verbose = 1, random_state=0, solver='lbfgs', max_iter=400)
cls.fit(X, train_y)
yp = cls.predict(X)
acc = metrics.accuracy_score(train_y, yp)
print("  Accuracy on %s  is: %s" % ('train', acc))

valid = tfidf.transform(X_val)
yp = cls.predict(valid)
acc = metrics.accuracy_score(y_val, yp)
print("  Accuracy on %s  is: %s" % ('valid', acc))

test = tfidf.transform(X_test)
yp = cls.predict(test)
acc = metrics.accuracy_score(y_test, yp)
print("  Accuracy on %s  is: %s" % ('test', acc))


