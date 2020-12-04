from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import numpy as np
import pandas as pd
import nltk
import string
from sklearn import metrics

def feature(datum, words, wordSet, wordId):
    feat = [0]*len(wordSet)
    t = datum
    t = t.lower() # lowercase string
    t = [c for c in t if not (c in punct)] # non-punct characters
    t = ''.join(t) # convert back to string
    words = t.strip().split() # tokenizes
    for w in words:
        if not (w in wordSet): continue
        feat[wordId[w]] += 1
    feat.append(1)
    return feat

df = pd.read_csv('./datasets/training80k.csv', names=['label', 'sentence'], sep='\t', engine='python')

X = df['sentence'].values
y_train = df['label'].values
df_test = pd.read_csv('./datasets/test20k.csv', names=['label', 'sentence'], sep='\t', engine='python')

X_val, X_test, y_val, y_test = train_test_split(df_test['sentence'].values, df_test['label'].values,random_state = 9, test_size=0.5)

punct = string.punctuation
wordCount = defaultdict(int)
totalWords = 0

for d in X:
    d = d.lower()
    d = [c for c in d if not (c in punct)]
    d = ''.join(d)
    words = d.strip().split()
    for w in words:
        totalWords += 1
        wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()
words = [w[1] for w in counts[:1000]] #Top 1000
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

X_train = [feature(d, words, wordSet, wordId) for d in X]
X_v = [feature(d, words, wordSet, wordId) for d in X_val]
X_t = [feature(d, words, wordSet, wordId) for d in X_test]

model = LogisticRegression(C=1, max_iter=500, verbose=1)
model.fit(X_train, y_train)
yp = model.predict(X_train)
acc = metrics.accuracy_score(y_train, yp)
print("  Accuracy on %s  is: %s" % ('train', acc))
yp = model.predict(X_v)
acc = metrics.accuracy_score(y_val, yp)
print("  Accuracy on %s  is: %s" % ('valid', acc))
yp = model.predict(X_t)
acc = metrics.accuracy_score(y_test, yp)
print("  Accuracy on %s  is: %s" % ('test', acc))