from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import numpy as np
import pandas as pd
import nltk
import string
from sklearn import metrics

def readText(negPath, posPath):
    neg = set()
    pos = set()
    for l in open(negPath):
        l = l.strip('\n')
        neg.add(l)
    for l in open(posPath):
        l = l.strip('\n')
        pos.add(l)
    return pos,neg

def predict(X):
    yp = []
    punct = string.punctuation
    for d in X:
        pos = 0
        neg = 0
        d = d.lower()
        d = [c for c in d if not (c in punct)]
        d = ''.join(d)
        words = d.strip().split()
        ##print(words)
        #input()
        for w in words:
            if w in posWords:
                pos += 1
            elif w in negWords:
                neg += 1
        #print(pos)
        
        #print(neg)
        if pos > neg:
            yp.append(1)
        else:
            yp.append(0)
    return yp
df = pd.read_csv('./datasets/training80k.csv', names=['label', 'sentence'], sep='\t', engine='python')

X = df['sentence'].values
y_train = df['label'].values
df_test = pd.read_csv('./datasets/test20k.csv', names=['label', 'sentence'], sep='\t', engine='python')

X_val, X_test, y_val, y_test = train_test_split(df_test['sentence'].values, df_test['label'].values,random_state = 9, test_size=0.5)
posWords, negWords = readText("negative-words.txt", "positive-words.txt")
yp = predict(X_val)
acc = metrics.accuracy_score(y_val, yp)
print("  Accuracy on %s  is: %s" % ('valid', acc))

yp = predict(X_test)
acc = metrics.accuracy_score(y_test, yp)
print("  Accuracy on %s  is: %s" % ('test', acc))