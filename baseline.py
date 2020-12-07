from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
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
    #feat.append(1)
    return feat

def bigramFeature(datum):
    feat = [0] * len(bigrams)
    r = ''.join((c for c in datum.lower() if not c in punctuation))
    sentence = r.split()
    for w in range(len(sentence)- 1):
        bg = (sentence[w], sentence[w+1])
        if bg in bigrams:
            feat[bigramId[bg]] += 1
        #feat.append(1)
    return feat

def getber(yp, y, dataset):
    if dataset == 'train':
        tn,fp,fn,tp = confusion_matrix(y_train, yp).ravel()
    elif dataset == 'valid':
        tn,fp,fn,tp = confusion_matrix(y_val, yp).ravel()
    elif dataset == 'test':
        tn,fp,fn,tp = confusion_matrix(y_test, yp).ravel()

    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    return  1 - 0.5 * (tpr + tnr)

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
words = [w[1] for w in counts[:1400]] #Top 1000
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)


X_train = [feature(d, words, wordSet, wordId) for d in X]
X_v = [feature(d, words, wordSet, wordId) for d in X_val]
X_t = [feature(d, words, wordSet, wordId) for d in X_test]

model = LogisticRegression(C=1, max_iter=500)
model.fit(X_train, y_train)

yp = model.predict(X_train)
acc = metrics.accuracy_score(y_train, yp)
ber = getber(yp, y_train, 'train')
print("  Accuracy on %s  is: %s" % ('train', acc))
print("  BER on %s  is: %s" % ('train', ber))

yp = model.predict(X_v)
acc = metrics.accuracy_score(y_val, yp)
ber = getber(yp, y_val, 'valid')
print("  Accuracy on %s  is: %s" % ('valid', acc))
print("  BER on %s  is: %s" % ('valid', ber))

yp = model.predict(X_t)
acc = metrics.accuracy_score(y_test, yp)
ber = getber(yp, y_test, 'test')
print("  Accuracy on %s  is: %s" % ('test', acc))
print("  BER on %s  is: %s" % ('test', ber))

bigramCount = defaultdict(int)
punctuation = set(string.punctuation)

print("Getting bigrams...")
for d in X:
    r = ''.join([c for c in d.lower() if not c in punctuation])
    sentence = r.split()
    for w in range(len(sentence) - 1):
        bigramCount[(sentence[w], sentence[w+1])] += 1

bigramCountList = list()
for b in bigramCount.keys():
    bigramCountList.append((bigramCount[b],b))
bigramCountList.sort()
bigramCountList.reverse()
print("Most frequent bigrams: ", bigramCountList[:5])
bigrams = [x[1] for x in bigramCountList[:2500]]
bigramId = dict(zip(bigrams, range(len(bigrams))))
bigramSet = set(bigrams)

X_train = [bigramFeature(d) for d in X]
X_v = [bigramFeature(d) for d in X_val]
X_t = [bigramFeature(d) for d in X_test]

model.fit(X_train, y_train)

yp = model.predict(X_train)
acc = metrics.accuracy_score(y_train, yp)
ber = getber(yp, y_train, 'train')
print("  Accuracy on %s  is: %s" % ('train', acc))
print("  BER on %s  is: %s" % ('train', ber))

yp = model.predict(X_v)
acc = metrics.accuracy_score(y_val, yp)
ber = getber(yp, y_val, 'valid')
print("  Accuracy on %s  is: %s" % ('valid', acc))
print("  BER on %s  is: %s" % ('valid', ber))

yp = model.predict(X_t)
acc = metrics.accuracy_score(y_test, yp)
ber = getber(yp, y_test, 'test')
print("  Accuracy on %s  is: %s" % ('test', acc))
print("  BER on %s  is: %s" % ('test', ber))

combinedCounts = counts + bigramCountList
combinedCounts.sort(key=lambda tup: tup[0])
combinedCounts.reverse()

combined = [x[1] for x in combinedCounts[:2200]]
combinedId = dict(zip(combined,range(len(combined))))
combinedSet = set(combined)

def combinedFeature(datum):
    feat = [0]*len(combined)
    r = ''.join([c for c in datum.lower() if not c in punctuation])
    sentence = r.split()
    for w in range(len(sentence)):
        if w < len(sentence) - 1:
            bg = (sentence[w], sentence[w+1])
            if bg in combined:
                feat[combinedId[bg]] += 1
        if sentence[w] in combined:
            feat[combinedId[sentence[w]]] += 1
    return feat

X_train = [combinedFeature(d) for d in X]
X_v = [combinedFeature(d) for d in X_val]
X_t = [combinedFeature(d) for d in X_test]

model.fit(X_train, y_train)

yp = model.predict(X_train)
acc = metrics.accuracy_score(y_train, yp)
ber = getber(yp, y_train, 'train')
print("  Accuracy on %s  is: %s" % ('train', acc))
print("  BER on %s  is: %s" % ('train', ber))

yp = model.predict(X_v)
acc = metrics.accuracy_score(y_val, yp)
ber = getber(yp, y_val, 'valid')
print("  Accuracy on %s  is: %s" % ('valid', acc))
print("  BER on %s  is: %s" % ('valid', ber))

yp = model.predict(X_t)
acc = metrics.accuracy_score(y_test, yp)
ber = getber(yp, y_test, 'test')
print("  Accuracy on %s  is: %s" % ('test', acc))
print("  BER on %s  is: %s" % ('test', ber))