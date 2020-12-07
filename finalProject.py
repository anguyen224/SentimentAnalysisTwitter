import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from sklearn.metrics import confusion_matrix
import re
import pickle

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

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

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
    return (" ".join(words)).lower().strip()

df = pd.read_csv('./datasets/training80k.csv', names=['label', 'sentence'], sep='\t', engine='python')
sentences = df['sentence'].values
y = df['label'].values
combined_pat = "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"
tok = WordPunctTokenizer()

testing = sentences
test_result = []
for t in testing:
    test_result.append(tweet_cleaner(t))
#print(test_result)

sentences = test_result
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.2, random_state=1000)
tokenizer = Tokenizer(num_words=5000)
#creates the vocabulary index based on work frequency
#every word gets its own unique integer value
tokenizer.fit_on_texts(sentences_train)
#takes each word in the training set and replaces
#it with its corresponding integer alue from the word_index dictionary

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)
vocab_size = len(tokenizer.word_index)+1
maxlen = 100

#pads each sentence with 0 up to length 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
print(X_train)
embedding_dim = 50
embedding_matrix = create_embedding_matrix('./glove.6B.50d.txt',tokenizer.word_index, embedding_dim)

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim,
                           weights=[embedding_matrix],
                           input_length=maxlen,
                           trainable=True))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,
                    epochs=15,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)

with open('trainedModel.nn', 'wb') as file:
    pickle.dump(model, file)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
