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
import re
import pickle
import spacy

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
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()

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

def embed_tweet(tweet):
    #clean tweet
    cleaned = []
    cleaned.append(tweet_cleaner(tweet))
    tokened = tokenizer.texts_to_sequences(cleaned)
    tokened = pad_sequences(tokened, padding='post', maxlen=100)

    guess = model.predict(tokened)
    return guess

def entityExtract(tweet):
    nlp = spacy.load("en_core_web_sm")
    s = nlp(tweet)

    for ent in s.ents:
        print(ent.text, ent.label_)

def find_sentiment(keyword, df):
    average = 0
    keyword_tweet_list = []
    for sentence in df['sentence']:
        if keyword in sentence.lower().split():
            keyword_tweet_list.append(sentence)
    for tweet in keyword_tweet_list:
        average += predict_tweet(tweet)
    if len(keyword_tweet_list) == 0:
        return
    return average/len(keyword_tweet_list)

def predict_tweet(tweet):
#clean tweet
    cleaned = []
    cleaned.append(tweet_cleaner(tweet))
    tokened = tokenizer.texts_to_sequences(cleaned)
    tokened = pad_sequences(tokened, padding='post', maxlen=100)

    guess = model.predict(tokened)
    return guess

df = pd.read_csv('./datasets/training80k.csv', names=['label', 'sentence'], sep='\t', engine='python')
sentences = df['sentence'].values
y = df['label'].values
combined_pat = "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"
tok = WordPunctTokenizer()

testing = sentences
test_result = []
for t in testing:
    test_result.append(tweet_cleaner(t))

sentences = test_result
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.2, random_state=1000)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)
vocab_size = len(tokenizer.word_index)+1
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
#print(X_train)
embedding_dim = 50
embedding_matrix = create_embedding_matrix('./glove.6B.50d.txt',tokenizer.word_index, embedding_dim)


with open('trainedModel.nn','rb') as file:
    model = pickle.load(file)

#print(model.summary())

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

df = pd.read_csv('./datasets/training400.csv', names=['label', 'sentence'], sep='\t', engine='python')

while True:
    option = input(
        "\nEnter '1' for sentiment analysis of a sentence. Enter '2' for sentiment analysis of an entity. Enter 'EXIT' to exit the program. \n>>> ")
    if option == '1':

        print ("\nEnter a string for sentiment analysis: ")
        print ("(Enter 'EXIT' to return to the menu)")
        while (True):
            inp = input(">>> ")
            if inp == "EXIT":
                break
            print(embed_tweet(str(inp)))
            #entityExtract(inp)
            print('\n')
    elif option == '2':
        print ("\nEnter a string for sentiment analysis of an entity: ")
        print ("(Enter 'EXIT' to return to the menu)")
        while (True):
            inp = input(">>> ")
            if inp == "EXIT":
                break
            out = (find_sentiment(inp, df))
            if not out:
                print ("Entity not found")
            else:
                print (out)
            print('\n')
    elif option == 'EXIT':
        break

