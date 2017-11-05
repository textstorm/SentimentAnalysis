
import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import Counter


def review_to_wordlist(review, remove_stopwords=False):
  review_text = BeautifulSoup(review, "html5lib").get_text()
  review_text = re.sub("[^a-zA-Z]"," ", review_text)
  words = review_text.lower().split()
  if remove_stopwords:
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]
  return(words)

def review_to_sentences(review, tokenizer, remove_stopwords=False):
  raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
  sentences = []
  for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
      sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
  return sentences

def build_vocab(sentences, max_words=None):
  word_count = Counter()
  for sentence in sentences:
    for word in sentence:
      word_count[word] += 1

  print("The dataset has %d different words totally" % len(word_count))
  if not max_words:
    max_words = len(word_count)
  else:
    filter_out_words = len(word_count) - max_words

  word_dict = word_count.most_common(max_words)
  return {word[0]: index + 1 for (index, word) in enumerate(word_dict)}

def vectorize(data, word_dict, verbose=True):
  reviews = []
  for idx, line in enumerate(data):
    seq_line = [word_dict[w] if w in word_dict else 0 for w in line]
    reviews.append(seq_line)

    if verbose and (idx % 5000 == 0):
      print("Vectorization: processed {}".format(idx))
  return reviews

data = pd.read_csv('data/labeledTrainData.tsv.zip', 
                    compression='zip', 
                    delimiter='\t', 
                    header=0, 
                    quoting=3)

reviews = data["review"]
labels = list(data['sentiment'])
sentences = []
for review in reviews:
  if len(review) > 0:
    sentences.append(review_to_wordlist(review.decode('utf8').strip(), remove_stopwords=True))

word_dict = build_vocab(sentences, max_words=5000)
vec_reviews = vectorize(sentences, word_dict, verbose=True)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D

max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2

train_data = vec_reviews[0: 20000]
test_data = vec_reviews[20000:]
y_train = labels[0:20000]
y_test = labels[20000:]
X_train = sequence.pad_sequences(train_data, maxlen=maxlen)
X_test = sequence.pad_sequences(test_data, maxlen=maxlen)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

model = Sequential()

model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen,
                    dropout=0.2))

model.add(Conv1D(filters=nb_filter,
                        kernel_size=filter_length,
                        padding='valid',
                        activation='relu'))

model.add(GlobalMaxPooling1D())

model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_data=(X_test, y_test))


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()