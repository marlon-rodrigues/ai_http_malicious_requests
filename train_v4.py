import sys
import os
import json
import pandas
import numpy
import optparse
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict
from sklearn.model_selection import train_test_split 

requests_data = pandas.read_csv('Datasets/csic_2010_csv.csv', engine='python', quotechar='|', header=None)
print("Data Sample")
print(requests_data.head(5))
print("\n")

dataset = requests_data.sample(frac=1).values # convert data into array
print("Data Array Sample")
print(dataset)
print("\n")

# Preprocess dataset
X = dataset[:,0]
print("---- X ----")
print(X)
print("\n")

Y = dataset[:,1]
print("----- Y -----")
print(Y)
print("\n")

print("Removing unnecessary columns...")
for index, item in enumerate(X):
    # Quick hack to space out json elements
    reqJson = json.loads(item, object_pairs_hook=OrderedDict)
    del reqJson['contentLength']
    del reqJson['cacheControl']
    del reqJson['index']
    del reqJson['label']
    X[index] = json.dumps(reqJson, separators=(',', ':'))
print("Removed unnecessary columns...\n")

print("Creating token object for X....")
tokenizer = Tokenizer(filters='\t\n', char_level=True)
tokenizer.fit_on_texts(X)
print("Token object created\n")

print("Word Index")
print(tokenizer.word_index)
print("\n")

# Extract and save word dictionary
word_dict_file = 'build/word-dictionary.json'

if not os.path.exists(os.path.dirname(word_dict_file)):
    os.makedirs(os.path.dirname(word_dict_file))

with open(word_dict_file, 'w') as outfile:
    json.dump(tokenizer.word_index, outfile, ensure_ascii=False)

num_words = len(tokenizer.word_index)+1
print("Number of Words")
print(num_words)
print("\n")

print("Tokenizing X...")
X = tokenizer.texts_to_sequences(X)
print("X Tokenized\n")

max_log_length = 1024
train_size = int(len(dataset) * .75)

print("Padding X....")
#X_processed = sequence.pad_sequences(X, maxlen=max_log_length)
X_processed = sequence.pad_sequences(X)
print("X Padded\n")

print("Creating test and training sets...")
#X_train, X_test = X_processed[0:train_size], X_processed[train_size:len(X_processed)]
#Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 36)
print("Test and training sets created...\n")

tb_callback = TensorBoard(log_dir='./logs', embeddings_freq=1)

model = Sequential()
#model.add(Embedding(num_words, 32, input_length=max_log_length))
model.add(Embedding(num_words, 128, input_length=len(X[0]), dropout = 0.2))
model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
#model.add(Dropout(0.5))
#model.add(LSTM(64, recurrent_dropout=0.5))
#model.add(LSTM(64, return_sequences=True))
#model.add(LSTM(64, return_sequences=True))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, validation_split=0.25, epochs=1, batch_size=128, callbacks=[tb_callback])

# Evaluate model
score, acc = model.evaluate(X_test, Y_test, verbose=1, batch_size=128)
print("Model Accuracy: {:0.2f}%".format(acc * 100))

# Save model
model.save_weights('model-weights.h5')
model.save('model.h5')
with open('model_json', 'w') as outfile:
    outfile.write(model.to_json())