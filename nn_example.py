import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import feature_extraction, model_selection, metrics
from Filter.Utils import load_spam, load_ham, clean_data
from string import punctuation
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation


from nltk.corpus import stopwords
from sklearn import svm, metrics, feature_extraction, preprocessing

data_files = ['data/nospamaudio.json', 'data/nospamaudio2.json', 'data/spamaudio2.json']
df = pd.DataFrame()
for file in data_files:
    if 'nospam' in file:
        df = df.append(load_ham(file))
    else:
        df = df.append(load_spam(file))

data = clean_data(df)
data = data['audio_title'] + ' ' + data['audio_description']

stop_words = set(stopwords.words('spanish'))
f = feature_extraction.text.CountVectorizer(stop_words=stop_words, max_features=50000)
f.fit(data)
X = f.transform(data)
# labels = np.zeros((len(df['label']), 2))
# i = 0
# for case in df['label']:
#     labels[i, case] = 1
#     i += 1
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, df['label'].values, test_size=0.2)

model = Sequential()
model.add(Dense(32, input_dim=50000))
model.add(Activation('relu'))
model.add(Dense(2, activation='softmax'))

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

one_hot_labels = keras.utils.to_categorical(y_train, num_classes=2)

# Train the model, iterating on the data in batches of 32 samples
history = model.fit(X_train, one_hot_labels, epochs=5, validation_split=0.1, batch_size=300, verbose=2)
print(history)

evaluation_data = model.evaluate(x=X_test, y=keras.utils.to_categorical(y_test, num_classes=2))
print(evaluation_data)

#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
#Compute probabilities
Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
#Plot statistics
print('Analysis of results')
target_names = ['Ham', 'Spam']
print(classification_report(y_test, y_pred,target_names=target_names))
print(confusion_matrix(y_test, y_pred))

Y_pred = model.predict(X)
y_pred = np.argmax(Y_pred, axis=1)

print(df[(df['label'] != y_pred) & (y_pred == 1)])

print(df[df['label'] != y_pred])
