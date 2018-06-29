import os
import numpy as np
import pandas as pd
from Filter.Utils import clean_data
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from nltk.corpus import stopwords
from sklearn import feature_extraction, model_selection
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


data = pd.read_pickle('data/FilterDF.pkl')
y = data.loc[:, 'label']
data = clean_data(data)
data = data['audio_title'] + ' ' + data['audio_description']

stop_words = set(stopwords.words('spanish'))
f = feature_extraction.text.CountVectorizer(stop_words=stop_words, min_df=5)
f.fit(data)
X = f.transform(data)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y.values, test_size=0.2)

model = Sequential()
model.add(Dense(32, input_dim=X.shape[1], kernel_regularizer=keras.regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
# model.add(Dense(32, kernel_regularizer=keras.regularizers.l2(0.01)))
# model.add(Dropout(0.25))
model.add(Activation('relu'))
model.add(Dense(2, activation='softmax'))

if __name__ == '__main__':
    # For a binary classification problem
    model.compile(optimizer='sgd',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    one_hot_labels = keras.utils.to_categorical(y_train, num_classes=2)

    # Train the model, iterating on the data in batches of 32 samples
    history = model.fit(X_train, one_hot_labels, epochs=7, validation_split=0.1, batch_size=300, verbose=2)
    print(history)

    evaluation_data = model.evaluate(x=X_test, y=keras.utils.to_categorical(y_test, num_classes=2), verbose=2)
    print(evaluation_data)

    # Confusion Matrix
    from sklearn.metrics import classification_report, confusion_matrix
    # Compute probabilities
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    # Plot statistics
    print('Analysis of results')
    target_names = ['Ham', 'Spam']
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred))
