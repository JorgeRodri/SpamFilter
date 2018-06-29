import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Input
import datetime
from pymysql import OperationalError
from Filter.Utils import load_data, load_download, load_model, normalize_text
import pickle
from sklearn.externals import joblib
from sklearn import metrics, feature_extraction, model_selection
import json
from time import time
import numpy as np
import pandas as pd
from keras.models import model_from_json
from Filter.Utils import clean_data
from nltk.corpus import stopwords


def main():
    model_save_path = 'model_data/'
    # data = '../data/'
    data = pd.read_pickle('data/FilterDF.pkl')
    credential_file = 'MySQL/credenciales.txt'
    stop_words = set(stopwords.words('spanish') + stopwords.words('English'))
    encoding_dim = 500  # TODO optimize the dimension, anything lower than 5000 is good?

    data = clean_data(data)
    data = data['audio_title'] + ' ' + data['audio_description']

    # data = data.sample(frac=0.5)

    f = feature_extraction.text.CountVectorizer(stop_words=stop_words, min_df=5)  # max_df=0.99, max_features=None)
    f.fit(data)
    X = f.transform(data)
    print(X.shape)
    # X_train, X_test = model_selection.train_test_split(X, test_size=0.2)

    # input_img = Input(shape=(X.shape[1],))
    # encoded = Dense(encoding_dim, activation='relu')(input_img)
    # decoded = Dense(X.shape[1], activation='sigmoid')(encoded)
    # autoencoder = Model(input_img, decoded)
    #
    # encoded_input = Input(shape=(encoding_dim,))
    # decoder_layer = autoencoder.layers[-1]
    # decoder = Model(encoded_input, decoder_layer(encoded_input))
    #
    # print(autoencoder.summary())
    # # autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    #
    # tensorboard = keras.callbacks.TensorBoard(log_dir="../TFlogs/{}".format(time()), histogram_freq=0, batch_size=300,
    #                                           write_grads=True, write_images=True)
    # history = autoencoder.fit(X_train, X_train, epochs=15, validation_data=(X_test, X_test), batch_size=300, verbose=2,
    #                           callbacks=[tensorboard])
    #
    # print(history.history)
    #
    # import matplotlib
    # matplotlib.use('agg')
    # import matplotlib.pyplot as plt
    # # Accuracy plot
    # plt.plot(history.history['val_loss'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train'], loc='upper left')
    # plt.savefig('graphs/' + 'autoencoder_model_accuracy' + '.png')
    # plt.close()
    # # Loss plot
    # plt.plot(history.history['loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train'], loc='upper left')
    # plt.savefig('graphs/' + 'autoencoder_model_loss' + '.png')


if __name__ == '__main__':
    main()
