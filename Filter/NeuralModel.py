import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
import datetime
from pymysql import OperationalError
from Filter.Utils import load_data, load_download, load_model, normalize_text, clean_data
import pickle
from sklearn.externals import joblib
from sklearn import metrics, feature_extraction, model_selection
import json
from time import time
import numpy as np
import pandas as pd
from keras.models import model_from_json


class NeuralFilter:
    author = 'Jorge'

    def __init__(self, connection_file, stop_words, classifier, data=None, p=1 / 3, __seed__=42,
                 n_features=5000):
        self.method = 'any'
        self.history = []
        self.score = []
        self.number_of_features = n_features

        try:
            with open('update.txt', 'r') as f:
                self.last_update = datetime.datetime.strptime(f.read(), '%Y-%m-%d %H:%M:%S')
        except (IOError, ValueError):
            self.last_update = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if type(data) == str:
            self.data = load_data(data)
            try:
                # ham_df = load_download('nospam', connection_file, self.last_update)
                spam_df = load_download('spam', connection_file, self.last_update)
                downloaded = spam_df  # ham_df.append(spam_df)
                self.data = self.data.append(downloaded)

            except OperationalError as e:
                print(e)
            except json.decoder.JSONDecodeError as e:
                print('Credential json error: ', str(e))

        else:
            self.data = data

        try:
            self.X = self.f.transform(self.data['audio_title'] + ' ' + self.data['audio_description'])

        except AttributeError:
            self.f = feature_extraction.text.CountVectorizer(stop_words=stop_words, max_features=n_features)
            self.X = self.f.fit_transform(self.data['audio_title'] + ' ' + self.data['audio_description'])
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(self.X,
                                                                                                keras.utils.
                                                                                                to_categorical(
                                                                                                    self.data['label'],
                                                                                                    num_classes=2),
                                                                                                test_size=p,
                                                                                                random_state=__seed__)
        self.clf = load_model(classifier, keras.models.Sequential)

    def get_features(self):
        self.X = self.f.transform(self.data['audio_title'] + ' ' + self.data['audio_description'])
        return self.X

    def clean_data(self):
        self.data['audio_description'] = self.data['audio_description'].apply(lambda x: x.replace('\n', ''))
        self.data['audio_description'] = self.data['audio_description'].apply(lambda x: x.replace('\r', ''))
        self.data['audio_description'] = self.data['audio_description'].apply(lambda x: x.replace('\t', ''))
        self.data['audio_description'] = self.data['audio_description'].apply(lambda x: x.replace('*', ''))
        self.data['audio_description'] = self.data['audio_description'].apply(normalize_text)

        self.data['audio_title'] = self.data['audio_title'].apply(lambda x: x.replace('\n', ''))
        self.data['audio_title'] = self.data['audio_title'].apply(lambda x: x.replace('\r', ''))
        self.data['audio_title'] = self.data['audio_title'].apply(lambda x: x.replace('\t', ''))
        self.data['audio_title'] = self.data['audio_title'].apply(lambda x: x.replace('*', ''))
        self.data['audio_title'] = self.data['audio_title'].apply(normalize_text)

    def get_test(self, p, __seed__=42):
        return model_selection.train_test_split(self.X, keras.utils.to_categorical(self.data['label'], num_classes=2),
                                                test_size=p, random_state=__seed__)

    def save_model(self, path_arch, path_weights):
        # TODO: configure it for NN
        # Saving model and weights
        nn_json = self.clf.to_json()
        with open(path_arch, 'w') as json_file:
            json_file.write(nn_json)
        self.clf.save_weights('../weights/' + path_weights, overwrite=True)

    def load_model(self, path_arch, path_weights):
        # Loading model and weights
        with open(path_arch, 'r') as f:
            nn_json = f.read()
        nn = model_from_json(nn_json)
        nn.load_weights('../weights/' + path_weights)
        self.clf = nn

    def get_neural(self, layers, output_dim=2, num_features=None, activation='relu', dropout_rate=None, batch_normalization=False):
        model = Sequential()
        if num_features is None:
            num_features = self.number_of_features
        model.add(Dense(layers[0], input_dim=num_features))
        model.add(Activation(activation))
        if dropout_rate:
            model.add(Dropout(dropout_rate))
        if len(layers) > 1:
            for i in range(1, len(layers)):
                if batch_normalization:
                    model.add(BatchNormalization())
                model.add(Dense(layers[i]))
                model.add(Activation(activation))
                if dropout_rate:
                    model.add(Dropout(dropout_rate))
        model.add(Dense(output_dim, activation='softmax'))
        self.clf = model
        return self.clf

    def compile(self, **compile_args):
        self.clf.compile(**compile_args)

    def fit(self, **fit_args):
        self.history = self.clf.fit(self.X_train, self.y_train, **fit_args)
        self.score = self.clf.evaluate(self.X_test, self.y_test, verbose=2)
        print(self.score)
        return metrics.confusion_matrix(np.argmax(self.y_test, 1), np.argmax(self.clf.predict(self.X_test), 1))

    def predict(self, strings):
        word_vector = self.f.transform(strings)
        return self.clf.predict(word_vector)


def main():
    model_save_path = '../model_data/'
    from nltk.corpus import stopwords
    # data = '../data/'
    data = pd.read_pickle('../data/FilterDF.pkl')
    credential_file = '../MySQL/credenciales.txt'
    stop_words = set(stopwords.words('spanish') + stopwords.words('English'))
    model = Sequential()
    model.add(Dense(32, input_dim=300000))
    model.add(Activation('elu'))  # TODO try elu and variable number of
    model.add(Dense(2, activation='softmax'))
    SpamFilter = NeuralFilter(credential_file, stop_words, model, data, n_features=300000, __seed__=155)

    data = clean_data(data)
    data = data['audio_title'] + ' ' + data['audio_description']
    SpamFilter.f = feature_extraction.text.CountVectorizer(stop_words=stop_words,
                                                           min_df=5)
                                                           # max_df=0.99,
                                                           # max_features=None)

    SpamFilter.f.fit(data)
    SpamFilter.X = SpamFilter.f.transform(data)
    print(SpamFilter.X.shape)

    SpamFilter.get_features()
    SpamFilter.X_train, SpamFilter.X_test, SpamFilter.y_train, SpamFilter.y_test = SpamFilter.get_test(1/3)
    SpamFilter.get_neural([52, 52], num_features=SpamFilter.X.shape[1], dropout_rate=0.2, batch_normalization=True)
    print(SpamFilter.clf.summary())
    SpamFilter.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    tensorboard = keras.callbacks.TensorBoard(log_dir="../TFlogs/{}".format(time()), histogram_freq=0, batch_size=300,
                                              write_grads=True, write_images=True)
    print(SpamFilter.fit(epochs=3, validation_split=0.1, batch_size=300, verbose=2, callbacks=[tensorboard]))
    print(SpamFilter.history.history)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Accuracy plot
    plt.plot(SpamFilter.history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('../graphs/' + 'model_accuracy' + '.png')
    plt.close()
    # Loss plot
    plt.plot(SpamFilter.history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig('../graphs/' + 'model_loss' + '.png')


if __name__ == '__main__':
    main()
