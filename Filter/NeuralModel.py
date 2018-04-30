import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import datetime
from pymysql import OperationalError
from Filter.Utils import load_data, load_download, load_model, normalize_text
import pickle
from sklearn.externals import joblib
from sklearn import metrics, feature_extraction, model_selection
import json
import numpy as np


class NeuralFilter:
    author = 'Jorge'

    def __init__(self, connection_file, stop_words, classifier, data_path='/data', p=1 / 3, __seed__=42,
                 n_features=5000):
        self.method = 'any'
        self.history = []
        self.number_of_features = n_features

        try:
            with open('update.txt', 'r') as f:
                self.last_update = datetime.datetime.strptime(f.read(), '%Y-%m-%d %H:%M:%S')
        except (IOError, ValueError):
            self.last_update = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.data = load_data(data_path)

        try:
            # ham_df = load_download('nospam', connection_file, self.last_update)
            spam_df = load_download('spam', connection_file, self.last_update)
            downloaded = spam_df  # ham_df.append(spam_df)
            self.data = self.data.append(downloaded)

        except OperationalError as e:
            print(e)
        except json.decoder.JSONDecodeError as e:
            print('Credential json error: ', str(e))

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

    def save_model(self, path):
        # TODO: configure it for NN
        _ = joblib.dump(self.clf, path + 'SVM_model.jblb', compress=0)
        _ = joblib.dump(self.f, path + 'dict_model.jblb', compress=0)
        with open(path + 'SVM_model.pkl', 'wb')as f:
            pickle.dump(self.clf, f)
        with open(path + 'dict_model.pkl', 'wb') as f:
            pickle.dump(self.f, f)

    def get_neural(self, layers, output_dim=2, activation='relu', dropout_rate=None):
        model = Sequential()
        model.add(Dense(layers[0], input_dim=self.number_of_features))
        model.add(Activation(activation))
        if dropout_rate:
            model.add(Dropout(dropout_rate))
        if len(layers) > 1:
            for i in range(1, len(layers)):
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
        print(self.clf.evaluate(self.X_test, self.y_test, verbose=2))
        return metrics.confusion_matrix(np.argmax(self.y_test, 1), np.argmax(self.clf.predict(self.X_test), 1))

    def predict(self, strings):
        word_vector = self.f.transform(strings)
        return self.clf.predict(word_vector)


def main():
    model_save_path = '../model_data/'
    print(datetime.datetime.now())
    global SpamFilter
    from nltk.corpus import stopwords
    data_path = '../data/'
    credential_file = '../MySQL/credenciales.txt'
    stop_words = set(stopwords.words('spanish') + stopwords.words('English'))
    model = Sequential()
    model.add(Dense(32, input_dim=50000))
    model.add(Activation('relu'))
    model.add(Dense(2, activation='softmax'))
    SpamFilter = NeuralFilter(credential_file, stop_words, model, data_path, n_features=50000, p=3/4)
    SpamFilter.get_features()
    SpamFilter.X_train, SpamFilter.X_test, SpamFilter.y_train, SpamFilter.y_test = SpamFilter.get_test(0.3)
    SpamFilter.get_neural([32, 32], dropout_rate=0.2)
    print(SpamFilter.clf.summary())
    SpamFilter.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    print(SpamFilter.fit(epochs=10, validation_split=0.1, batch_size=300, verbose=2))


if __name__ == '__main__':
    main()
