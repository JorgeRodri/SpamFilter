from sklearn import feature_extraction, model_selection, metrics
from sklearn.externals import joblib
from pymysql import OperationalError
from Filter.Utils import load_model, load_download, load_data, normalize_text
import datetime, pickle


class FilterModel:
    author = 'Jorge'

    def __init__(self, connection_file, stopwords, classifier, data_path='data/', p=1/3, __seed__=42, n_features=None):
        self.method = 'any'

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

        try:
            self.X = self.f.transform(self.data['audio_title'] + ' ' + self.data['audio_description'])

        except AttributeError:
            self.f = feature_extraction.text.CountVectorizer(stop_words=stopwords, max_features=n_features)
            self.X = self.f.fit_transform(self.data['audio_title'] + ' ' + self.data['audio_description'])
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(self.X,
                                                                                                self.data['label'],
                                                                                                test_size=p,
                                                                                                random_state=__seed__)
        self.clf = load_model(classifier)

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
        return model_selection.train_test_split(self.X, self.data['label'], test_size=p, random_state=__seed__)

    def save_model(self, path):
        _ = joblib.dump(self.clf, path + 'SVM_model.jblb', compress=0)
        _ = joblib.dump(self.f, path + 'dict_model.jblb', compress=0)
        with open(path + 'SVM_model.pkl', 'wb')as f:
            pickle.dump(self.clf, f)
        with open(path + 'dict_model.pkl', 'wb') as f:
            pickle.dump(self.f, f)

    def train(self):
        self.clf.fit(self.X_train, self.y_train)
        return metrics.confusion_matrix(self.y_test, self.clf.predict(self.X_test))

    def predict(self, strings):
        word_vector = self.f.transform(strings)
        try:
            return self.clf.predict_proba(word_vector)
        except AttributeError:
            return self.clf.predict(word_vector)
