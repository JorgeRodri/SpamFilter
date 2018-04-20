from flask import Flask
from Filter.MainModel import FilterModel
from sklearn.externals import joblib
from sklearn import svm, metrics, feature_extraction
from flask import request
from io import open
from nltk.corpus import stopwords
import os, multiprocessing, pymysql, datetime, sys, pickle
from Filter.html import body
from Filter.Utils import parallel_optimizer, ImmutableMultiDict_transform, load_download, load_data, normalize_text
import numpy as np
import pandas as pd
from functools import partial


app = Flask(__name__)


def parallel_runs(data_list):
    processors = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=processors)
    partial_c = partial(parallel_optimizer,
                        x_train=SpamFilter.X_train,
                        y_train=SpamFilter.y_train,
                        x_test=SpamFilter.X_test,
                        y_test=SpamFilter.y_test)
    result_list = pool.map(partial_c, data_list)
    return np.matrix(result_list)


@app.route('/bowall')
def get_bow_all():

    return 1


@app.route('/train', methods=['GET'])
def train():
    try:
        clf = svm.SVC(**ImmutableMultiDict_transform(request.args))
    except (ValueError, TypeError) as e:
        return str(-1)
    SpamFilter.clf = clf

    m = SpamFilter.train()
    SpamFilter.save_model('')
    return body.format(m[0, 0], m[0, 1], m[1, 0], m[1, 1])


@app.route('/load')
def load_corpus():
    # TODO load Data again
    SpamFilter.data = load_data(data_path)
    SpamFilter.last_update = '2016-02-28 12:28:29'
    try:
        # ham_df = load_download('nospam', connection_file, self.last_update)
        spam_df = load_download('spam', credential_file, SpamFilter.last_update)
        downloaded = spam_df  # ham_df.append(spam_df)
        SpamFilter.data = SpamFilter.data.append(downloaded)

    except pymysql.OperationalError as e:
        print(e)
    print(len(SpamFilter.data))
    # SpamFilter.clean_data()
    return str(10)


@app.route('/trainbow')
def train_bow():
    # stop_words = set(stopwords.words('spanish') + stopwords.words('English'))
    stop_words = []
    try:
        bow = feature_extraction.text.CountVectorizer(stop_words=stop_words, **ImmutableMultiDict_transform(request.args))
    except (ValueError, TypeError) as e:
        bow = feature_extraction.text.CountVectorizer(stop_words=stop_words)
    SpamFilter.f = bow
    # print(SpamFilter.data.tail())
    SpamFilter.f.fit(SpamFilter.data['audio_title'] + ' ' + SpamFilter.data['audio_description'])
    return str(10)


@app.route('/transform')
def transform_data():
    SpamFilter.X = SpamFilter.get_features()
    print(SpamFilter.X.shape)
    SpamFilter.X_train, SpamFilter.X_test, SpamFilter.y_train, SpamFilter.y_test = SpamFilter.get_test(0.3)
    return str(10)


@app.route('/predict', methods=['GET'])
def predict():
    path = ''  # '/var/www/new_ivoox.com/svm_spam_files/'
    try:
        desc = request.args.get('file')
    except TypeError as e:
        return str(-1)
    try:
        with open(path + desc, 'r', encoding='utf8') as f:
            string = f.read()
    except IOError as e:
        return str(-1)
    except TypeError as e:
        return str(-1)
    try:
        return '%d' % SpamFilter.clf.predict(SpamFilter.f.transform([string]))[0]
    except:
        return str(sys.exc_info())


@app.route('/params')
def get_params():
    return str(SpamFilter.clf.get_params()) + '\n\n' + str(SpamFilter.f.get_params())


@app.route('/dummy')
def simple():
    m = metrics.confusion_matrix(SpamFilter.y_test, SpamFilter.clf.predict(SpamFilter.X_test))
    return body.format(m[0, 0], m[0, 1], m[1, 0], m[1, 1])


@app.route('/get_best')
def train_all():
    print(datetime.datetime.now())
    r = parallel_runs(np.concatenate((np.arange(10, 100, 10), np.arange(100, 50, 400))))

    models = pd.DataFrame(data=r,
                          columns=['C', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
    if models['Test Precision'].max() == 1:
        best_index = models[models['Test Precision'] == 1]['Test Accuracy'].idxmax()
    else:
        best_index = models['Test Precision'].idxmax()
    SpamFilter.clf = svm.SVC(C=models['C'][best_index])
    SpamFilter.clf.fit(SpamFilter.X_train, SpamFilter.y_train)

    SpamFilter.save_model('model_data/best_')
    print(datetime.datetime.now())
    m = metrics.confusion_matrix(SpamFilter.y_test, SpamFilter.clf.predict(SpamFilter.X_test))
    return body.format(m[0, 0], m[0, 1], m[1, 0], m[1, 1])


@app.route('/save')
def save_model():
    try:
        SpamFilter.save_model('model_data/')
    except NameError:
        pass
    with open('dict_model.pkl', 'wb') as fili:
        pickle.dump(SpamFilter.f, fili)
    with open('SVM_model.pkl', 'wb') as fili:
        pickle.dump(SpamFilter.clf, fili)
    return str(10)


@app.errorhandler(404)
def page_not_found(e):
    return str(-1)


if __name__ == '__main__':
    model_save_path = 'model_data/'
    global SpamFilter
    data_path = 'data/'
    credential_file = 'credenciales.txt'
    stop_words = set(stopwords.words('spanish') + stopwords.words('English'))
    model = 'SVM_model.pkl'  # svm.SVC(C=90)
    # SpamFilter = FilterModel(credential_file, stop_words, model, data_path)
    SpamFilter = FilterModel(credential_file, [], model, data_path)
    try:
        # SpamFilter.f = joblib.load(model_save_path + 'dict_model.pkl')
        with open('best_dict_model.pkl', 'rb')as fili:
            SpamFilter.f = pickle.load(fili)
        # SpamFilter.clf = joblib.load(model_save_path + 'SVM_model.pkl')
        with open('best_SVM_model.pkl', 'rb') as fili:
            SpamFilter.clf = pickle.load(fili)
    except IOError:
        print('loading error, file not found')
    except KeyError:
        print('loading error. Pickle and Joblib incompatible')
    if len(sys.argv) == 3:
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]
        app.run(host=arg1, port=int(arg2))
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        app.run(host='127.0.0.1', port=int(arg))
    else:
        app.run()
