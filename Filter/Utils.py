# -*- coding: utf-8 -*-
import abc
import pandas as pd
from sklearn.externals import joblib
from sklearn import svm, metrics
import numpy as np
from MySQL.connection import getConnection
import os
from io import open
from bson import json_util
from string import punctuation
import pymysql
# from main import SpamFilter
import multiprocessing


def un_punctuate(text, p_):
    for char in p_:
        # print(char)
        text = text.replace(char, ' ' + char + ' ')  # ' ' + char + ' ')
    return text


def normalize_text(text):
    norm_text = text.lower()
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', '')
    # Pad punctuation with spaces on both sides
    non_words = list(punctuation)
    non_words = ['*']
    non_words.extend([u'¿', u'¡'])
    non_words.extend(map(str, range(10)))
    non_words.extend(['\n', '\r', '\t'])
    norm_text = un_punctuate(norm_text, non_words)
    return norm_text


def load_model(classifier, nip=None):
    if type(classifier) == str:
        model = joblib.load(classifier)
    elif type(classifier) == abc.ABCMeta:
        model = classifier
    elif type(classifier) == svm.SVC:
        model = classifier
    elif classifier is None:
        raise AttributeError
    elif type(classifier) is nip:
        model = classifier
    else:
        raise AttributeError
    return model


def load_download(label, connection_credentials_files, last_update):
    d = {'spam': 1, 'nospam': 0}
    with open(connection_credentials_files, 'r') as f:
        connection_credentials = json_util.loads(f.read())
    try:
        connection = getConnection(connection_credentials)
        query = 'SELECT * FROM ivoox.{0}audio WHERE {0}audio_insertdate > "{1}"'
        df = pd.read_sql(query.format(label, last_update), connection)
        df = df[['spamaudio_description', 'spamaudio_duration', 'spamaudio_fksubcategory',
                 'spamaudio_fkaudio', 'spamaudio_title', 'spamaudio_insertdate']]
        df.columns = ['audio_description', 'audio_duration', 'audio_fksubcategory',
                      'audio_fkaudio', 'audio_title', 'insertdate']
        df['label'] = d[label]
        return df
    except pymysql.DataError as e:
        print("DataError")
        print(e)
    except pymysql.InternalError as e:
        print("InternalError")
        print(e)
    except pymysql.IntegrityError as e:
        print("IntegrityError")
        print(e)
    except pymysql.OperationalError as e:
        print("OperationalError")
        print(e)
    except pymysql.NotSupportedError as e:
        print("NotSupportedError")
        print(e)
    except pymysql.ProgrammingError as e:
        print("ProgrammingError")
        print(e)
    except Exception as e:
        print(e)
        print("Unknown error occurred")
    return pd.DataFrame()


def load_spam(file_name):
    with open(file_name, 'r', encoding='utf8') as f:
        data = json_util.loads(f.read())
    df = pd.DataFrame(data)
    df = df[['spamaudio_description', 'spamaudio_duration', 'spamaudio_fksubcategory',
             'spamaudio_fkaudio', 'spamaudio_title']]
    df.columns = ['audio_description', 'audio_duration', 'audio_fksubcategory',
                  'audio_fkaudio', 'audio_title']

    df['label'] = 1
    return df


def load_ham(file_name):
    with open(file_name, 'r', encoding='utf8') as fly:
        data = json_util.loads(fly.read())
    df = pd.DataFrame(data)

    df.columns = ['audio_description', 'audio_duration', 'audio_fksubcategory',
                  'audio_fkaudio', 'audio_title']

    df['label'] = 0
    return df


def load_data(path):
    data_files = os.listdir(path)
    data_files = [path + i for i in data_files]
    for file_name in data_files:
        if 'nospam' in file_name:
            try:
                df = df.append(load_ham(file_name))
            except NameError:
                df = load_ham(file_name)

        elif 'spam' in file_name:
            # continue
            try:
                df = df.append(load_spam(file_name))
            except NameError:
                df = load_spam(file_name)
        else:
            continue
    return df


def select_best_model(self, classifier, params, **kwargs):
    score_train = np.zeros(len(params))
    score_test = np.zeros(len(params))
    recall_test = np.zeros(len(params))
    precision_test = np.zeros(len(params))
    count = 0
    for value in params:
        clf = classifier(value, **kwargs)
        clf.fit(self.X_train, self.y_train)
        score_train[count] = clf.score(self.X_train, self.y_train)
        score_test[count] = clf.score(self.X_test, self.y_test)
        recall_test[count] = metrics.recall_score(self.y_test, clf.predict(self.X_test))
        precision_test[count] = metrics.precision_score(self.y_test, clf.predict(self.X_test))
        count = count + 1

    matrix = np.matrix(np.c_[params, score_train, score_test, recall_test, precision_test])
    models = pd.DataFrame(data=matrix,
                          columns=['C', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
    if models['Test Precision'].max() == 1:
        best_index = models[models['Test Precision'] == 1]['Test Accuracy'].idxmax()
    else:
        best_index = models['Test Precision'].idxmax()
    self.model = classifier(params[best_index], **kwargs)
    self.model.fit(self.X_train, self.y_train)
    return metrics.confusion_matrix(self.y_test, self.model.predict(self.X_test))


def parallel_optimizer(c, x_train, y_train, x_test, y_test):
    svc = svm.SVC(C=c)
    svc.fit(x_train, y_train)
    score_train = svc.score(x_train, y_train)
    score_test = svc.score(x_test, y_test)
    recall_test = metrics.recall_score(y_test, svc.predict(x_test))
    precision_test = metrics.precision_score(y_test, svc.predict(x_test))
    return c, score_train, score_test, recall_test, precision_test


def ImmutableMultiDict_transform(d):
    d_dict = d.to_dict()
    for k, v in d_dict.iteritems():
        if v == 'True' or v == 'true' or v == '1':
            d_dict[k] = True
        elif v == 'False' or v == 'false' or v == '0':
            d_dict[k] = False
        else:
            try:
                d_dict[k] = float(v)
            except ValueError:
                d_dict[k] = v
    return d_dict


def clean_data(data):
    data['audio_description'] = data['audio_description'].apply(lambda x: x.replace('\n', ''))
    data['audio_description'] = data['audio_description'].apply(lambda x: x.replace('\r', ''))
    data['audio_description'] = data['audio_description'].apply(lambda x: x.replace('\t', ''))
    data['audio_description'] = data['audio_description'].apply(lambda x: x.replace('*', ''))
    data['audio_description'] = data['audio_description'].apply(normalize_text)

    data['audio_title'] = data['audio_title'].apply(lambda x: x.replace('\n', ''))
    data['audio_title'] = data['audio_title'].apply(lambda x: x.replace('\r', ''))
    data['audio_title'] = data['audio_title'].apply(lambda x: x.replace('\t', ''))
    data['audio_title'] = data['audio_title'].apply(lambda x: x.replace('*', ''))
    data['audio_title'] = data['audio_title'].apply(normalize_text)
    return data