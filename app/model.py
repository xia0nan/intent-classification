import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

#%% Constants
PATH_PROJ = Path(__file__).parent
PATH_DATA = PATH_PROJ / 'lib' / 'data'
PATH_MODELS = PATH_PROJ / 'lib' / 'models'
sys.path.append(str(PATH_PROJ))

class IntentClassifier(object):
    """ Intent Classification model """
    def __init__(self):
        """ Intent classifier take one intput and return prediction
        Attributes:
            clf: sklearn classifier model
            vectorizor: TFIDF vectorizer or similar
        """
        self.clf = RandomForestClassifier(max_depth=50, n_estimators=1000)
        self.v_noun = TfidfVectorizer()
        self.v_verb = TfidfVectorizer()
        self.v_keyword = TfidfVectorizer()
        self.v_lemma = TfidfVectorizer()
        self.intent_list: list
        self.dict_cluster: dict
        self.word2vec: dict
        self.idf: dict
        self.intent2index: dict
        self.countvector_cols = ['lemma', 'keyword', 'noun', 'verb']
        self.top_clusters_cols = ['clusters_1', 'clusters_2', 'clusters_3']
        self.feature_cols = self.countvector_cols + self.top_clusters_cols

    def vectorizer_fit(self, X_train):
        x_train_noun = self.v_noun.fit_transform(X_train['noun'])
        x_train_verb = self.v_verb.fit_transform(X_train['verb'])
        x_train_keyword = self.v_keyword.fit_transform(X_train['keyword'])
        x_train_lemma = self.v_lemma.fit_transform(X_train['lemma'])
        return x_train_noun, x_train_verb, x_train_keyword, x_train_lemma

    def combine_features(self, X, x_noun, x_verb, x_keyword, x_lemma):
        """ get RF classifier input for both X_train and X_test """
        x_combined = hstack((x_lemma,
                            x_keyword,
                            x_noun,
                            x_verb,
                            X[self.top_clusters_cols].values),
                            format='csr')
        x_combined_columns= (self.v_lemma.get_feature_names()+
                            self.v_keyword.get_feature_names()+
                            self.v_noun.get_feature_names()+
                            self.v_verb.get_feature_names()+
                            self.top_clusters_cols)
        x_combined = pd.DataFrame(x_combined.toarray())
        x_combined.columns = x_combined_columns
        return x_combined

    def vectorizer_transform(self, X_test):
        """Transform the text data to a sparse TFIDF matrix
        """
        x_test_lemma = self.v_lemma.transform(X_test['lemma'])
        x_test_keyword = self.v_keyword.transform(X_test['keyword'])
        x_test_noun = self.v_noun.transform(X_test['noun'])
        x_test_verb = self.v_verb.transform(X_test['verb'])
        return x_test_lemma, x_test_keyword, x_test_noun, x_test_verb

    def train(self, x_train_combined, y_train):
        """Trains the classifier to associate the label with the sparse matrix
        """
        self.clf.fit(x_train_combined, y_train)

    def predict_proba(self, x_test_combined):
        """Returns probability for each prediction in a numpy array
        """
        y_proba = self.clf.predict_proba(x_test_combined)
        return y_proba[:, 1]

    def predict(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.clf.predict(X)
        return y_pred
    
    def pickle_vectorizer(self):
        """Saves the trained vectorizer for future use.
        """
        with open(str(PATH_MODELS/'TFIDFVectorizer_lemma.pkl'), 'wb') as f:
            pickle.dump(self.v_lemma, f)
        with open(str(PATH_MODELS/'TFIDFVectorizer_keyword.pkl'), 'wb') as f:
            pickle.dump(self.v_keyword, f)
        with open(str(PATH_MODELS/'TFIDFVectorizer_noun.pkl'), 'wb') as f:
            pickle.dump(self.v_noun, f)
        with open(str(PATH_MODELS/'TFIDFVectorizer_verb.pkl'), 'wb') as f:
            pickle.dump(self.v_verb, f)

    def pickle_clf(self, model_filename='RFClassifier.pkl'):
        """Saves the trained classifier for future use.
        """
        with open(str(PATH_MODELS/model_filename), 'wb') as f:
            pickle.dump(self.clf, f)        


