# Starting point
import os
import sys
from pathlib import Path

# General Import
import re
import math
import string
import pickle

import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_distances

import gensim.downloader as api

from nltk.tokenize import word_tokenize

import spacy
from spacy.lang.en.stop_words import STOP_WORDS


PATH_PROJ = Path(__file__).parent

PATH_DATA = PATH_PROJ / 'lib' / 'data'
PATH_DATA.mkdir(parents=True, exist_ok=True)

PATH_MODELS = PATH_PROJ / 'lib' / 'models'
PATH_MODELS.mkdir(parents=True, exist_ok=True)

sys.path.append(str(PATH_PROJ))

# TRAIN
df_train = pd.read_csv('data2.csv')
df_train.dropna(inplace=True)
print(df_train.shape)
print(df_train.head(2))


# rename dataframe
df_train = df_train.rename(columns={'Intent': 'intent', 'Questions': 'query'})
df_train = df_train[['intent', 'query']]
print(df_train.head(2))

# TEST
df_test = pd.read_csv('uat_data_intent.csv')
df_test.dropna(inplace=True)
print(df_test.shape)
print(df_test.head(2))

df_test['correct_google'] = np.where(df_test['User Clicked intent'] == df_test['Google-intent'], 1, 0)

google_accuracy = sum(df_test['correct_google']) / len(df_test['correct_google'])
print(" Google NLU accuracy is {:.1%}".format(google_accuracy))

# rename dataframe
df_test = df_test.rename(columns={'User Clicked intent': 'intent', 'Question': 'query'})
df_test = df_test[['intent', 'query']]



# save the model to disk
model_filename = 'RFClassifier.pkl'
pickle.dump(clf, open(model_filename, 'wb'))

# save vectorizer
pickle.dump(v_lemma, open('TFIDFVectorizer_lemma', 'wb'))
pickle.dump(v_keyword, open('TFIDFVectorizer_keyword', 'wb'))
pickle.dump(v_noun, open('TFIDFVectorizer_noun', 'wb'))
pickle.dump(v_verb, open('TFIDFVectorizer_verb', 'wb'))