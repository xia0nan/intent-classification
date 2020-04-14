#%% Import
# Default
import os
import sys
from pathlib import Path
import json
import math
import pickle

from timeit import default_timer as timer
# Third party
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, hstack
# NLTK
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
# Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
# Gensim for word embeddings
from gensim.models import KeyedVectors
# Extra
import plotly.express as px
import xlrd

#%% Constants
PATH_PROJ = Path.home() / 'projects/intent-classification'
PATH_DATA = PATH_PROJ

#%% Functions
def get_data(path=PATH_DATA/'data.csv'):
    """ load data from csv """
    df = pd.read_csv(path, usecols=['intent', 'query'])
    df = df.dropna().drop_duplicates()
    # From EDA, the intent only have 1 example
    df = df.drop(df[df.intent == 'Late fee waiver for credit card'].index)
    return df

import re
def clean_text(text):
    """ Basic text cleaning """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

from nltk.tokenize import word_tokenize
def nltk_tokenize(text):
    # import nltk
    # nltk.download('punkt')
    return ' '.join(word_tokenize(text))

import spacy
class SpacyPipeline():
    """ Utilize Spacy Pipeline """
    def __init__(self, word_vector='en_core_web_lg'):
        self.nlp = spacy(word_vector)

    def tokenize(self, text):
        """ Return tokenized text 
            "hello world!" -> "hello", "world", "!"
        """
        return self.nlp(text)

    def get_word_embed(self, text):
        """ Get individual word embedding: result.shape = (num_of_words, 300)"""
        with self.nlp.disable_pipes():
            vectors = np.array([token.vector for token in self.nlp(text)])
        return vectors

    def get_doc_embed(self, doc):
        """ Get sentence embeddings based on average of word embeddings of each sentence
            input: pd.Series of sentences
            result.shape = (num_of_sentences, 300)
        """
        with self.nlp.disable_pipes():
            doc_vectors = np.array([self.nlp(text).vector for text in doc])
        return doc_vectors

    