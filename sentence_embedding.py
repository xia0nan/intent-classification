"""
Text clustering based on TF-IDF and DBSCAN
"""
import re
import math
from string import punctuation

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from gensim.models import KeyedVectors

def string_clean(text):
    """ Basic text cleaning """
    # Remove numbers
    # Remove punctuations
    # Remove single character
    # Stemming
    
    text = text.lower()
    text = re.sub(r'[^a-z0-9] \n', '', text)
    return text

def build_sentence_vec(sentence, model, num_features, index2word_set, idf=None):
    """ Build sentence embedding by averaging word embeddings """
    words = sentence.split()
    # initialize sentence feature vector
    feature_vec = np.zeros((num_features, ), dtype='float32')
    # words counter
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            if idf is None:
                idf_weight = 1
            else:
                idf_weight = idf[word]
            feature_vec = np.add(feature_vec, model[word] * idf_weight)
    if n_words > 0:
        # sentence vector is the average of word embeddings
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


def compute_idf_weights(doc_list):
    """ Compute idf based on all documnets """
    df = {}  # data frequency
    for doc in doc_list:
        words = set(doc.strip().split())
        for word in words:
            if word not in df:
                df[word] = 0.0
            df[word] += 1.0

    idf = {}
    N = len(doc_list)
    for word, count in df.items():
        idf[word] = math.log(float(N) / (1 + count))
    return idf


def get_data(path):
    """ load data from csv """
    # read data
    df = pd.read_csv(path)
    # drop na
    df = df.dropna()
    # drop label with only one query
    df = df.drop(df[df.label == 31].index)
    return df

def load_word2vec():
    # !wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin.gz' # from above
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    return word2vec

def sentence_embedding(method='tfidf', word2vec_model=None):
    # 1. load data
    df = get_data("data.csv")
    data = df.copy()

    # 2. Preprocessing
    data['query'] = data['query'].apply(string_clean)

    # 3. Tokenizing
    def tokenize(s): 
        return " ".join(word_tokenize(s))
    data['query'] = data['query'].apply(tokenize)

    # 4. Get similarity matrix and distnace
    if method == 'tfidf':
        # Use tfidf alone to calculate similarity
        tfidf = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 3)).fit_transform(data['model_text'].values.astype('U'))

        # cosine similarity
        similarity = tfidf * tfidf.T
        distance = 1 - similarity.todense()

    else:
        # Use word2vec
        index2word_set = set(word2vec_model.index2word)
        if method == 'word2vec':
            emb = [build_sentence_vec(data.iloc[i, data.columns.get_loc('query')], model=word2vec_model, num_features=300,
                                  index2word_set=index2word_set) for i in range(len(data))]

        # Use word2vec + idf
        elif method == 'idf-word2vec':
            idf = compute_idf_weights(data['query'])
            emb = [build_sentence_vec(data.iloc[i, data.columns.get_loc('query')], model=word2vec_model, num_features=300,
                               index2word_set=index2word_set, idf=idf) for i in range(len(data))]
        emb = np.array(emb)
        similarity = cosine_similarity(emb)
        distance = 1 - similarity

    return distance

## Download word2vec
# !wget -P /root/input/ -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
# from gensim.models import KeyedVectors
# EMBEDDING_FILE = '/root/input/GoogleNews-vectors-negative300.bin.gz' # from above
# word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)