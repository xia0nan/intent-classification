"""
Text clustering based on TF-IDF and word2vec
"""
import re
import math

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors

def string_clean(text):
    """ Basic text cleaning: lowercase and remove special characters """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
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


def get_data(path="data.csv"):
    """ load data from csv """
    # read data, default "./data.csv"
    df = pd.read_csv(path)
    df = df.dropna()
    # drop label with only one query (through EDA)
    df = df.drop(df[df.label == 31].index)
    return df

def load_word2vec():
    # Download word2vec
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

    # 3. Tokenizing using NLTK
    def tokenize(s): 
        return " ".join(word_tokenize(s))
    data['query'] = data['query'].apply(tokenize)

    # 4. Get similarity matrix and distnace
    if method == 'tfidf':
        # Use tfidf alone to calculate similarity
        tfidf = TfidfVectorizer(lowercase=True, 
                                stop_words="english", 
                                ngram_range=(1, 3)).fit_transform(data['query'].values.astype('U'))

        # cosine similarity
        similarity = tfidf * tfidf.T
        distance = 1 - similarity.todense()

    else:
        # Use word2vec
        index2word_set = set(word2vec_model.index2word)
        if method == 'word2vec':
            emb = [build_sentence_vec(data.iloc[i, data.columns.get_loc('query')], 
                                      model=word2vec_model, num_features=300,
                                      index2word_set=index2word_set) for i in range(len(data))]

        # Use word2vec + idf
        elif method == 'idf-word2vec':
            idf = compute_idf_weights(data['query'])
            emb = [build_sentence_vec(data.iloc[i, data.columns.get_loc('query')], 
                                      model=word2vec_model, num_features=300,
                                      index2word_set=index2word_set, 
                                      idf=idf) for i in range(len(data))]
        emb = np.array(emb)
        similarity = cosine_similarity(emb)
        distance = 1 - similarity

    return distance

def test_word2vec_idf():
    sen = "test cat and dog"
    
    idf = {}
    idf["test"] = 1/3.
    idf["cat"] = 1/3.
    idf["and"] = 0.
    idf["dog"] = 1/3.
    
    words = sen.split()
    words = [word for word in words if word in word2vec.vocab]
    
    idf_series = np.array([idf[word] for word in words])
    print(idf_series.reshape(1, -1).shape)
    idf_series = idf_series.reshape(1, -1).shape
    
    print(word2vec[words].shape)
    result = np.matmul(idf_series.reshape(1, -1), word2vec[words]).reshape(-1)
    print(result.shape)
    
    emb1 = word2vec[words].mean(axis=0)
    emb2 = result
    
    assert emb1 == emb2
    return result

def get_sentence_vec(sentence, word2vec, idf=None):
    words = sentence.split()
    words = [word for word in words if word in word2vec.vocab]
    # use mean if no idf provided
    if idf is None:
        emb = word2vec[words].mean(axis=0)
    else:
        # get all idf of words
        idf_series = np.array([idf[word] for word in words])
        # change shape to 1 x num_of_words
        idf_series = idf_series.reshape(1, -1)
        # use matrix multiplication to get weighted word vector sum for sentence embeddings
        emb = np.matmul(idf_series, word2vec[words]).reshape(-1)
    return emb

# TODO: 
# 1. def cluster_centre()
# 2. def test(text)
# 3. pipeline?
