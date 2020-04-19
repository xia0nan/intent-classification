#%% Import
# Default
import os
import sys
from pathlib import Path
import json
import math
import pickle
import re
import string

# Third party
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import spacy

# NLTK
from nltk.tokenize import word_tokenize
# Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_distances

#%% Constants
PATH_PROJ = Path(__file__).parent
PATH_DATA = PATH_PROJ / 'lib' / 'data'
PATH_MODELS = PATH_PROJ / 'lib' / 'models'
sys.path.append(str(PATH_PROJ))

def clean_text(text):
    """ Basic text cleaning
        
        1. lowercase
        2. remove special characters
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def nltk_tokenize(text):
    """ tokenize text using NLTK and join back as sentence"""
    # import nltk
    # nltk.download('punkt')
    return ' '.join(word_tokenize(text))


class SpacyTokenizer(object):
    """ """
    def __init__(self):
        # Create our list of punctuation marks
        self.punctuations = string.punctuation

        # Create our list of stopwords
        self.nlp = spacy.load('en_core_web_lg')
        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS

    # Creating our tokenizer function
    def tokenize(self, sentence):
        # Creating our token object, which is used to create documents with linguistic annotations.
        mytokens = self.nlp(sentence)

        # Lemmatizing each token and converting each token into lowercase
        mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

        # Removing stop words
        mytokens = [ word for word in mytokens if word not in self.stop_words and word not in self.punctuations ]

        # return preprocessed list of tokens
        return mytokens

def get_idf_TfidfVectorizer(sentences):
    """ Get idf dictionary by using TfidfVectorizer
    
    Args:
        sentences (list): list of input sentences (str)

    Returns:
        idf (dict): idf[word] = inverse document frequency of that word in all training queries
    """

    # use customized Spacy tokenizer
    vectorizer = TfidfVectorizer(tokenizer=nltk_tokenize)
    vectorizer.fit(sentences)
    # TODO: normalize the idf weights
    idf = {k:vectorizer.idf_[v] for k,v in vectorizer.vocabulary_.items()}
    return idf

def get_sentence_vec(sentence, word2vec, idf=None):
    """ Get embedding of sentence by using word2vec embedding of words
    
    If idf is provided, the sentence is the weighted embedding by
        SUM( embedding[word] x idf[word] )
    
    Args:
        sentence (str): input sentence
        word2vec (dict): loaded word2vec model from Gensim
        idf (dict, optional): inverse document frequency of words in all queries

    Returns:
        emb (np.array): 300-dimentions embedding of sentence
    """
    words = sentence.split()
    words = [word for word in words if word in word2vec.vocab]
    
    # if no word in word2vec vocab, return 0x300 embedding
    if len(words)==0:
        return np.zeros((300,), dtype='float32')
    
    # use mean if no idf provided
    if idf is None:
        emb = word2vec[words].mean(axis=0)
    else:
        # get all idf of words, if new word is not in idf, assign 0.0 weights
        idf_series = np.array([idf.get(word, 0.0) for word in words])
        # change shape to 1 x num_of_words
        idf_series = idf_series.reshape(1, -1)
        # use matrix multiplication to get weighted word vector sum for sentence embeddings
        emb = np.matmul(idf_series, word2vec[words]).reshape(-1)
    return emb


def get_sentences_centre(sentences, word2vec, idf=None, num_features=300):
    """ Get sentences centre by averaging all embeddings of sentences in a list
    
    Depends on function get_sentence_vec()
    
    Args:
        sentence (list): list of input sentences (str)
        word2vec (dict): loaded word2vec model from Gensim
        idf (dict, optional): inverse document frequency of words in all queries

    Returns:
        emb (np.array): 300-dimentions embedding of sentence
    """
    # convert list of sentences to their vectors
    sentences_vec = [get_sentence_vec(sentence, word2vec, idf) for sentence in sentences]
    
    # each row in matrix is 300 dimensions embedding of a sentence
    sentences_matrix = np.vstack(sentences_vec)
    # print(sentences_matrix.shape)
    
    # average of all rows, take mean at y-axis
    sentences_centre = sentences_matrix.mean(axis=0)
    
    # result should be (300,) same as single sentence
    # print(sentences_centre.shape)
    return sentences_centre

def get_cluster_centre(df, intent_list, word2vec, idf=None):
    """ get intent cluster centre based on intent list and word embeddings
    
    Depends on function get_sentences_centre()
    
    Args:
        intent_list (list): List of unique intents(str)
        word2vec (dict): word embeddings dictionary 

    Returns:
        result (dict): intent cluster centres in dictionary format - {intent1:embedding1, intent2:embedding2,...}
    """ 
    result = {intent:get_sentences_centre(df[df.intent == intent]['query'].values, word2vec, idf) for intent in intent_list}
    return result

def get_distance_matrix(df_in, word2vec, leave_one_out=False, idf=False):
    """ Get distance for each query to every intent center
    
    Depends on function get_cluster_centre()
    
    Args:
        df_in (pd.DataFrame): input dataframe with intent and query
        word2vec (dict): word embeddings dictionary 
        leave_one_out (bool): whether leave the input query out of training
        idf (bool): whether use weighted word vectors to get sentence embedding

    Returns:
        result (pd.DataFrame): distance matrix for each query, lowest distance intent idealy should match label
    """
    df = df_in.copy()
    intent_list = df.intent.unique().tolist()
    
    if leave_one_out:
        # print("Leave one out")
        sentence_distance = []
        
        for ind in df.index:
            sentence_distance_tmp = []
            query = df.loc[ind, 'query']
            df_data = df.drop(ind)
            
            sentence_centre_dic = get_cluster_centre(df_data, intent_list, word2vec, idf)
            for intent in intent_list:
                sentence_distance_tmp.append(cosine_distances(get_sentence_vec(query, word2vec, idf).reshape(1,-1), 
                                                              sentence_centre_dic[intent].reshape(1,-1)).item())
            sentence_distance.append(sentence_distance_tmp)

        df_sentence_distance = pd.DataFrame(sentence_distance, columns=intent_list)
        df.reset_index(drop=True, inplace=True)
        result = pd.concat([df, df_sentence_distance], axis=1)
    
    else:

        sentence_centre_dic = get_cluster_centre(df, intent_list, word2vec, idf)
        # build dataframe that contains distance between each query to all intent cluster centre
        for intent in intent_list:
            # distance = cosine_similarity(sentence embedding, intent cluster centre embedding)
            df[intent] = df['query'].apply(lambda x: cosine_distances(get_sentence_vec(x, word2vec, idf).reshape(1,-1), 
                                                                      sentence_centre_dic[intent].reshape(1,-1)).item())
        result = df

    return result

def evaluate_distance_matrix(df_in):
    """ Evaluate distance matrix by compare closest intent center and label """
    df = df_in.copy()
    df.set_index(['intent', 'query'], inplace=True)
    df['cluster'] = df.idxmin(axis=1)
    df.reset_index(inplace=True)
    df['correct'] = (df.cluster == df.intent)
    accuracy = sum(df.correct) / len(df)
    # print("Accuracy for distance-based classification is", '{:.2%}'.format(result))
    return accuracy

def get_clustering_accuracy(df_in, word2vec):
    """ get text clutering result without idf """
    df_result = get_distance_matrix(df_in, word2vec)
    # print(df_result.head())
    accuracy = evaluate_distance_matrix(df_result)
    return df_result, accuracy

# TEST
def get_idf_acc(df_in, word2vec, idf):
    """ get text clutering result using idf """
    df_result = get_distance_matrix(df_in, word2vec, leave_one_out=False, idf=idf)
    # print(df_result.head())
    accuracy = evaluate_distance_matrix(df_result)
    return df_result, accuracy

