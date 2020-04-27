#%% Import
import sys
import re
import math
import string
import time
from pathlib import Path

import numpy as np
import pandas as pd

import string
import pickle

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_distances

import spacy

from gensim.models import KeyedVectors

from nltk.tokenize import word_tokenize

#%% Constants
PATH_PROJ = Path(__file__).parent
PATH_DATA = PATH_PROJ / 'lib' / 'data'
PATH_MODELS = PATH_PROJ / 'lib' / 'models'
sys.path.append(str(PATH_PROJ))

# timer
start_time = time.time()
print("Start loading...")
#%% Load
# list of punctuation marks
punctuations = string.punctuation

# Create spacy word2vec and list of stopwords
nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# load classifier
with open(str(PATH_MODELS/"RFClassifier.pkl"), 'rb') as f:
    clf = pickle.load(f)

# load vectorizer
with open(str(PATH_MODELS/'TFIDFVectorizer_lemma.pkl'), 'rb') as f:
    v_lemma = pickle.load(f)
with open(str(PATH_MODELS/'TFIDFVectorizer_keyword.pkl'), 'rb') as f:
    v_keyword = pickle.load(f)
with open(str(PATH_MODELS/'TFIDFVectorizer_noun.pkl'), 'rb') as f:
    v_noun = pickle.load(f)
with open(str(PATH_MODELS/'TFIDFVectorizer_verb.pkl'), 'rb') as f:
    v_verb = pickle.load(f)

# load intent list
with open(str(PATH_MODELS/'intent_list.pkl'), 'rb') as f:
    intent_list = pickle.load(f)

# load clustering centres
with open(str(PATH_MODELS/'dict_cluster.pkl'), 'rb') as f:
    dict_cluster = pickle.load(f)

# load idf
with open(str(PATH_MODELS/'idf.pkl'), 'rb') as f:
    idf = pickle.load(f)

# load intent2index
with open(str(PATH_MODELS/'intent2index.pkl'), 'rb') as f:
    intent2index = pickle.load(f)

# load keyword_list_lemma
with open(str(PATH_MODELS/'keyword_list_lemma.pkl'), 'rb') as f:
    keyword_list_lemma = pickle.load(f)
    
EMBEDDING_FILE = str(PATH_MODELS / 'GoogleNews-vectors-negative300.bin.gz')
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

duration= time.time() - start_time
print(f"all pickles loaded, using {duration} sec")

#%% Pipeline
def get_intent_nlp_clustering(query):
    """
        return a dataframe df
        columns: pred_seq, intent_class, intent_string, pred_prob
        rows: top 3 prediciton, example for first row: 1, 0, Promotions, 0.66
    """
    # setup timer
    start = time.time()

    #%% pipeline
    # convert question to dataframe
    df = pd.DataFrame()
    df = pd.DataFrame(columns=['query'])
    df.loc[0] = [query]

    # preprocessing test query
    df['query'] = df['query'].apply(clean_text)
    df['query'] = df['query'].apply(nltk_tokenize)
    df['query'] = df['query'].apply(lambda x:' '.join([token.lemma_ for token in nlp(x) if token.lemma_ not in stop_words]))
    df['query'] = df['query'].str.lower()
    
    # get nlp features
    df = get_nlp_features(df, keyword_list_lemma)

    # get clustering matrix
    df_cluster = get_distance_matrix_idf(df, intent_list, dict_cluster, word2vec, idf)

    # get top 3 clusters
    top_3 = get_top_3(df_cluster, intent_list)
    # print(top_3)

    # get inputs for RF classifier
    top_clusters_cols = ['clusters_1', 'clusters_2', 'clusters_3']
    # get input vector for RFClassifier
    X_in = add_nlp_vec(top_3, v_lemma, v_keyword, v_noun, v_verb, top_clusters_cols)

    # get prediction proba
    probs = clf.predict_proba(X_in)

    # get index for top 3 prediction by proba
    ind = np.argsort(probs, axis=1)[:,-3:]

    # save probability
    proba = probs[0][ind[0]]

    # save predicitons as dataframe
    best_3 = pd.DataFrame(ind,columns=['top3','top2','top1'])
    best_3['top1'] = clf.classes_[best_3['top1']]
    best_3['top2'] = clf.classes_[best_3['top2']]
    best_3['top3'] = clf.classes_[best_3['top3']]
    best_3['top3_prob'] = proba[0]
    best_3['top2_prob'] = proba[1]
    best_3['top1_prob'] = proba[2]

    # get index to intent dictionary from intent2index
    index2intent = {y:x for x,y in intent2index.items()}

    # get class name of top predictions
    best_3['top1_name'] = best_3['top1'].apply(get_target_name, index2intent=index2intent)
    best_3['top2_name'] = best_3['top2'].apply(get_target_name, index2intent=index2intent)
    best_3['top3_name'] = best_3['top3'].apply(get_target_name, index2intent=index2intent)

    # output prediction
    top1 = best_3.at[0,'top1_name']
    top2 = best_3.at[0,'top2_name']
    top3 = best_3.at[0,'top3_name']
    top1_prob = best_3.at[0,'top1_prob']
    top2_prob = best_3.at[0,'top2_prob']
    top3_prob = best_3.at[0,'top3_prob']

    # print(f'For sentence:\n{query}\n')
    # print(f'Top 1 prediction intent is {top1} with probability {100*top1_prob:.2f}%')
    # print(f'Top 2 prediction intent is {top2} with probability {100*top2_prob:.2f}%')
    # print(f'Top 3 prediction intent is {top3} with probability {100*top3_prob:.2f}%')

    top1_class = best_3.at[0,'top1']
    top2_class = best_3.at[0,'top2']
    top3_class = best_3.at[0,'top3']

    # convert to output
    df = pd.DataFrame([
            [1, top1_class, top1, top1_prob],
            [2, top2_class, top2, top2_prob],
            [3, top3_class, top3, top3_prob]
        ], columns=['pred_seq', 'intent_class', 'intent', 'pred_prob'])

    inference_time = time.time() - start
    print("inference_time", inference_time)

    output = process_output(df)
    return output

def process_output(df):
    """ Process DataFrame to output format            
    
    Return:
        internal_model_results = {'intent': [1,2,3]}
    """
    result_list = []

    # get mapping
    df_map = pd.read_csv(str(PATH_DATA/'intent_index.csv'))
    dict_map = df_map.set_index('Intent')['Label'].to_dict()

    
    # get list of intents
    for i in range(1,4):
        intent = df.loc[df['pred_seq'] == i]['intent'].values[0]
        intent_label = dict_map[intent]
        result_list.append(intent_label)

    return result_list


#%% utilities
def get_nlp_features(df, keyword_list_lemma):
    """ Get keyword features from dataframe """
    data = df.copy()
    data['lemma'] = data['query'].apply(lambda x:' '.join([token.lemma_ for token in nlp(x) if token.lemma_ not in stop_words]))
    data['keyword'] = data['lemma'].apply(lambda x: list(set([token.lemma_ for token in nlp(x) if token.lemma_ in keyword_list_lemma])))

    data['noun'] = data['query'].apply(lambda x: list(set([token.lemma_ for token in nlp(x) if token.pos_ in ['NOUN','PROPN'] and token.lemma_ not in stop_words])))
    data['verb'] = data['query'].apply(lambda x: list(set([token.lemma_ for token in nlp(x) if token.pos_ in ['VERB'] and token.lemma_ not in stop_words])))

    data['noun'] = data['noun'].apply(lambda x: ' '.join([w for w in x]))
    data['verb'] = data['verb'].apply(lambda x: ' '.join([w for w in x]))
    data['keyword'] = data['keyword'].apply(lambda x: ' '.join([w for w in x]))
    return data

def get_distance_matrix_idf(df_test, intent_list, dict_cluster, word2vec, idf):
    """ Get distance for each query to every intent center
        
    Args:
        df_test (pd.DataFrame): input test dataframe with intent and query
        intent_list (list): list of intents to loop through
        dict_cluster (dict): dictionary of cluster centres
        word2vec (dict): word embeddings dictionary
        idf (dict): idf of each words

    Returns:
        result (pd.DataFrame): distance matrix for each query, lowest distance intent idealy should match label
    """
    df = df_test.copy()
    for intent in intent_list:
        # distance = cosine_similarity(sentence embedding, intent cluster centre embedding)
        df[intent] = df['query'].apply(lambda x: cosine_distances(get_sentence_vec(x, word2vec, idf).reshape(1,-1), 
                                                                  dict_cluster[intent].reshape(1,-1)).item())
    return df

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

def get_top_3(data, intent_list):
    data = data.copy()
    cluster_cols = intent_list.copy()

    data['clusters_top3'] = data.apply(lambda x: np.argsort(x[cluster_cols].values)[:3].tolist(), axis=1)

    top_clusters_cols = pd.DataFrame(data['clusters_top3'].values.tolist(),columns = ['clusters_1','clusters_2','clusters_3']).reset_index(drop=True)
    data = data.reset_index(drop=True)
    data = pd.concat([data,top_clusters_cols], axis=1)

    data.drop(columns = 'clusters_top3', inplace=True)
    data.drop(columns = cluster_cols, inplace=True)
    
    # print(data.head())
    return data

def add_nlp_vec(df, v_lemma, v_keyword, v_noun, v_verb, top_clusters_cols):
    """ Transform NLP features to vector for input X using TFIDF """
    x_test_lemma = v_lemma.transform(df['lemma'])
    x_test_keyword = v_keyword.transform(df['keyword'])
    x_test_noun = v_noun.transform(df['noun'])
    x_test_verb = v_verb.transform(df['verb'])
    
    # combine all features 
    x_test_combined = hstack((x_test_lemma,
                              x_test_keyword,
                              x_test_noun,
                              x_test_verb,
                              df[top_clusters_cols].values),format='csr')

    x_test_combined_columns = v_lemma.get_feature_names()+\
                              v_keyword.get_feature_names()+\
                              v_noun.get_feature_names()+\
                              v_verb.get_feature_names()+\
                              top_clusters_cols
    
    x_test_combined = pd.DataFrame(x_test_combined.toarray())
    x_test_combined.columns = x_test_combined_columns
    
    return x_test_combined

def get_target_name(index, index2intent):
    return index2intent[index]
