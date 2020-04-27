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

# from model import IntentClassifier

PATH_PROJ = Path(__file__).parent

PATH_DATA = PATH_PROJ / 'lib' / 'data'
PATH_DATA.mkdir(parents=True, exist_ok=True)

PATH_MODELS = PATH_PROJ / 'lib' / 'models'
PATH_MODELS.mkdir(parents=True, exist_ok=True)

sys.path.append(str(PATH_PROJ))


# TRAIN
df_train = pd.read_csv(str(PATH_DATA / 'train.csv'))
df_train.dropna(inplace=True)

# TEST
df_test = pd.read_csv('uat_data_intent.csv')
df_test.dropna(inplace=True)

# rename dataframe
df_train = df_train.rename(columns={'Intent': 'intent', 'Questions': 'query'})
df_train = df_train[['intent', 'query']]
# rename df_test
df_test = df_test.rename(columns={'User Clicked intent': 'intent', 'Question': 'query'})
df_test = df_test[['intent', 'query']]

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

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = nlp(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

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
    vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer)
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

def test_clustering_accuracy(df_in, word2vec):
    """ test accuracy based on distance of sentence to each cluster center"""
    df_result = get_distance_matrix(df_in, word2vec)
    # print(df_result.head())
    accuracy = evaluate_distance_matrix(df_result)
    return df_result, accuracy

# TEST
def test_idf_acc(df_in, word2vec, idf):
    df_result = get_distance_matrix(df_in, word2vec, leave_one_out=False, idf=idf)
    # print(df_result.head())
    accuracy = evaluate_distance_matrix(df_result)
    return df_result, accuracy

# preprocessing questions
df_train['query'] = df_train['query'].apply(clean_text)
df_train['query'] = df_train['query'].apply(nltk_tokenize)
df_train['query'] = df_train['query'].apply(lambda x:' '.join([token.lemma_ for token in nlp(x) if token.lemma_ not in stop_words]))
df_train['query'] = df_train['query'].str.lower()

# preprocessing test as well
df_test['query'] = df_test['query'].apply(clean_text)
df_test['query'] = df_test['query'].apply(nltk_tokenize)
df_test['query'] = df_test['query'].apply(lambda x:' '.join([token.lemma_ for token in nlp(x) if token.lemma_ not in stop_words]))
df_test['query'] = df_test['query'].str.lower()

# all unique intents
intent_list = df_train.intent.unique().tolist()

# get idf
idf = get_idf_TfidfVectorizer(df_train['query'].tolist())

try:
    word2vec
except NameError:
    word2vec = api.load("word2vec-google-news-300")  

df_result, accuracy = test_idf_acc(df_train, word2vec, idf)
print("Traing accuracy for word2vec + IDF is", '{:.2%}'.format(accuracy)) 

# get cluster centers from training set
idf = get_idf_TfidfVectorizer(df_train['query'].tolist())
dict_cluster = get_cluster_centre(df_train, intent_list, word2vec, idf)

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

df_test_cluster = get_distance_matrix_idf(df_test, intent_list, dict_cluster, word2vec, idf)

def get_top_3_clusters(data, intent_list):
    data = data.copy()
    cluster_cols = intent_list.copy()

    data['clusters_top3'] = data.apply(lambda x: np.argsort(x[cluster_cols].values)[:3].tolist(), axis=1)

    intents = cluster_cols # get all tickers
    intent2index = {v: i for (i, v) in enumerate(intents)}

    data['target'] = data['intent'].apply(lambda x: intent2index[x])

    top_clusters_cols = pd.DataFrame(data['clusters_top3'].values.tolist(),columns = ['clusters_1','clusters_2','clusters_3']).reset_index(drop=True)
    data = data.reset_index(drop=True)
    data = pd.concat([data,top_clusters_cols], axis=1)

    data.drop(columns = 'clusters_top3', inplace=True)
    data.drop(columns = cluster_cols, inplace=True)
    
    # print(data.head())
    return data, intent2index

df_train, intent2index = get_top_3_clusters(df_result, cluster_cols)
df_test_cluster_top_n, _ = get_top_3_clusters(df_test_cluster, intent_list)

def get_accuracy(data, top=1):
    data = data.copy()
    
    assert top in (1,2,3), "top must be in (0, 1, 2)"
    
    if top == 1:
        # top 1 accuracy
        accuracy = (data[(data['clusters_1'] == data['target'])].shape[0] / data.shape[0])
    elif top == 2:
        # top 2 accuracy
        data["exists"] = data.drop(data.columns[[0,1,2,5]], 1).isin(data["target"]).any(1)
        accuracy = sum(data['exists'])/ data.shape[0]
    elif top == 3:
        # top 3 accuracy
        data["exists"] = data.drop(data.columns[[0,1,2]], 1).isin(data["target"]).any(1)
        accuracy = sum(data['exists'])/ data.shape[0]
    else:
        raise ValueError("top must be in (0, 1, 2)") 
    
    print('Accuracy for top {} clustering result is {:.1%}'.format(top, accuracy))
    return accuracy

get_accuracy(df_test_cluster_top_n, 1)
get_accuracy(df_test_cluster_top_n, 2)
get_accuracy(df_test_cluster_top_n, 3)

def get_keywords(intent_list):
    """ Get list of keywords from intent """
    keywords = []
    for intent in list(set(intent_list)):
        keywords.extend(intent.strip().split(' '))
    keyword_list = list(set(keywords))
    keyword_list = [i.lower() for i in keyword_list if i.lower() not in stop_words]
    keyword_list.append('nsip')

    keyword_list_lemma = []
    text = nlp(' '.join([w for w in keyword_list]))
    for token in text:
        keyword_list_lemma.append(token.lemma_)
    return keyword_list_lemma

keyword_list_lemma = get_keywords(intent_list)

def get_nlp_features(df, keyword_list_lemma):
    """ Get keyword features from dataframe """
    data = df.copy()
    data['lemma'] = data['query'].apply(lambda x:' '.join([token.lemma_ for token in nlp(x) if token.lemma_ not in stop_words]))
    data['keyword'] = data['lemma'].apply(lambda x: list(set([token.lemma_ for token in nlp(x) if token.lemma_ in keyword_list_lemma])))

    data['noun'] = data['query'].apply(lambda x: list(set([token.lemma_ for token in nlp(x) if token.pos_ in ['NOUN','PROPN'] and token.lemma_ not in stop_words])))
    data['verb'] = data['query'].apply(lambda x: list(set([token.lemma_ for token in nlp(x) if token.pos_ in ['VERB'] and token.lemma_ not in stop_words])))
    data['adj'] = data['query'].apply(lambda x: list(set([token.lemma_ for token in nlp(x) if token.pos_ in ['ADJ'] and token.lemma_ not in stop_words])))

    data['noun'] = data['noun'].apply(lambda x: ' '.join([w for w in x]))
    data['verb'] = data['verb'].apply(lambda x: ' '.join([w for w in x]))
    data['adj'] = data['adj'].apply(lambda x: ' '.join([w for w in x]))
    data['keyword'] = data['keyword'].apply(lambda x: ' '.join([w for w in x]))
    return data

df_train = get_nlp_features(df_train, keyword_list_lemma)
df_test = get_nlp_features(df_test_cluster_top_n)


# combine model score
countvector_cols = ['lemma', 'keyword', 'noun', 'verb']
top_clusters_cols = ['clusters_1', 'clusters_2', 'clusters_3']

feature_cols = countvector_cols + top_clusters_cols

def get_train_test(df_train, df_test, feature_cols):
    """ split dataset, get X_train, X_test, y_train, y_test """
    X_train = df_train[feature_cols]
    # print(X_train.head(1))
    y_train = df_train['target']
    # print(y_train.head(1))
    X_test = df_test[feature_cols]
    y_test = df_test['target']
    # print(X_test.head(1))
    # print(y_test.head(1))
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = get_train_test(df_train, df_test, feature_cols)

def add_nlp_to_x(X_train, X_test):
    """ Add NLP features to input X """
    v_lemma = TfidfVectorizer()
    x_train_lemma = v_lemma.fit_transform(X_train['lemma'])
    x_test_lemma = v_lemma.transform(X_test['lemma'])
    vocab_lemma = dict(v_lemma.vocabulary_)

    v_keyword = TfidfVectorizer()
    x_train_keyword = v_keyword.fit_transform(X_train['keyword'])
    x_test_keyword = v_keyword.transform(X_test['keyword'])
    vocab_keyword = dict(v_keyword.vocabulary_)

    v_noun = TfidfVectorizer()
    x_train_noun = v_noun.fit_transform(X_train['noun'])
    x_test_noun = v_noun.transform(X_test['noun'])
    vocab_noun = dict(v_noun.vocabulary_)

    v_verb = TfidfVectorizer()
    x_train_verb = v_verb.fit_transform(X_train['verb'])
    x_test_verb = v_verb.transform(X_test['verb'])
    vocab_verb = dict(v_verb.vocabulary_)
    
    # combine all features 
    x_train_combined = hstack((x_train_lemma,x_train_keyword,x_train_noun,x_train_verb,X_train[top_clusters_cols].values),format='csr')
    x_train_combined_columns= v_lemma.get_feature_names()+v_keyword.get_feature_names()+v_noun.get_feature_names()+v_verb.get_feature_names()+top_clusters_cols

    x_test_combined = hstack((x_test_lemma,x_test_keyword,x_test_noun,x_test_verb,X_test[top_clusters_cols].values),format='csr')
    x_test_combined_columns= v_lemma.get_feature_names()+v_keyword.get_feature_names()+v_noun.get_feature_names()+v_verb.get_feature_names()+top_clusters_cols

    x_train_combined = pd.DataFrame(x_train_combined.toarray())
    x_train_combined.columns = x_train_combined_columns

    x_test_combined = pd.DataFrame(x_test_combined.toarray())
    x_test_combined.columns = x_test_combined_columns
    
    return x_train_combined, x_test_combined

x_train_combined, x_test_combined = add_nlp_to_x(X_train, X_test)

# build classifier
clf = RandomForestClassifier(max_depth=50, n_estimators=1000)
clf.fit(x_train_combined, y_train)

probs = clf.predict_proba(x_test_combined)
best_3 = pd.DataFrame(np.argsort(probs, axis=1)[:,-3:],columns=['top3','top2','top1'])
best_3['top1'] = clf.classes_[best_3['top1']]
best_3['top2'] = clf.classes_[best_3['top2']]
best_3['top3'] = clf.classes_[best_3['top3']]

result = pd.concat([best_3.reset_index(drop=True),pd.DataFrame(y_test).reset_index(drop=True), X_test[feature_cols].reset_index(drop=True)], axis=1)
score_1 = result[result['top1'] == result['target']].shape[0] / result.shape[0]
score_2 = result[(result['top1'] == result['target']) | (result['top2'] == result['target'])].shape[0] / result.shape[0]
score_3 = result[(result['top1'] == result['target']) | (result['top2'] == result['target'])| (result['top3'] == result['target'])].shape[0] / result.shape[0]

print('Accuracy for top 1 clustering + classifier result is {:.1%}'.format(score_1))
print('Accuracy for top 2 clustering + classifier result is {:.1%}'.format(score_2))
print('Accuracy for top 3 clustering + classifier result is {:.1%}'.format(score_3))
