"""
Text clustering based on TF-IDF and DBSCAN
"""
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

import utility.string_clean as string_clean
import importlib
importlib.reload(string_clean)


def build_sentence_vec(sentence, model, num_features, index2word_set, idf=None):
    """ Build sentence embedding """
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
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


def get_data(db, item_id):
    """ load data from article server """
    sql = f'SELECT * FROM article WHERE search_id IN ' \
          f'(SELECT search_id FROM item_search WHERE item_id = {item_id}) ' \
          f'AND (relevance_manual is null or relevance_manual = 1)'
    data = pd.read_sql(sql, con=db.sql_engine)

    if len(data) == 0:
        print('No news.')
        return

    # Combine duplicate news
    data = data.sort_values('search_rank').groupby('href').head(1)

    return data


def text_clustering(db, item_id, method='tfidf', word2vec_model=None):
    # 1. load data
    data = get_data(db, item_id)
    news = data.copy()

    # 2. Preprocessing
    # Remove numbers
    # news['model_text'] = news['model_text'].apply(string_clean.remove_numbers)

    # Remove punctuations
    news['model_text'] = news['model_text'].apply(string_clean.remvoe_punctuation)

    # Remove single character
    # news['model_text'] = news['model_text'].apply(string_clean.remove_single_character)

    # Stemming
    # ps = PorterStemmer()
    # news['model_text'] = news['model_text'].apply(ps.stem)

    def tokenize(s): 
        return " ".join(word_tokenize(s))
    news['model_text'] = news['model_text'].apply(tokenize)

    
    # 3. Get similarity matrix and distnace for DBSCAN
    if method == 'tfidf':
        # Use tfidf alone to calculate similarity
        tfidf = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 3)).fit_transform(news['model_text'].values.astype('U'))

        # cosine similarity
        similarity = tfidf * tfidf.T

        distance = 1-similarity.todense()

        eps = 0.85

    else:
        # Use word2vec
        index2word_set = set(word2vec_model.index2word)
        if method == 'word2vec':
            emb = [build_sentence_vec(news.iloc[i, news.columns.get_loc('model_text')], model=word2vec_model, num_features=300,
                                  index2word_set=index2word_set) for i in range(len(news))]
            eps = 0.125

        # Use word2vec + idf
        elif method == 'idf-word2vec':
            idf = compute_idf_weights(news['model_text'])
            emb = [build_sentence_vec(news.iloc[i, news.columns.get_loc('model_text')], model=word2vec_model, num_features=300,
                               index2word_set=index2word_set, idf=idf) for i in range(len(news))]
            eps = 0.16
        emb = np.array(emb)
        similarity = cosine_similarity(emb)
        distance = 1 - similarity


    # 4. Clustering using DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=2, metric='precomputed').fit(distance)

    #get labels
    labels = dbscan.labels_

    #get number of clusters
    no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print('No of clusters:', no_clusters)

    #%% Rank topics
    # assign cluster labels
    news = data.copy()
    news['topic_id'] = labels

    # for standalone articles, assign negative cluster labels
    outlier = news['topic_id'] == -1
    labels_outlier = news.loc[outlier, 'article_id']\
        .reset_index()\
        .drop('index', axis=1)\
        .reset_index()\
        .rename(columns={'index': 'topic_id_outlier'})
    labels_outlier['topic_id_outlier'] = -1*(labels_outlier['topic_id_outlier'] + 1)

    news = news.merge(labels_outlier, how='left', on='article_id')
    outlier = news['topic_id'] == -1
    news.loc[outlier, 'topic_id'] = news.loc[outlier, 'topic_id_outlier']
    news.drop(['topic_id_outlier'], axis=1, inplace=True)

    # compute article score
    news['search_rank_score'] = 10/(9+news['search_rank'])
    news['related_article_count'].fillna(0, inplace=True)
    news['related_article_count_score'] = 0
    if max(news['related_article_count']) > 0:
        news['related_article_count_score'] = news['related_article_count'] / max(news['related_article_count'])
        news['related_article_count_score'].fillna(0, inplace=True)
    news['article_score'] = news['search_rank_score'] + 0.5*news['related_article_count_score']

    # save article score
    try:
        for i in range(len(news)):
            row = news.iloc[i]
            sql = "UPDATE article SET article_score=%s WHERE article_id=%s"
            db.cursor.execute(sql, (row['article_score'], row['article_id']))
        db.conn.commit()
    except:
        db.conn.rollback()
        raise

    # compute topic score
    # topic_score = sum of(top three article scores)
    news['article_score'] = pd.to_numeric(news['article_score'])
    topic_score = news.groupby('topic_id')['article_score']\
        .apply(lambda grp: grp.nlargest(3).sum())\
        .reset_index()\
        .rename(columns={'article_score': 'topic_score'})

    news = news.merge(topic_score, on='topic_id')

    # rank topics
    news = news.sort_values(['topic_score', 'article_score'], ascending=[False, False]) \
               .loc[:, ['topic_id',
                        'article_id',
                        'date',
                        'title',
                        'full_title',
                        'media',
                        'language',
                        'topic_score',
                        'article_score',
                        'href',
                        'thumbnail',
                        'abstract']]

    news['topic_id'] = news['topic_id'].astype(int)

    # pick top 30 topics
    # top_topic_id = news['topic_id'].unique()[:30]
    # news = news[news['topic_id'].isin(top_topic_id)]

    #%% For each cluster, find title of cluster center
    def get_cluster_center(news, topic_id, distance):
        indexes = news[news.topic_id == topic_id].index
        within_cluster_distance = distance[np.ix_(indexes, indexes)]
        return news.loc[indexes[np.argmin(np.sum(within_cluster_distance, axis=0))], 'title']

    topic_ids = news.topic_id.drop_duplicates()
    topic_title_cluster_center = {topic_id: get_cluster_center(news, topic_id, distance) for topic_id in topic_ids}

    #%% For each cluster, find title of cluster center
    # scale scores
    news['topic_score'] = (news['topic_score'] - min(news['topic_score'])) / (max(news['topic_score']) - min(news['topic_score']))
    news['article_score'] = (news['article_score'] - min(news['article_score'])) / (max(news['article_score']) - min(news['article_score']))

    topic_ids = news['topic_id'].unique()
    topic_rank = 1

    # remove existing topics (if any) for this item_id
    try:
        sql = 'SET SQL_SAFE_UPDATES = 0'
        db.cursor.execute(sql)

        sql = f'DELETE t, ita FROM topic t JOIN item_topic_article ita ON t.topic_id = ita.topic_id WHERE ita.item_id = {str(item_id)}'
        db.cursor.execute(sql)
        db.conn.commit()
    except:
        db.conn.rollback()
        raise

    # add topic to database
    try:
        for topic_id in topic_ids:
            news_idx = news.topic_id == topic_id
            topic_score = news.loc[news_idx, 'topic_score'].values[0]

            sql = f"INSERT INTO topic (topic_importance, topic_title_cluster_center) VALUES (%s, %s)"
            db.cursor.execute(sql, (str(topic_score), topic_title_cluster_center[topic_id]))
            topic_id = db.cursor.lastrowid

            for article_id in news.loc[news_idx, 'article_id'].values:
                sql = f"INSERT INTO item_topic_article (item_id, topic_id, article_id, topic_rank) VALUES (%s, %s, %s, %s)"
                db.cursor.execute(sql, (item_id, topic_id, article_id, topic_rank))

            topic_rank += 1

        db.conn.commit()
    except:
        db.conn.rollback()
        raise()
