import time
import pickle

import numpy as np
import pandas as pd

from utils import clean_text, nltk_tokenize, get_nlp_features, get_distance_matrix_idf

# load models


test_query = "Please show me the current promotions"

df = pd.DataFrame()
df = pd.DataFrame(columns=['query'])
df.loc[0] = [test_query]

# preprocessing test as well
df['query'] = df['query'].apply(clean_text)
df['query'] = df['query'].apply(nltk_tokenize)
df['query'] = df['query'].apply(lambda x:' '.join([token.lemma_ for token in nlp(x) if token.lemma_ not in stop_words]))
df['query'] = df['query'].str.lower()

df = get_nlp_features(df)

df_cluster = get_distance_matrix_idf(df, intent_list, dict_cluster, word2vec, idf)

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

top_3 = get_top_3(df_cluster, intent_list)

def add_nlp(df, v_lemma, v_keyword, v_noun, v_verb, top_clusters_cols):
    """ Add NLP features to input X """
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

X_in = add_nlp(top_3, v_lemma, v_keyword, v_noun, v_verb, top_clusters_cols)

probs = clf.predict_proba(X_in)

best_3 = pd.DataFrame(np.argsort(probs, axis=1)[:,-3:],columns=['top3','top2','top1'])
best_3['top1'] = clf.classes_[best_3['top1']]
best_3['top2'] = clf.classes_[best_3['top2']]
best_3['top3'] = clf.classes_[best_3['top3']]

print(best_3)
index2intent = {y:x for x,y in intent2index.items()}

def get_target_name(index, index2intent=index2intent):
    return index2intent[index]

best_3['top1_name'] = best_3['top1'].apply(get_target_name)
best_3['top2_name'] = best_3['top2'].apply(get_target_name)
best_3['top3_name'] = best_3['top3'].apply(get_target_name)

top1 = best_3.at[0,'top1_name']
top2 = best_3.at[0,'top2_name']
top3 = best_3.at[0,'top3_name']

print(f'For sentence:\n{test_query}\n')
print(f'Top 1 prediction intent is {top1}')
print(f'Top 2 prediction intent is {top2}')
print(f'Top 3 prediction intent is {top3}')

# load classification model outside the function
def get_intent_nlp(question, classifier_intent_nlp):
    start = time.time()

	# implementation here
	
	# return a dataframe df
	# columns: pred_seq, intent_class, intent, pred_prob
	# rows: top 3 prediciton, example for first row: 1, 0, Promotions, 0.66
    df = pd.DataFrame([
        [1, 0, 'Promotions', 0.52],
        [2, 2, 'Card Promotions', 0.21],
        [3, 8, 'Card Cancellation', 0.12]
    ], columns=['pred_seq', 'intent_class', 'intent', 'pred_prob'])

    inference_time = time.time() - start
    return df, inference_time
	
# load word2vec dict outside the function
# load classification model outside the function 
def get_intent_nlp_clustering(question, classifier_intent_nlp_clustering, word2vec):
	# implementation here
	
	# return a dataframe df
	# columns: pred_seq, intent_class, intent_string, pred_prob
	# rows: top 3 prediciton, example for first row: 1, 0, Promotions, 0.66
    df = pd.DataFrame([
        [1, 0, 'Promotions', 0.52],
        [2, 2, 'Card Promotions', 0.21],
        [3, 8, 'Card Cancellation', 0.12]
    ], columns=['pred_seq', 'intent_class', 'intent', 'pred_prob'])
    return df