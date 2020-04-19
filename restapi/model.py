import pandas as pd
import time

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