"""
Text clustering pipeline
"""

#%% Import
# Default
import os
import sys
import json
import re
import math
import pickle
from pathlib import Path
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

#%% add current directory to sys.path
sys.path.append(str(PATH_PROJ))
from utils import get_data, clean_text, nltk_tokenize
from utils import SpacyPipeline

#%% Pipeline
def main():
    # 1. load data
    df = get_data()
    # 2. preprocessing
    df['query'] = df['query'].apply(clean_text)
    # 3. tokenize
    sp = SpacyPipeline()

if __name__ == "__main__":
    start = timer()
    main()
    end = timer()
    print(f"Used {end - start} seconds")
