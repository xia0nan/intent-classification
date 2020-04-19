import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Create our list of stopwords
    nlp = spacy.load('en')

    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = nlp(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in STOP_WORDS and word not in string.punctuation ]

    # return preprocessed list of tokens
    return mytokens
