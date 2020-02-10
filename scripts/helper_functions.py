import pandas as pd
from pandas import DataFrame
import nltk
from nltk import FreqDist
import numpy as np
import re
import spacy
from nltk.corpus import stopwords
import pickle
# Gensim
import gensim
from gensim import corpora
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns



def lemmatization(texts, tags=['NOUN', 'ADJ']):
    """
        Word lemmatization function.
        Args:
            texts: entire source lists of tokenized comments
            tags: list of tokenized part of speech
        Returns:
            output: list of lemmatized word
    """
    output = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent))
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output

def remove_stopwords(rev):
    """
        Remove stopwords.
        Args:
            rev: lists of entire source tokenized comments
        Returns:
            rev_new: List of tokenized comments without stopwords
    """
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

def add_bigram(token_list):
    """add bigrams in the data"""
    bigram = gensim.models.Phrases(token_list)
    bigram = [bigram[line] for line in token_list]
    return bigram


def add_trigram(token_list):
    """add trigrams in the data"""
    bigram = add_bigram(token_list)
    trigram = gensim.models.Phrases(bigram)
    trigram = [trigram[line] for line in bigram]
    return trigram

def compute_coherence_lda(corpus, dictionary, start=None, limit=None, step=None):
    """Compute c_v coherence for various number of topics """
    topic_coherence = []
    model_list = []
    tokens_list = df.trigram_tokens.values.tolist()
    texts = [[token for sub_token in tokens_list for token in sub_token]]
    for num_topics in range(start, limit, step):
        model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                             eta='auto', workers=4, passes=20, iterations=100,
                             random_state=42, eval_every=None,
                             alpha='asymmetric',  # shown to be better than symmetric in most cases
                             decay=0.5, offset=64  # best params from Hoffman paper
                             )
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        topic_coherence.append(coherencemodel.get_coherence())

    return model_list, topic_coherence

