import pandas as pd
import csv
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import fileinput
import scipy.special
import numpy as np
import scipy.stats
import pickle
import numpy as np
import pickle
import io 
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

from keras.preprocessing import text ##needed
# install Graphviz after download installer (https://www.graphviz.org/)
# insert in code this two lines:
import os
from pickle import dump




def build_corpus(phrase):
    corpus_all = [word.lower() for word in phrase]
    tokenizer_all = text.Tokenizer()
    tokenizer_all.fit_on_texts(corpus_all)
    word2id_all = tokenizer_all.word_index
    id2word_all = {v:k for k, v in word2id_all.items()}
    vocab_size_all = len(word2id_all) + 1 
    wids_all = [word2id_all[word] for word in corpus_all]
    dump(tokenizer_all, open('/various/mtzortzi/from_nikela/tokenizer_all.pkl', 'wb'))   
    return corpus_all, vocab_size_all, wids_all, id2word_all, word2id_all, tokenizer_all


def create_merged(filename):
	merged_static = pd.read_csv(filename) ##pd.read_csv('/various/mtzortzi/LogAider-master/RAS-JOB/merged_static.csv')
	un = merged_static['STATIC'].unique()
# I remove all space characters from MESSAGE, so as to use each message as a single phrase in DESH. What's more I remove all the parentheses so as not to be a problem in skip-gram.
# The new hyphenated multi-word entities are saved in a new column named MULTIWORDS.
	msg_list = []
	for msg in merged_static['STATIC']:
		msg = ((msg.strip('()')).replace(')', '')).replace('(', '')
		msg_list.append(msg)
	# print(msg_list[:10])
	merged_static['MULTIWORDS'] = msg_list

	merged_static['MULTIWORDS'] = merged_static['MULTIWORDS'].str.replace(' ', '')
	merged_static['MULTIWORDS'] = merged_static['MULTIWORDS'].str.replace('.', '')
	merged_static['MULTIWORDS'] = merged_static['MULTIWORDS'].str.replace(':', '')
	merged_static['STATIC'] = merged_static['STATIC'].str.lower()
	merged_static['MULTIWORDS'] = merged_static['MULTIWORDS'].str.lower()
	return merged_static


def dataset_split_store(dataset, path):
	dataset_shuffled = dataset.reindex(np.random.permutation(dataset.index))
	train30, test70 = np.split(dataset_shuffled,  [int(.3*len(dataset_shuffled))])
	train30.to_csv(path+'train30.csv', index = False, sep=',', quoting=csv.QUOTE_ALL)
	test70.to_csv(path+'test70.csv', index = False, sep=',', quoting=csv.QUOTE_ALL)
	return


