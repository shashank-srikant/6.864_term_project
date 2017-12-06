import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.autograd as autograd
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import ConfigParser
from tqdm import tqdm
from time import time
import cPickle as pickle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from ranking_metrics import compute_mrr, precision_at_k, compute_map

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
np.random.seed(0)
from operator import itemgetter, attrgetter
from sklearn.metrics.pairwise import cosine_similarity
from meter import AUCMeter
#torch.manual_seed(0)

def get_similarity(embed1, embed2):
    # embed1, embed2 could be tf-idf vectors, word embeddings, anything.
    return cosine_similarity(embed1, embed2)

def process_file(query_id_path, text_pd, vectorizer, ground_truth):
    similarity_vector = []
    ground_truth_arr = []
    
    data_frame = pd.read_table(query_id_path, sep=' ', header=None)
    data_frame.columns = ['query_id', 'candidate_id']
    
    #num_samples = min(100,data_frame.shape[0])
    num_samples = data_frame.shape[0]
    for idx in tqdm(range(num_samples)):
        #try:
            ind1 = np.where(text_pd['idz'] == data_frame.loc[idx,'query_id'])
            ind2 = np.where(text_pd['idz'] == data_frame.loc[idx,'candidate_id'])
            ind1 = int(ind1[0])
            ind2 = int(ind2[0])
            q1 = text_pd.loc[ind1,'body']
            q2 = text_pd.loc[ind2,'body']
            s = get_similarity(vectorizer.transform([q1]),vectorizer.transform([q2]))
            similarity_vector.append(float(s[0][0]))
            ground_truth_arr.append(ground_truth)
        #except:
         #    print "oopsie1" 
            
        
    return similarity_vector, ground_truth_arr



config = ConfigParser.ConfigParser()
config.readfp(open(r'../src/config.ini'))
SAVE_PATH = config.get('paths', 'save_path')
DATA_FILE_NAME = config.get('paths', 'extracted_data_file_name')
TRAIN_TEST_FILE_NAME = config.get('paths', 'train_test_file_name')
SAVE_NAME = config.get('cnn_params', 'save_name')
NUM_NEGATIVE = int(config.get('data_params', 'NUM_NEGATIVE')) 
DATA_PATH_TARGET = config.get('paths', 'data_path_target')
MAX_TITLE_LEN = int(config.get('data_params', 'MAX_TITLE_LEN'))
MAX_BODY_LEN = int(config.get('data_params', 'MAX_BODY_LEN'))

data_filename = SAVE_PATH + DATA_FILE_NAME
train_test_filename = SAVE_PATH + TRAIN_TEST_FILE_NAME

print "loading pickled data..."
tic = time()
with open(data_filename) as f:  
    train_text_df, train_idx_df, dev_idx_df, test_idx_df, embeddings, word_to_idx = pickle.load(f)
f.close()
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

train_text_df['title_body'] = train_text_df['title'] + " " + train_text_df['body']

vectorizer = TfidfVectorizer(max_df=0.8, max_features=None,
                                 min_df=2, stop_words='english', strip_accents = 'ascii',
                             )
vec_obj = vectorizer.fit(train_text_df['title_body'].tolist())

target_text_file = DATA_PATH_TARGET + 'corpus.txt'
target_text_df = pd.read_table(target_text_file, sep='\t', header=None)
target_text_df.columns = ['id', 'title', 'body']

text_pd = pd.read_table(DATA_PATH_TARGET + 'corpus.txt', sep='\t', header=None)
text_pd.columns = ['idz', 'text','body']
text_pd['body'] = text_pd['text'] + " " + text_pd['body']
text_pd = text_pd.dropna()
text_pd = text_pd.reset_index()


target_test_neg = DATA_PATH_TARGET + 'test.neg.txt'
sim_test_neg, ground_truth_neg = process_file(target_test_neg, text_pd, vectorizer, 0)

target_test_pos = DATA_PATH_TARGET + 'test.pos.txt'
sim_test_pos, ground_truth_pos = process_file(target_test_pos, text_pd, vectorizer, 1)

auc_meter = AUCMeter()
auc_meter.add(np.array(sim_test_pos), np.array(ground_truth_pos))
auc_meter.add(np.array(sim_test_neg), np.array(ground_truth_neg))
print auc_meter.value(0.05)


target_dev_neg = DATA_PATH_TARGET + 'dev.neg.txt'
sim_dev_neg, ground_truth_neg = process_file(target_dev_neg, text_pd, vectorizer, 0)
target_dev_pos = DATA_PATH_TARGET + 'dev.pos.txt'
sim_dev_pos, ground_truth_pos = process_file(target_dev_pos, text_pd, vectorizer, 1)

auc_meter1 = AUCMeter()
auc_meter1.add(np.array(sim_dev_pos), np.array(ground_truth_pos))
auc_meter1.add(np.array(sim_dev_neg), np.array(ground_truth_neg))
print auc_meter1.value(0.05)