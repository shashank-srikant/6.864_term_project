import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import gensim
import multiprocessing
from random import shuffle

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(0)

DATA_PATH_SOURCE = '/data/vision/fisher/data1/vsmolyakov/nlp_project/data/askubuntu/'
DATA_PATH_TARGET = '/data/vision/fisher/data1/vsmolyakov/nlp_project/data/android/'

tokenizer = RegexpTokenizer(r'\w+')
stop = set(stopwords.words('english'))

def step_decay(epoch):
    lr_init = 0.025
    drop = 0.5
    epochs_drop = 4.0
    lr_new = lr_init * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr_new

def read_corpus(data_frame, tokens_only=False):
    for row_idx in tqdm(range(1000)):
    #for row_idx in tqdm(range(data_frame.shape[0])):
        title = data_frame.loc[row_idx, 'title']
        body = data_frame.loc[row_idx, 'body']
        line = title + " " + body

        if tokens_only:
            yield gensim.utils.simple_preprocess(line)
        else:
            yield gensim.models.doc2vec.TaggedDocument(
                  gensim.utils.simple_preprocess(line),[row_idx])
     #end for

#load data
print "loading data..."
tic = time()
source_text_file = DATA_PATH_SOURCE + '/text_tokenized.txt'
source_text_df = pd.read_table(source_text_file, sep='\t', header=None)
source_text_df.columns = ['id', 'title', 'body']
source_text_df = source_text_df.dropna()
source_text_df['title'] = source_text_df['title'].apply(lambda words: ' '.join(filter(lambda x: x not in stop, words.split())))
source_text_df['body'] = source_text_df['body'].apply(lambda words: ' '.join(filter(lambda x: x not in stop, words.split())))
source_text_df['title_len'] = source_text_df['title'].apply(lambda words: len(tokenizer.tokenize(str(words))))
source_text_df['body_len'] = source_text_df['body'].apply(lambda words: len(tokenizer.tokenize(str(words))))
source_text_df = source_text_df.dropna()
source_text_df = source_text_df.reset_index()

target_text_file = DATA_PATH_TARGET + '/corpus.tsv'
target_text_df = pd.read_table(target_text_file, sep='\t', header=None)
target_text_df.columns = ['id', 'title', 'body']
target_text_df = target_text_df.dropna()
target_text_df['title'] = target_text_df['title'].apply(lambda words: ' '.join(filter(lambda x: x not in stop, words.split())))
target_text_df['body'] = target_text_df['body'].apply(lambda words: ' '.join(filter(lambda x: x not in stop, words.split())))
target_text_df['title_len'] = target_text_df['title'].apply(lambda words: len(tokenizer.tokenize(str(words))))
target_text_df['body_len'] = target_text_df['body'].apply(lambda words: len(tokenizer.tokenize(str(words))))
target_text_df = target_text_df.dropna()
target_text_df = target_text_df.reset_index()

target_pos_file = DATA_PATH_TARGET + '/test.pos.txt'
target_pos_df = pd.read_table(target_pos_file, sep=' ', header=None)
target_pos_df.columns = ['id', 'pos']

target_neg_file = DATA_PATH_TARGET + '/test.neg.txt'
target_neg_df = pd.read_table(target_neg_file, sep=' ', header=None)
target_neg_df.columns = ['id', 'neg']
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

print "preparing data for doc2vec..."
source_corpus = list(read_corpus(source_text_df))

print "initializing doc2vec..."
#training parameters
num_epochs = 16 
lr_init= 0.025
lr_min = 0.001
lr_delta = (lr_init - lr_min) / float(num_epochs)
num_cores = multiprocessing.cpu_count()

model = gensim.models.doc2vec.Doc2Vec(dm=1, size=256, window=10, min_count=2, alpha=lr_init, min_alpha=lr_min, workers=num_cores) #PV-DM (distributed memory doc2vec)
print model

model.build_vocab(source_corpus)
print "vocab size: ", len(model.wv.vocab)
#TODO: create binary tree (for visualization)

print "doc2vec training..."
training_loss = []
for epoch in tqdm(range(num_epochs)):

    shuffle(source_corpus)
    model.train(source_corpus, total_examples=len(source_corpus), epochs=1, compute_loss=True)

    #update learning rate
    model.alpha -= lr_delta
    model.min_alpha = model.alpha

    #log training loss
    training_loss.append(model.get_latest_training_loss()) #NOTE: not implemented

    #save model
    model.save(DATA_PATH_SOURCE + '/domain_transfer.doc2vec')
#end for

print "computing doc2vec vectors for target data..."
target_doc2vec_dict = {}
for row_idx in tqdm(range(target_text_df.shape[0])):
    target_id = target_text_df.loc[row_idx, 'id']
    target_title_body = target_text_df.loc[row_idx,'title'] + ' ' + target_text_df.loc[row_idx,'body']
    target_title_body_tokens = gensim.utils.simple_preprocess(target_title_body)
    target_doc2vec = model.infer_vector(target_title_body_tokens) #NOTE: could be different
    target_doc2vec_dict[target_id] = target_doc2vec.reshape(1,-1) 
#end for

print "scoring similarity between target questions..."
y_true, y_pred = [], []
for row_idx in tqdm(range(target_pos_df.shape[0])):
    y_true.append(1) #true label (similar)
    
    q1_idx = target_pos_df.loc[row_idx,'id']
    q2_idx = target_pos_df.loc[row_idx,'pos']

    score = cosine_similarity(target_doc2vec_dict[q1_idx], target_doc2vec_dict[q2_idx])
    y_pred.append(score[0][0])
#end for

for row_idx in tqdm(range(target_neg_df.shape[0])):
    y_true.append(0) #true label (not similar)
    
    q1_idx = target_neg_df.loc[row_idx,'id']
    q2_idx = target_neg_df.loc[row_idx,'neg']

    score = cosine_similarity(target_doc2vec_dict[q1_idx], target_doc2vec_dict[q2_idx])
    y_pred.append(score[0][0])
#end for

roc_auc = roc_auc_score(y_true, y_pred)
print "area under ROC curve: ", roc_auc

fpr, tpr, thresholds = roc_curve(y_true, y_pred)

idx_fpr_thresh = np.where(fpr < 0.05)[0]
roc_auc_0p05fpr = auc(fpr[idx_fpr_thresh], tpr[idx_fpr_thresh])
print "ROC AUC(0.05): ", roc_auc_0p05fpr

y_df = pd.DataFrame()
y_df['y_pred'] = y_pred
y_df['y_true'] = y_true
bins = np.linspace(min(y_pred)-0.1, max(y_pred)+0.1, 100)

#generate plots
plt.figure()
plt.plot(fpr, tpr, c='b', lw=2.0, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], c='k', lw=2.0, linestyle='--')
plt.title('Doc2Vec Domain Transfer')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('../figures/domain_transfer_doc2vec.png')

plt.figure()
sns.distplot(y_df[y_df['y_true']==1]['y_pred'], bins, kde=True, norm_hist=True, color='b', label='pos class')
sns.distplot(y_df[y_df['y_true']==0]['y_pred'], bins, kde=True, norm_hist=True, color='r', label='neg class')
plt.xlim([0,1])
plt.legend(loc='upper right')
plt.ylabel('normalized histogram')
plt.title('pos and neg class separation')
plt.savefig('../figures/domain_transfer_doc2vec_hist.png')


