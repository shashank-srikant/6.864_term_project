import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import ConfigParser
from tqdm import tqdm
from time import time
import cPickle as pickle
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from meter import AUCMeter 
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(0)

DATA_PATH_SOURCE = '/data/vision/fisher/data1/vsmolyakov/nlp_project/data/askubuntu/'
DATA_PATH_TARGET = '/data/vision/fisher/data1/vsmolyakov/nlp_project/data/android/'

config = ConfigParser.ConfigParser()
config.readfp(open(r'../src/config.ini'))
SAVE_PATH = config.get('paths', 'save_path')

tokenizer = RegexpTokenizer(r'\w+')
stop = set(stopwords.words('english'))

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

target_text_file = DATA_PATH_TARGET + '/corpus.tsv'
target_text_df = pd.read_table(target_text_file, sep='\t', header=None)
target_text_df.columns = ['id', 'title', 'body']
target_text_df = target_text_df.dropna()
target_text_df['title'] = target_text_df['title'].apply(lambda words: ' '.join(filter(lambda x: x not in stop, words.split())))
target_text_df['body'] = target_text_df['body'].apply(lambda words: ' '.join(filter(lambda x: x not in stop, words.split())))
target_text_df['title_len'] = target_text_df['title'].apply(lambda words: len(tokenizer.tokenize(str(words))))
target_text_df['body_len'] = target_text_df['body'].apply(lambda words: len(tokenizer.tokenize(str(words))))

target_pos_file = DATA_PATH_TARGET + '/test.pos.txt'
target_pos_df = pd.read_table(target_pos_file, sep=' ', header=None)
target_pos_df.columns = ['id', 'pos']

target_neg_file = DATA_PATH_TARGET + '/test.neg.txt'
target_neg_df = pd.read_table(target_neg_file, sep=' ', header=None)
target_neg_df.columns = ['id', 'neg']
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

print "fitting tf-idf vectorizer on source data..."
tic = time()
tfidf = TfidfVectorizer(tokenizer=tokenizer.tokenize, analyzer='word', ngram_range=(1,1), max_df=0.8, min_df=2, stop_words='english')
tfidf.fit(source_text_df['title'].tolist() + source_text_df['body'].tolist())
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)
vocab = tfidf.vocabulary_
print "source vocab size: ", len(vocab)

print "computing tf-idf vectors for target data..."
target_tfidf_dict = {}
for row_idx in tqdm(range(target_text_df.shape[0])):
    target_id = target_text_df.loc[row_idx, 'id']
    target_title_body = target_text_df.loc[row_idx,'title'] + ' ' + target_text_df.loc[row_idx,'body']
    target_tfidf = tfidf.transform([target_title_body])
    target_tfidf_dict[target_id] = target_tfidf
#end for

print "scoring similarity between target questions..."
y_true, y_pred = [], []
auc_meter = AUCMeter()
for row_idx in tqdm(range(target_pos_df.shape[0])):
    y_true.append(1) #true label (similar)
    
    q1_idx = target_pos_df.loc[row_idx,'id']
    q2_idx = target_pos_df.loc[row_idx,'pos']

    score = cosine_similarity(target_tfidf_dict[q1_idx], target_tfidf_dict[q2_idx])
    y_pred.append(score[0][0])
    auc_meter.add(np.array([score[0][0]]),np.array([1]))
#end for

for row_idx in tqdm(range(target_neg_df.shape[0])):
    y_true.append(0) #true label (not similar)
    
    q1_idx = target_neg_df.loc[row_idx,'id']
    q2_idx = target_neg_df.loc[row_idx,'neg']

    score = cosine_similarity(target_tfidf_dict[q1_idx], target_tfidf_dict[q2_idx])
    y_pred.append(score[0][0])
    auc_meter.add(np.array([score[0][0]]),np.array([0]))
#end for

roc_auc = roc_auc_score(y_true, y_pred)
print "area under ROC curve: ", roc_auc

fpr, tpr, thresholds = roc_curve(y_true, y_pred)

idx_fpr_thresh = np.where(fpr < 0.05)[0]
roc_auc_0p05fpr = auc(fpr[idx_fpr_thresh], tpr[idx_fpr_thresh])
print "ROC AUC(0.05) sklearn: ", roc_auc_0p05fpr

roc_auc_0p05fpr_meter = auc_meter.value(0.05)
print "ROC AUC(0.05) meter: ", roc_auc_0p05fpr_meter

y_df = pd.DataFrame()
y_df['y_pred'] = y_pred
y_df['y_true'] = y_true
bins = np.linspace(min(y_pred)-0.1, max(y_pred)+0.1, 100)

#generate plots
plt.figure()
plt.plot(fpr, tpr, c='b', lw=2.0, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], c='k', lw=2.0, linestyle='--')
plt.title('TF-IDF Domain Transfer')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('../figures/domain_transfer_tfidf.png')

plt.figure()
sns.distplot(y_df[y_df['y_true']==1]['y_pred'], bins, kde=True, norm_hist=True, color='b', label='pos class')
sns.distplot(y_df[y_df['y_true']==0]['y_pred'], bins, kde=True, norm_hist=True, color='r', label='neg class')
plt.xlim([0,1])
plt.legend(loc='upper right')
plt.ylabel('normalized histogram')
plt.title('pos and neg class separation')
plt.savefig('../figures/domain_transfer_tfidf_hist.png')

#save for plotting
figures_da_tfidf = {}
figures_da_tfidf['tfidf_ytrue'] = y_true 
figures_da_tfidf['tfidf_ypred'] = y_pred 
figures_da_tfidf['tfidf_roc_auc'] = roc_auc
figures_da_tfidf['tfidf_auc_meter'] = roc_auc_0p05fpr_meter
figures_da_tfidf['tfidf_auc_sklearn'] = roc_auc_0p05fpr


filename = SAVE_PATH + 'figures_da_tfidf.dat' 
with open(filename, 'w') as f:
    pickle.dump(figures_da_tfidf, f)


