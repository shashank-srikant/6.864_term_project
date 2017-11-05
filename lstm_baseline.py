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

from tqdm import tqdm
from time import time
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(0)


DATA_PATH = '/data/vision/fisher/data1/vsmolyakov/nlp_project/data/askubuntu/'

SAVE_PATH = './lstm_baseline.pt' 
EMBEDDINGS_FILE = DATA_PATH + '/vector/vectors_pruned.200.txt'
MAX_TITLE_LEN = 20
MAX_BODY_LEN = 100  #max number of words per sentence

tokenizer = RegexpTokenizer(r'\w+')

def get_embeddings():
    lines = []
    with open(EMBEDDINGS_FILE, 'r') as f:
        lines = f.readlines()
        f.close()
    
    embedding_tensor = []
    word_to_idx = {}
    
    for idx, l in enumerate(lines):
        word, emb = l.split()[0], l.split()[1:]
        vector = [float(x) for x in emb]
        if idx == 0: #reserved
            embedding_tensor.append(np.zeros(len(vector)))
        embedding_tensor.append(vector)
        word_to_idx[word] = idx+1
    #end for
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)    
    return embedding_tensor, word_to_idx
        
def get_tensor_idx(text, word_to_idx, max_len):
    null_idx = 0  #idx if word is not in the embeddings dictionary
    text_idx = [word_to_idx[x] if x in word_to_idx else null_idx for x in text][:max_len]
    if len(text_idx) < max_len:
        text_idx.extend([null_idx for _ in range(max_len - len(text_idx))])    
    x = torch.LongTensor(text_idx)  #64-bit integer
    return x
        

#load data
print "loading data..."
tic = time()
train_text_file = DATA_PATH + '/text_tokenized.txt'
train_text_df = pd.read_table(train_text_file, sep='\t', header=None)
train_text_df.columns = ['id', 'title', 'body']
train_text_df = train_text_df.dropna()
train_text_df['title_len'] = train_text_df['title'].apply(lambda words: len(tokenizer.tokenize(str(words))))
train_text_df['body_len'] = train_text_df['body'].apply(lambda words: len(tokenizer.tokenize(str(words))))

train_idx_file = DATA_PATH + '/train_random.txt' 
train_idx_df = pd.read_table(train_idx_file, sep='\t', header=None)
train_idx_df.columns = ['query_id', 'similar_id', 'random_id']

dev_idx_file = DATA_PATH + '/dev.txt'
dev_idx_df = pd.read_table(dev_idx_file, sep='\t', header=None)
dev_idx_df.columns = ['query_id', 'similar_id', 'retrieved_id', 'bm25_score']
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

print "loading embeddings..."
tic = time()
embeddings, word_to_idx = get_embeddings()
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)


#visualize data
f, (ax1, ax2) = plt.subplots(1, 2)
sns.distplot(train_text_df['title_len'], hist=True, kde=True, color='b', label='title len', ax=ax1)
sns.distplot(train_text_df[train_text_df['body_len'] < 256]['body_len'], hist=True, kde=True, color='r', label='body len', ax=ax2)
ax1.axvline(x=MAX_TITLE_LEN, color='k', linestyle='--', label='max len')
ax2.axvline(x=MAX_BODY_LEN, color='k', linestyle='--', label='max len')
ax1.set_title('title length histogram'); ax1.legend(loc=1); 
ax2.set_title('body length histogram'); ax2.legend(loc=1);
plt.savefig('./figures/question_len_hist.png')

print "fitting tf-idf vectorizer..."
tic = time()
tfidf = TfidfVectorizer(tokenizer=tokenizer.tokenize, analyzer='word', ngram_range=(1,1))
tfidf.fit(train_text_df['title'].tolist() + train_text_df['body'].tolist())
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

vocab = tfidf.vocabulary_
print "vocab size: ", len(vocab)
print "embeddings size: ", embeddings.shape




