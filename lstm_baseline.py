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
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(0)
#torch.manual_seed(0)

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
print "vocab size (embeddings): ", len(word_to_idx)
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

"""
print "fitting tf-idf vectorizer..."
tic = time()
tfidf = TfidfVectorizer(tokenizer=tokenizer.tokenize, analyzer='word', ngram_range=(1,1))
tfidf.fit(train_text_df['title'].tolist() + train_text_df['body'].tolist())
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

vocab = tfidf.vocabulary_
print "vocab size: ", len(vocab)
print "embeddings size: ", embeddings.shape
"""

print "generating training, validation, test datasets..."
tic = time()
train_data = []
for idx in tqdm(range(10)):
#for idx in tqdm(range(train_idx_df.shape[0])):
    query_id = train_idx_df.loc[idx, 'query_id']
    similar_id_list = map(int, train_idx_df.loc[idx, 'similar_id'].split(' '))
    random_id_list = map(int, train_idx_df.loc[idx, 'random_id'].split(' '))
    
    #query title and body tensor ids
    query_title = train_text_df[train_text_df['id'] == query_id].title.tolist() 
    query_body = train_text_df[train_text_df['id'] == query_id].body.tolist()
    query_title_tokens = tokenizer.tokenize(query_title[0])[:MAX_TITLE_LEN]
    query_body_tokens = tokenizer.tokenize(query_body[0])[:MAX_BODY_LEN]
    query_title_tensor_idx = get_tensor_idx(query_title_tokens, word_to_idx, MAX_TITLE_LEN) 
    query_body_tensor_idx = get_tensor_idx(query_body_tokens, word_to_idx, MAX_BODY_LEN) 

    for similar_id in similar_id_list:
        sample = {}  #reset sample dictionary here
        sample['query_title'] = query_title_tensor_idx
        sample['query_body'] = query_body_tensor_idx

        similar_title = train_text_df[train_text_df['id'] == similar_id].title.tolist() 
        similar_body = train_text_df[train_text_df['id'] == similar_id].body.tolist()
        similar_title_tokens = tokenizer.tokenize(similar_title[0])[:MAX_TITLE_LEN]
        similar_body_tokens = tokenizer.tokenize(similar_body[0])[:MAX_BODY_LEN]
        similar_title_tensor_idx = get_tensor_idx(similar_title_tokens, word_to_idx, MAX_TITLE_LEN) 
        similar_body_tensor_idx = get_tensor_idx(similar_body_tokens, word_to_idx, MAX_BODY_LEN)
        sample['similar_title'] = similar_title_tensor_idx
        sample['similar_body'] = similar_body_tensor_idx

        for ridx, random_id in enumerate(random_id_list):
            random_title_name = 'random_title_' + str(ridx)
            random_body_name = 'random_body_' + str(ridx)
        
            random_title = train_text_df[train_text_df['id'] == random_id].title.tolist() 
            random_body = train_text_df[train_text_df['id'] == random_id].body.tolist()
            random_title_tokens = tokenizer.tokenize(random_title[0])[:MAX_TITLE_LEN]
            random_body_tokens = tokenizer.tokenize(random_body[0])[:MAX_BODY_LEN]
            random_title_tensor_idx = get_tensor_idx(random_title_tokens, word_to_idx, MAX_TITLE_LEN) 
            random_body_tensor_idx = get_tensor_idx(random_body_tokens, word_to_idx, MAX_BODY_LEN)
            sample[random_title_name] = random_title_tensor_idx
            sample[random_body_name] = random_body_tensor_idx
        #end for
        train_data.append(sample)
    #end for
#end for
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)


#training parameters
num_epochs = 32 
batch_size = 8 

#model parameters
embed_dim = embeddings.shape[1] #200
hidden_dim = embed_dim / 2 
weight_decay = 1e-3 
learning_rate = 1e-3 

#RNN architecture
class RNN(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, vocab_size):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
    
    def init_hidden(self):
        #[num_layers, minibatch_size, hidden_dim]
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, x_idx):
        all_x = self.embedding_layer(x_idx)   
        lstm_out, self.hidden = self.lstm(all_x.view(len(x_idx),1,-1), self.hidden)
        return self.hidden 
        
use_gpu = torch.cuda.is_available()

model = RNN(embed_dim, hidden_dim, len(word_to_idx))
if use_gpu:
    print "found CUDA GPU..."
    model = model.cuda()
    
print model

#define loss and optimizer
criterion = nn.MarginRankingLoss(margin=1, size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)













