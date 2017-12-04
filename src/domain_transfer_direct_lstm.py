import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import ConfigParser
from tqdm import tqdm
from time import time
import cPickle as pickle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.autograd as autograd
from torch.autograd import Variable

from meter import AUCMeter 
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(0)

DATA_PATH_SOURCE = '/data/vision/fisher/data1/vsmolyakov/nlp_project/data/askubuntu/'
DATA_PATH_TARGET = '/data/vision/fisher/data1/vsmolyakov/nlp_project/data/android/'

tokenizer = RegexpTokenizer(r'\w+')
stop = set(stopwords.words('english'))

config = ConfigParser.ConfigParser()
config.readfp(open(r'config.ini'))
SAVE_PATH = config.get('paths', 'save_path')
RNN_SAVE_NAME = config.get('rnn_params', 'save_name')
CNN_SAVE_NAME = config.get('cnn_params', 'save_name')
EMBEDDINGS_FILE = config.get('paths', 'embeddings_path')
MAX_TITLE_LEN = int(config.get('data_params', 'MAX_TITLE_LEN'))
MAX_BODY_LEN = int(config.get('data_params', 'MAX_BODY_LEN'))
#TODO: do we keep the same title and body len for android dataset?

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
        
def generate_data(data_frame, train_text_df, word_to_idx, tokenizer):

    dataset = []
    for idx in tqdm(range(data_frame.shape[0])):
    #for idx in tqdm(range(1000)):
        q1_id = data_frame.loc[idx, 'id_1']
        q2_id = data_frame.loc[idx, 'id_2']

        #q1 title and body tensor ids
        q1_title = train_text_df[train_text_df['id'] == q1_id].title.tolist() 
        q1_body = train_text_df[train_text_df['id'] == q1_id].body.tolist()
        q1_title_tokens = tokenizer.tokenize(q1_title[0])[:MAX_TITLE_LEN]
        q1_body_tokens = tokenizer.tokenize(q1_body[0])[:MAX_BODY_LEN]
        q1_title_tensor_idx = get_tensor_idx(q1_title_tokens, word_to_idx, MAX_TITLE_LEN) 
        q1_body_tensor_idx = get_tensor_idx(q1_body_tokens, word_to_idx, MAX_BODY_LEN)

        #q2 title and body tensor ids
        q2_title = train_text_df[train_text_df['id'] == q2_id].title.tolist() 
        q2_body = train_text_df[train_text_df['id'] == q2_id].body.tolist()
        q2_title_tokens = tokenizer.tokenize(q2_title[0])[:MAX_TITLE_LEN]
        q2_body_tokens = tokenizer.tokenize(q2_body[0])[:MAX_BODY_LEN]
        q2_title_tensor_idx = get_tensor_idx(q2_title_tokens, word_to_idx, MAX_TITLE_LEN) 
        q2_body_tensor_idx = get_tensor_idx(q2_body_tokens, word_to_idx, MAX_BODY_LEN)

        sample = {}  #reset sample dictionary here
        sample['q1_idx'] = q1_id
        sample['q1_title'] = q1_title_tensor_idx
        sample['q1_body'] = q1_body_tensor_idx

        sample['q2_idx'] = q2_id
        sample['q2_title'] = q2_title_tensor_idx
        sample['q2_body'] = q2_body_tensor_idx

        dataset.append(sample)
    #end for
    return dataset 

#load data
print "loading data..."
tic = time()
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
target_pos_df.columns = ['id_1', 'id_2']

target_neg_file = DATA_PATH_TARGET + '/test.neg.txt'
target_neg_df = pd.read_table(target_neg_file, sep=' ', header=None)
target_neg_df.columns = ['id_1', 'id_2']
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)


print "loading embeddings..."
tic = time()
embeddings, word_to_idx = get_embeddings()
print "vocab size (embeddings): ", len(word_to_idx)
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

print "generating test pos and neg datasets..."
tic = time()
target_pos_data = generate_data(target_pos_df, target_text_df, word_to_idx, tokenizer)
target_neg_data = generate_data(target_neg_df, target_text_df, word_to_idx, tokenizer)
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

print "loading LSTM model pre-trained on source dataset..."
#RNN architecture
class RNN(nn.Module):
    
    def __init__(self, embed_dim, hidden_size, vocab_size, batch_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        #TODO: ignore loss computations on 0 embedding index inputs 
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim) 
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.embedding_layer.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        #[num_layers, batch_size, hidden_size] for (h_n, c_n)
        return (Variable(torch.zeros(2, self.batch_size, self.hidden_size)),
                Variable(torch.zeros(2, self.batch_size, self.hidden_size)))

    def forward(self, x_idx):
        all_x = self.embedding_layer(x_idx)
        #[batch_size, seq_length (num_words), embed_dim]
        lstm_out, self.hidden = self.lstm(all_x.view(self.batch_size, x_idx.size(1), -1), self.hidden)
        h_avg_pool = torch.mean(lstm_out, dim=1)          #average pooling
        #h_n, c_n = self.hidden[0], self.hidden[1]        #last pooling
        #h_last_pool = torch.cat([h_n[0], h_n[1]], dim=1) #[batch_size, 2 x hidden_size] 

        return h_avg_pool 

#RNN parameters
batch_size = 32 
embed_dim = embeddings.shape[1] #200
hidden_size = 128 #hidden vector dim 
model = RNN(embed_dim, hidden_size, len(word_to_idx), batch_size)

#RNN weights
model = torch.load(SAVE_PATH + RNN_SAVE_NAME)

batch_size = model.batch_size
hidden_size = model.hidden_size
print "pre-trained LSTM batch-size: ", batch_size
print "pre-trained LSTM hidden-size: ", hidden_size

use_gpu = torch.cuda.is_available()
if use_gpu:
    print "found CUDA GPU..."
    model = model.cuda()

model.eval()

print "scoring similarity between target questions..."
y_true, y_pred_lstm = [], []
auc_meter = AUCMeter()

test_data_loader_pos = torch.utils.data.DataLoader(
    target_pos_data, 
    batch_size = batch_size,
    shuffle = False,
    num_workers = 4, 
    drop_last = True)
        
for batch in tqdm(test_data_loader_pos):

    q1_idx = batch['q1_idx']
    q1_title = Variable(batch['q1_title'])
    q1_body = Variable(batch['q1_body'])

    q2_idx = batch['q2_idx']
    q2_title = Variable(batch['q2_title'])
    q2_body = Variable(batch['q2_body'])

    if use_gpu:
        q1_title, q1_body = q1_title.cuda(), q1_body.cuda()
        q2_title, q2_body = q2_title.cuda(), q2_body.cuda()

    #q1 title
    model.hidden = model.init_hidden() 
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
    lstm_q1_title = model(q1_title)

    #q1 body
    model.hidden = model.init_hidden() 
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
    lstm_q1_body = model(q1_body)

    lstm_q1 = (lstm_q1_title + lstm_q1_body)/2.0

    #q2 title
    model.hidden = model.init_hidden() 
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
    lstm_q2_title = model(q2_title)

    #q2 body
    model.hidden = model.init_hidden() 
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
    lstm_q2_body = model(q2_body)

    lstm_q2 = (lstm_q2_title + lstm_q2_body)/2.0

    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    score_pos = cosine_similarity(lstm_q1, lstm_q2)

    score_pos_numpy = score_pos.cpu().data.numpy() #TODO: check (some scores are negative)

    y_true.extend(np.ones(batch_size)) #true label (similar)
    y_pred_lstm.extend(score_pos_numpy.tolist())
    auc_meter.add(score_pos_numpy, np.ones(batch_size))
#end for        

test_data_loader_neg = torch.utils.data.DataLoader(
    target_neg_data, 
    batch_size = batch_size,
    shuffle = False,
    num_workers = 4, 
    drop_last = True)
        
for batch in tqdm(test_data_loader_neg):

    q1_idx = batch['q1_idx']
    q1_title = Variable(batch['q1_title'])
    q1_body = Variable(batch['q1_body'])

    q2_idx = batch['q2_idx']
    q2_title = Variable(batch['q2_title'])
    q2_body = Variable(batch['q2_body'])

    if use_gpu:
        q1_title, q1_body = q1_title.cuda(), q1_body.cuda()
        q2_title, q2_body = q2_title.cuda(), q2_body.cuda()

    #q1 title
    model.hidden = model.init_hidden() 
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
    lstm_q1_title = model(q1_title)

    #q1 body
    model.hidden = model.init_hidden() 
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
    lstm_q1_body = model(q1_body)

    lstm_q1 = (lstm_q1_title + lstm_q1_body)/2.0

    #q2 title
    model.hidden = model.init_hidden() 
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
    lstm_q2_title = model(q2_title)

    #q2 body
    model.hidden = model.init_hidden() 
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
    lstm_q2_body = model(q2_body)

    lstm_q2 = (lstm_q2_title + lstm_q2_body)/2.0

    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    score_neg = cosine_similarity(lstm_q1, lstm_q2)

    score_neg_numpy = score_neg.cpu().data.numpy() #TODO: check (some scores are negative)

    y_true.extend(np.zeros(batch_size)) #true label (not similar)
    y_pred_lstm.extend(score_neg_numpy.tolist())
    auc_meter.add(score_neg_numpy, np.zeros(batch_size))
#end for        

roc_auc = roc_auc_score(y_true, y_pred_lstm)
print "area under ROC curve: ", roc_auc

fpr, tpr, thresholds = roc_curve(y_true, y_pred_lstm)

idx_fpr_thresh = np.where(fpr < 0.05)[0]
roc_auc_0p05fpr = auc(fpr[idx_fpr_thresh], tpr[idx_fpr_thresh])
print "ROC AUC(0.05) sklearn: ", roc_auc_0p05fpr

roc_auc_0p05fpr_meter = auc_meter.value(0.05)
print "ROC AUC(0.05) meter: ", roc_auc_0p05fpr_meter

y_df = pd.DataFrame()
y_df['y_pred'] = y_pred_lstm
y_df['y_true'] = y_true
bins = np.linspace(min(y_pred_lstm)-0.1, max(y_pred_lstm)+0.1, 100)

#generate plots
plt.figure()
plt.plot(fpr, tpr, c='b', lw=2.0, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], c='k', lw=2.0, linestyle='--')
plt.title('LSTM Domain Transfer Direct')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('../figures/domain_transfer_direct_lstm.png')

plt.figure()
sns.distplot(y_df[y_df['y_true']==1]['y_pred'], bins, kde=True, norm_hist=True, color='b', label='pos class')
sns.distplot(y_df[y_df['y_true']==0]['y_pred'], bins, kde=True, norm_hist=True, color='r', label='neg class')
plt.xlim([0,1])
plt.legend(loc='upper right')
plt.ylabel('normalized histogram')
plt.title('pos and neg class separation')
plt.savefig('../figures/domain_transfer_direct_lstm_hist.png')



