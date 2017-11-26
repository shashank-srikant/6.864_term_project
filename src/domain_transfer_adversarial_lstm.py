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

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(0)

DATA_PATH_SOURCE = '/data/vision/fisher/data1/vsmolyakov/nlp_project/data/askubuntu/'
DATA_PATH_TARGET = '/data/vision/fisher/data1/vsmolyakov/nlp_project/data/android/'

tokenizer = RegexpTokenizer(r'\w+')
stop = set(stopwords.words('english'))

NUM_CLASSES = 2
config = ConfigParser.ConfigParser()
config.readfp(open(r'config.ini'))
SAVE_PATH = config.get('paths', 'save_path')
RNN_SAVE_NAME = config.get('rnn_params', 'save_name')
CNN_SAVE_NAME = config.get('cnn_params', 'save_name')
EMBEDDINGS_FILE = config.get('paths', 'embeddings_path')
MAX_TITLE_LEN = int(config.get('data_params', 'MAX_TITLE_LEN'))
MAX_BODY_LEN = int(config.get('data_params', 'MAX_BODY_LEN'))
NUM_NEGATIVE = int(config.get('data_params', 'NUM_NEGATIVE'))
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

#source_pos_data = generate_data_source(source_idx_df, source_text_df, word_to_idx, tokenizer, type='pos')
def generate_data_source(data_frame, train_text_df, word_to_idx, tokenizer, num_negative, type='pos'):

    source_dataset = []
    #for idx in tqdm(range(data_frame.shape[0])):
    for idx in tqdm(range(100)):
        query_id = data_frame.loc[idx, 'query_id']
        similar_id_list = map(int, data_frame.loc[idx, 'similar_id'].split(' '))
        random_id_list = map(int, data_frame.loc[idx, 'random_id'].split(' '))
    
        #query title and body tensor ids
        q1_title = train_text_df[train_text_df['id'] == query_id].title.tolist() 
        q1_body = train_text_df[train_text_df['id'] == query_id].body.tolist()
        q1_title_tokens = tokenizer.tokenize(q1_title[0])[:MAX_TITLE_LEN]
        q1_body_tokens = tokenizer.tokenize(q1_body[0])[:MAX_BODY_LEN]
        q1_title_tensor_idx = get_tensor_idx(q1_title_tokens, word_to_idx, MAX_TITLE_LEN) 
        q1_body_tensor_idx = get_tensor_idx(q1_body_tokens, word_to_idx, MAX_BODY_LEN)

        if (type == 'pos'):
            for similar_id in similar_id_list:
                sample = {}  #reset sample dictionary here
                sample['q1_idx'] = query_id
                sample['q1_title'] = q1_title_tensor_idx
                sample['q1_body'] = q1_body_tensor_idx

                q2_title = train_text_df[train_text_df['id'] == similar_id].title.tolist() 
                q2_body = train_text_df[train_text_df['id'] == similar_id].body.tolist()
                q2_title_tokens = tokenizer.tokenize(q2_title[0])[:MAX_TITLE_LEN]
                q2_body_tokens = tokenizer.tokenize(q2_body[0])[:MAX_BODY_LEN]
                q2_title_tensor_idx = get_tensor_idx(q2_title_tokens, word_to_idx, MAX_TITLE_LEN) 
                q2_body_tensor_idx = get_tensor_idx(q2_body_tokens, word_to_idx, MAX_BODY_LEN)
                
                sample['q2_idx'] = similar_id
                sample['q2_title'] = q2_title_tensor_idx
                sample['q2_body'] = q2_body_tensor_idx
                sample['domain_label'] = 0  #source domain
                
                source_dataset.append(sample)
            #end for
        elif (type == 'neg'):
            for random_id in random_id_list[:num_negative]:
                sample = {} #reset sample dictionary here
                sample['q1_idx'] = query_id
                sample['q1_title'] = q1_title_tensor_idx
                sample['q1_body'] = q1_body_tensor_idx

                q2_title = train_text_df[train_text_df['id'] == random_id].title.tolist() 
                q2_body = train_text_df[train_text_df['id'] == random_id].body.tolist()
                
                if (len(q2_title) > 0 and len(q2_body) > 0):
                    q2_title_tokens = tokenizer.tokenize(q2_title[0])[:MAX_TITLE_LEN]
                    q2_body_tokens = tokenizer.tokenize(q2_body[0])[:MAX_BODY_LEN]
                    q2_title_tensor_idx = get_tensor_idx(q2_title_tokens, word_to_idx, MAX_TITLE_LEN) 
                    q2_body_tensor_idx = get_tensor_idx(q2_body_tokens, word_to_idx, MAX_BODY_LEN)
                    
                    sample['q2_idx'] = random_id
                    sample['q2_title'] = q2_title_tensor_idx
                    sample['q2_body'] = q2_body_tensor_idx
                    sample['domain_label'] = 0  #source domain
                
                    source_dataset.append(sample)
                #end if 
            #end for
        else:
            print "Incorrect type: specify either pos or neg type!"
            break
        #end if
    #end for
    return source_dataset 


def generate_data_target(data_frame, train_text_df, word_to_idx, tokenizer):

    target_dataset = []
    #for idx in tqdm(range(data_frame.shape[0])):
    for idx in tqdm(range(1000)):
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

        sample['domain_label'] = 1  #target domain
        target_dataset.append(sample)
    #end for
    return target_dataset 

#load data
print "loading source data..."
tic = time()
source_text_file = DATA_PATH_SOURCE + '/text_tokenized.txt'
source_text_df = pd.read_table(source_text_file, sep='\t', header=None)
source_text_df.columns = ['id', 'title', 'body']
source_text_df = source_text_df.dropna()
source_text_df['title'] = source_text_df['title'].apply(lambda words: ' '.join(filter(lambda x: x not in stop, words.split())))
source_text_df['body'] = source_text_df['body'].apply(lambda words: ' '.join(filter(lambda x: x not in stop, words.split())))
source_text_df['title_len'] = source_text_df['title'].apply(lambda words: len(tokenizer.tokenize(str(words))))
source_text_df['body_len'] = source_text_df['body'].apply(lambda words: len(tokenizer.tokenize(str(words))))

source_idx_file = DATA_PATH_SOURCE + '/train_random.txt' 
source_idx_df = pd.read_table(source_idx_file, sep='\t', header=None)
source_idx_df.columns = ['query_id', 'similar_id', 'random_id']
source_idx_df = source_idx_df.dropna()
source_idx_df = source_idx_df.reset_index()
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

#load data
print "loading target data..."
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

print "generating (train) pos and neg source datasets..."
tic = time()
source_pos_data = generate_data_source(source_idx_df, source_text_df, word_to_idx, tokenizer, NUM_NEGATIVE, type='pos')
source_neg_data = generate_data_source(source_idx_df, source_text_df, word_to_idx, tokenizer, NUM_NEGATIVE, type='neg')
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

print "generating (test) pos and neg target datasets..."
tic = time()
target_pos_data = generate_data_target(target_pos_df, target_text_df, word_to_idx, tokenizer)
target_neg_data = generate_data_target(target_neg_df, target_text_df, word_to_idx, tokenizer)
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

print "loading LSTM model pre-trained on source dataset..."
#RNN architecture
class RNN(nn.Module):
    def __init__(self, embed_dim, hidden_size, vocab_size, batch_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding_layer = nn.Embedding(vocab_size, embed_dim) 
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=1, batch_first=True)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        #[num_layers, batch_size, hidden_size] for (h_n, c_n)
        return (Variable(torch.zeros(1, self.batch_size, self.hidden_size)),
                Variable(torch.zeros(1, self.batch_size, self.hidden_size)))

    def forward(self, x_idx):
        all_x = self.embedding_layer(x_idx)
        lstm_out, self.hidden = self.lstm(all_x.view(self.batch_size, x_idx.size(1), -1), self.hidden)
        #h_n dim: [1, batch_size, hidden_size]
        h_n, c_n = self.hidden[0], self.hidden[1]
        return h_n.squeeze(0)

#RNN parameters
batch_size = 32 
embed_dim = embeddings.shape[1] #200
hidden_size = 240 # number of LSTM cells 
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

print "creating gradient reversal layer..."
#GRL architecture
class GRL(nn.Module):
    def __init__(self, Lambda):
        super(GRL, self).__init__()
        self.Lambda = Lambda

    def forward(self, x):
        return x

    def backward(self, x):
        return -self.Lambda * x


print "instantiating domain classifier model..."
#domain classifier architecture
class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim  #2
 
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x 

#DNN parameters
dnn_input_dim = hidden_size
dnn_output_dim = NUM_CLASSES
domain_clf = DNN(dnn_input_dim, dnn_output_dim)
if use_gpu:
    print "found CUDA GPU..."
    domain_clf = domain_clf.cuda()

print "training..."








print "scoring similarity between target questions..."
y_true, y_pred_lstm = [], []

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
#end for        

roc_auc = roc_auc_score(y_true, y_pred_lstm)
print "area under ROC curve: ", roc_auc

fpr, tpr, thresholds = roc_curve(y_true, y_pred_lstm)

plt.figure()
plt.plot(fpr, tpr, c='b', lw=2.0, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], c='k', lw=2.0, linestyle='--')
plt.title('LSTM Domain Transfer Direct')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('../figures/domain_transfer_direct_lstm.png')



