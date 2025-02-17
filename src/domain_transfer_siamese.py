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
from torch.optim.lr_scheduler import StepLR

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.autograd as autograd
from torch.autograd import Variable

from meter import AUCMeter 
from random import shuffle
from sklearn.metrics import auc
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
EMBEDDINGS_FILE = config.get('paths', 'glove_path')
MAX_TITLE_LEN = int(config.get('data_params', 'MAX_TITLE_LEN'))
MAX_BODY_LEN = int(config.get('data_params', 'MAX_BODY_LEN'))
TRAIN_SAMPLE_SIZE = int(config.get('data_params', 'TRAIN_SAMPLE_SIZE'))
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

def generate_train_data(data_frame, train_text_df, word_to_idx, tokenizer, num_samples, num_negative, type='pos'):

    if num_samples == -1:
        num_samples = data_frame.shape[0]

    source_dataset = []
    for idx in tqdm(range(num_samples)):
        query_id = data_frame.loc[idx, 'query_id']
        similar_id_list = map(int, data_frame.loc[idx, 'similar_id'].split(' '))
        random_id_list = map(int, data_frame.loc[idx, 'random_id'].split(' '))
    
        #query title and body tensor ids
        query_title = train_text_df[train_text_df['id'] == query_id].title.tolist() 
        query_body = train_text_df[train_text_df['id'] == query_id].body.tolist()
        query_title_tokens = tokenizer.tokenize(query_title[0])[:MAX_TITLE_LEN]
        query_body_tokens = tokenizer.tokenize(query_body[0])[:MAX_BODY_LEN]
        query_title_tensor_idx = get_tensor_idx(query_title_tokens, word_to_idx, MAX_TITLE_LEN) 
        query_body_tensor_idx = get_tensor_idx(query_body_tokens, word_to_idx, MAX_BODY_LEN)

        if (type == 'pos'):
            for similar_id in similar_id_list:
                sample = {}  #reset sample dictionary here
                sample['label'] = 0 #similar
                sample['q1_idx'] = query_id
                sample['q1_title'] = query_title_tensor_idx
                sample['q1_body'] = query_body_tensor_idx

                similar_title = train_text_df[train_text_df['id'] == similar_id].title.tolist() 
                similar_body = train_text_df[train_text_df['id'] == similar_id].body.tolist()
                similar_title_tokens = tokenizer.tokenize(similar_title[0])[:MAX_TITLE_LEN]
                similar_body_tokens = tokenizer.tokenize(similar_body[0])[:MAX_BODY_LEN]
                similar_title_tensor_idx = get_tensor_idx(similar_title_tokens, word_to_idx, MAX_TITLE_LEN) 
                similar_body_tensor_idx = get_tensor_idx(similar_body_tokens, word_to_idx, MAX_BODY_LEN)

                sample['q2_idx'] = similar_id
                sample['q2_title'] = similar_title_tensor_idx
                sample['q2_body'] = similar_body_tensor_idx
                source_dataset.append(sample)
            #end for
        elif (type == 'neg'):
            for random_id in random_id_list[:num_negative]:
                sample = {}  #reset sample dictionary here
                sample['label'] = 1 #different
                sample['q1_idx'] = query_id
                sample['q1_title'] = query_title_tensor_idx
                sample['q1_body'] = query_body_tensor_idx
       
                random_title = train_text_df[train_text_df['id'] == random_id].title.tolist() 
                random_body = train_text_df[train_text_df['id'] == random_id].body.tolist()
                random_title_tokens = tokenizer.tokenize(random_title[0])[:MAX_TITLE_LEN]
                random_body_tokens = tokenizer.tokenize(random_body[0])[:MAX_BODY_LEN]
                random_title_tensor_idx = get_tensor_idx(random_title_tokens, word_to_idx, MAX_TITLE_LEN) 
                random_body_tensor_idx = get_tensor_idx(random_body_tokens, word_to_idx, MAX_BODY_LEN)
                sample['q2_idx'] = random_id
                sample['q2_title'] = random_title_tensor_idx
                sample['q2_body'] = random_body_tensor_idx
                source_dataset.append(sample)
            #end for
        else:
            print "invalid type: enter either pos or neg!"
        #end if 
    #end for
    return source_dataset 

def generate_test_data(data_frame, train_text_df, word_to_idx, tokenizer):

    target_dataset = []
    #for idx in tqdm(range(data_frame.shape[0])): #TODO: uncomment
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


print "generating training data ..."
tic = time()
source_train_pos_data = generate_train_data(source_idx_df, source_text_df, word_to_idx, tokenizer, TRAIN_SAMPLE_SIZE, NUM_NEGATIVE, type='pos')
source_train_neg_data = generate_train_data(source_idx_df, source_text_df, word_to_idx, tokenizer, TRAIN_SAMPLE_SIZE, NUM_NEGATIVE, type='neg')

train_data_combined = source_train_pos_data + source_train_neg_data 
shuffle(train_data_combined) #permute randomly in-place 

num_source_pos = len(source_train_pos_data)
num_source_neg = len(source_train_neg_data)
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

print "generating test data..."
tic = time()
target_test_pos_data = generate_test_data(target_pos_df, target_text_df, word_to_idx, tokenizer)
target_test_neg_data = generate_test_data(target_neg_df, target_text_df, word_to_idx, tokenizer)
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

print "instantiating siamese LSTM model..."
#training parameters
num_epochs = 16 
batch_size = 256 

#RNN parameters
embed_dim = embeddings.shape[1] #200
hidden_size = 128 #hidden vector dim 
weight_decay = 1e-5 
learning_rate = 1e-3 

#RNN architecture
class SIAMESE_RNN(nn.Module):
    def __init__(self, embed_dim, hidden_size, vocab_size, batch_size):
        super(SIAMESE_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding_layer = nn.Embedding(vocab_size, embed_dim) 
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.embedding_layer.weight.requires_grad = False 
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=1, 
                            bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 2, int(hidden_size * 4.0/3.0))
        self.fc2 = nn.Linear(int(hidden_size * 4.0/3.0), hidden_size)
        self.dropout = nn.Dropout(0.5) #prob of dropout = (1 - keep prob)  
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
        x = F.relu(self.fc1(h_avg_pool))
        x = self.dropout(x)
        x = self.fc2(x)
        return x 

use_gpu = torch.cuda.is_available()

model = SIAMESE_RNN(embed_dim, hidden_size, len(word_to_idx), batch_size)
if use_gpu:
    print "found CUDA GPU..."
    model = model.cuda()
    
print model

#define loss and optimizer
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        contrastive_loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),2))

        return contrastive_loss


class_weights = np.array([num_source_pos, num_source_neg], dtype=np.float32) #TODO: check order
class_weights = sum(class_weights) / class_weights
class_weights_tensor = torch.from_numpy(class_weights)
if use_gpu:
    class_weights_tensor = class_weights_tensor.cuda()
print "class weights: ", class_weights

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
tot_num_params = sum([np.prod(p.size()) for p in model_parameters])
print "number of trainable params: ", tot_num_params

criterion = ContrastiveLoss(margin=0.5) 
optimizer = torch.optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=4, gamma=0.5) #half learning rate every 4 epochs

training_loss = []

print "training..."
for epoch in range(num_epochs):
    
    running_train_loss = 0.0

    train_data_loader = torch.utils.data.DataLoader(
        train_data_combined, 
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4, 
        drop_last = True)
        
    model.train()
    scheduler.step() 
        
    for batch in tqdm(train_data_loader):
     
        q1_idx = batch['q1_idx']
        q1_title = Variable(batch['q1_title'])
        q1_body = Variable(batch['q1_body'])
       
        q2_idx = batch['q2_idx']
        q2_title = Variable(batch['q2_title'])
        q2_body = Variable(batch['q2_body'])

        label = Variable(batch['label'].view(-1,1).type(torch.FloatTensor))

        if use_gpu:
            label = label.cuda()
            q1_title, q1_body = q1_title.cuda(), q1_body.cuda()
            q2_title, q2_body = q2_title.cuda(), q2_body.cuda()

        optimizer.zero_grad() 

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

        #similar body
        model.hidden = model.init_hidden() 
        if use_gpu:
            model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
        lstm_q2_body = model(q2_body)

        lstm_q2 = (lstm_q2_title + lstm_q2_body)/2.0

        loss = criterion(lstm_q1, lstm_q2, label) 
        
        loss.backward() 
        optimizer.step()
        running_train_loss += loss.cpu().data[0]        
    #end for
    training_loss.append(running_train_loss)
    print "epoch: %4d, training loss: %.4f" %(epoch+1, running_train_loss)
    
    torch.save(model, SAVE_PATH + '/domain_transfer_siamese.pt')
#end for
"""
print "loading pre-trained model..."
model = torch.load(SAVE_PATH + '/domain_transfer_siamese.pt')
if use_gpu:
    print "found CUDA GPU..."
    model = model.cuda()
"""

print "scoring similarity between target questions..."
y_true, y_pred_lstm = [], []
auc_meter = AUCMeter()

test_data_loader_pos = torch.utils.data.DataLoader(
    target_test_pos_data, 
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

    euclidean_distance = F.pairwise_distance(lstm_q1, lstm_q2) 
    score_pos_numpy = euclidean_distance.cpu().data.numpy().ravel() 

    y_true.extend(np.zeros(batch_size)) #true label (similar) 
    y_pred_lstm.extend(score_pos_numpy.tolist())
    auc_meter.add(score_pos_numpy, np.zeros(batch_size))
#end for        

test_data_loader_neg = torch.utils.data.DataLoader(
    target_test_neg_data, 
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

    euclidean_distance = F.pairwise_distance(lstm_q1, lstm_q2) 
    score_neg_numpy = euclidean_distance.cpu().data.numpy().ravel() 

    y_true.extend(np.ones(batch_size)) #true label (not similar)
    y_pred_lstm.extend(score_neg_numpy.tolist())
    auc_meter.add(score_neg_numpy, np.ones(batch_size))
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
plt.title('Siamese LSTM Domain Transfer')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('../figures/domain_transfer_siamese.png')

plt.figure()
plt.plot(training_loss, label='loss')
plt.title("Siamese LSTM Domain Transfer Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('../figures/domain_transfer_siamese_loss.png')

plt.figure()
sns.distplot(y_df[y_df['y_true']==1]['y_pred'], bins, kde=True, norm_hist=True, color='b', label='pos class')
sns.distplot(y_df[y_df['y_true']==0]['y_pred'], bins, kde=True, norm_hist=True, color='r', label='neg class')
plt.xlim([0,1])
plt.legend(loc='upper right')
plt.ylabel('normalized histogram')
plt.title('pos and neg class separation')
plt.savefig('../figures/domain_transfer_direct_lstm_hist.png')

#save for plotting
figures_da_siamese = {}
figures_da_siamese['siamese_ytrue'] = y_true 
figures_da_siamese['siamese_ypred'] = y_pred_lstm 
figures_da_siamese['siamese_roc_auc'] = roc_auc
figures_da_siamese['siamese_auc_meter'] = roc_auc_0p05fpr_meter
figures_da_siamese['siamese_auc_sklearn'] = roc_auc_0p05fpr
figures_da_siamese['siamese_training_loss'] = training_loss

filename = SAVE_PATH + 'figures_da_siamese.dat' 
with open(filename, 'w') as f:
    pickle.dump(figures_da_siamese, f)



