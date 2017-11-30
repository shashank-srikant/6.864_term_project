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

np.random.seed(0)
#torch.manual_seed(0)

config = ConfigParser.ConfigParser()
config.readfp(open(r'config.ini'))

SAVE_PATH = config.get('paths', 'save_path')
DATA_FILE_NAME = config.get('paths', 'extracted_data_file_name')
TRAIN_TEST_FILE_NAME = config.get('paths', 'train_test_file_name')
SAVE_NAME = config.get('rnn_params', 'save_name')
NUM_NEGATIVE = int(config.get('data_params', 'NUM_NEGATIVE')) 

MAX_TITLE_LEN = int(config.get('data_params', 'MAX_TITLE_LEN'))
MAX_BODY_LEN = int(config.get('data_params', 'MAX_BODY_LEN'))

data_filename = SAVE_PATH + DATA_FILE_NAME
train_test_filename = SAVE_PATH + TRAIN_TEST_FILE_NAME

print "loading pickled data..."
tic = time()
with open(data_filename) as f:  
    train_text_df, train_idx_df, dev_idx_df, test_idx_df, embeddings, word_to_idx = pickle.load(f)
f.close()
with open(train_test_filename) as f:
    train_data, val_data, test_data = pickle.load(f)
f.close()
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

tokenizer = RegexpTokenizer(r'\w+')
stop = set(stopwords.words('english'))

#visualize data
f, (ax1, ax2) = plt.subplots(1, 2)
sns.distplot(train_text_df['title_len'], hist=True, kde=True, color='b', label='title len', ax=ax1)
sns.distplot(train_text_df[train_text_df['body_len'] < 256]['body_len'], hist=True, kde=True, color='r', label='body len', ax=ax2)
ax1.axvline(x=MAX_TITLE_LEN, color='k', linestyle='--', label='max len')
ax2.axvline(x=MAX_BODY_LEN, color='k', linestyle='--', label='max len')
ax1.set_title('title length histogram'); ax1.legend(loc=1); 
ax2.set_title('body length histogram'); ax2.legend(loc=1);
plt.savefig('../figures/question_len_hist.png')

#training parameters
num_epochs = 8 #16 
batch_size = 32 

#model parameters
embed_dim = embeddings.shape[1] #200
hidden_size = 240 #number of LSTM cells 
weight_decay = 1e-5 
learning_rate = 1e-3

#RNN architecture
class RNN(nn.Module):
    
    def __init__(self, embed_dim, hidden_size, vocab_size, batch_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        #TODO: ignore loss computations on 0 embedding index inputs 
        #TODO: average pooling
        #TODO: display gradient magnitude
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
        #[batch_size, seq_length (num_words), embed_dim]
        lstm_out, self.hidden = self.lstm(all_x.view(self.batch_size, x_idx.size(1), -1), self.hidden)
        #h_n dim: [1, batch_size, hidden_size]
        h_n, c_n = self.hidden[0], self.hidden[1]
        return h_n.squeeze(0) 
        
use_gpu = torch.cuda.is_available()

model = RNN(embed_dim, hidden_size, len(word_to_idx), batch_size)
if use_gpu:
    print "found CUDA GPU..."
    model = model.cuda()
    
print model

#define loss and optimizer
criterion = nn.MultiMarginLoss(p=1, margin=0.4, size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=4, gamma=0.5) #half learning rate every 4 epochs

learning_rate_schedule = [] 
training_loss, validation_loss, test_loss = [], [], []

print "training..."
for epoch in range(num_epochs):
    
    running_train_loss = 0.0
    
    train_data_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4, 
        drop_last = True)
        
    model.train()
    scheduler.step()
        
    for batch in tqdm(train_data_loader):
      
        query_title = Variable(batch['query_title'])
        query_body = Variable(batch['query_body'])
        similar_title = Variable(batch['similar_title'])
        similar_body = Variable(batch['similar_body'])

        random_title_list = []
        random_body_list = []
        for ridx in range(NUM_NEGATIVE): #100, number of random (negative) examples 
            random_title_name = 'random_title_' + str(ridx)
            random_body_name = 'random_body_' + str(ridx)
            random_title_list.append(Variable(batch[random_title_name]))
            random_body_list.append(Variable(batch[random_body_name]))

        if use_gpu:
            query_title, query_body = query_title.cuda(), query_body.cuda()
            similar_title, similar_body = similar_title.cuda(), similar_body.cuda()
            random_title_list = map(lambda item: item.cuda(), random_title_list)
            random_body_list = map(lambda item: item.cuda(), random_body_list)
        
        optimizer.zero_grad() 

        #query title
        model.hidden = model.init_hidden() 
        if use_gpu:
            model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
        lstm_query_title = model(query_title)

        #query body
        model.hidden = model.init_hidden() 
        if use_gpu:
            model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
        lstm_query_body = model(query_body)

        lstm_query = (lstm_query_title + lstm_query_body)/2.0

        #similar title
        model.hidden = model.init_hidden() 
        if use_gpu:
            model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
        lstm_similar_title = model(similar_title)

        #similar body
        model.hidden = model.init_hidden() 
        if use_gpu:
            model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
        lstm_similar_body = model(similar_body)

        lstm_similar = (lstm_similar_title + lstm_similar_body)/2.0

        lstm_random_list = []
        for ridx in range(len(random_title_list)):
            #random title
            model.hidden = model.init_hidden() 
            if use_gpu:
                model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
            lstm_random_title = model(random_title_list[ridx])

            #random body
            model.hidden = model.init_hidden() 
            if use_gpu:
                model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
            lstm_random_body = model(random_body_list[ridx])

            lstm_random = (lstm_random_title + lstm_random_body)/2.0
            lstm_random_list.append(lstm_random)
        #end for
           
        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        score_pos = cosine_similarity(lstm_query, lstm_similar)

        score_list = []
        score_list.append(score_pos)
        for ridx in range(len(lstm_random_list)):
            score_neg = cosine_similarity(lstm_query, lstm_random_list[ridx])
            score_list.append(score_neg)

        X_scores = torch.stack(score_list, 1) #[batch_size, K=101]
        y_targets = Variable(torch.zeros(X_scores.size(0)).type(torch.LongTensor)) #[batch_size]
        if use_gpu:
            y_targets = y_targets.cuda()
        loss = criterion(X_scores, y_targets) #y_target=0
        loss.backward()
        optimizer.step()
                
        running_train_loss += loss.cpu().data[0]        
        
    #end for
    training_loss.append(running_train_loss)
    learning_rate_schedule.append(scheduler.get_lr())
    print "epoch: %4d, training loss: %.4f" %(epoch+1, running_train_loss)
    
    torch.save(model, SAVE_PATH + SAVE_NAME)

    #early stopping
    patience = 4
    min_delta = 0.1
    if epoch == 0:
        patience_cnt = 0
    elif epoch > 0 and training_loss[epoch-1] - training_loss[epoch] > min_delta:
        patience_cnt = 0
    else:
        patience_cnt += 1

    if patience_cnt > patience:
        print "early stopping..."
        break
#end for
"""
print "loading pre-trained model..."
model = torch.load(SAVE_PATH + SAVE_NAME)
if use_gpu:
    print "found CUDA GPU..."
    model = model.cuda()
"""

print "scoring test questions..."
running_test_loss = 0.0

test_data_loader = torch.utils.data.DataLoader(
    test_data, 
    batch_size = batch_size,
    shuffle = False,
    num_workers = 4, 
    drop_last = True)
        
model.eval()

for batch in tqdm(test_data_loader):

    query_idx = batch['query_idx']
    query_title = Variable(batch['query_title'])
    query_body = Variable(batch['query_body'])
    similar_title = Variable(batch['similar_title'])
    similar_body = Variable(batch['similar_body'])

    random_title_list = []
    random_body_list = []
    for ridx in range(20): #number of retrieved (bm25) examples 
        random_title_name = 'random_title_' + str(ridx)
        random_body_name = 'random_body_' + str(ridx)
        random_title_list.append(Variable(batch[random_title_name]))
        random_body_list.append(Variable(batch[random_body_name]))

    if use_gpu:
        query_title, query_body = query_title.cuda(), query_body.cuda()
        similar_title, similar_body = similar_title.cuda(), similar_body.cuda()
        random_title_list = map(lambda item: item.cuda(), random_title_list)
        random_body_list = map(lambda item: item.cuda(), random_body_list)

    #query title
    model.hidden = model.init_hidden() 
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
    lstm_query_title = model(query_title)

    #query body
    model.hidden = model.init_hidden() 
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
    lstm_query_body = model(query_body)

    lstm_query = (lstm_query_title + lstm_query_body)/2.0

    #similar title
    model.hidden = model.init_hidden() 
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
    lstm_similar_title = model(similar_title)

    #similar body
    model.hidden = model.init_hidden() 
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
    lstm_similar_body = model(similar_body)

    lstm_similar = (lstm_similar_title + lstm_similar_body)/2.0

    lstm_random_list = []
    for ridx in range(len(random_title_list)):
        #random title
        model.hidden = model.init_hidden() 
        if use_gpu:
            model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
        lstm_random_title = model(random_title_list[ridx])

        #random body
        model.hidden = model.init_hidden() 
        if use_gpu:
            model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))
        lstm_random_body = model(random_body_list[ridx])

        lstm_random = (lstm_random_title + lstm_random_body)/2.0
        lstm_random_list.append(lstm_random)
    #end for
           
    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    score_pos = cosine_similarity(lstm_query, lstm_similar)

    score_list = []
    score_list.append(score_pos)
    for ridx in range(len(lstm_random_list)):
        score_neg = cosine_similarity(lstm_query, lstm_random_list[ridx])
        score_list.append(score_neg)

    X_scores = torch.stack(score_list, 1) #[batch_size, K=101]
    y_targets = Variable(torch.zeros(X_scores.size(0)).type(torch.LongTensor)) #[batch_size]
    if use_gpu:
        y_targets = y_targets.cuda()
    loss = criterion(X_scores, y_targets) #y_target=0
    running_test_loss += loss.cpu().data[0]        
    
    #save scores to data frame
    lstm_query_idx = query_idx.cpu().numpy()
    lstm_retrieved_scores = X_scores.cpu().data.numpy()[:,1:] #skip positive score
    for row, qidx in enumerate(lstm_query_idx):
        test_idx_df.loc[test_idx_df['query_id'] == qidx, 'lstm_score'] = " ".join(lstm_retrieved_scores[row,:].astype('str'))
#end for        
    
print "total test loss: ", running_test_loss
print "number of NaN: \n", test_idx_df.isnull().sum()
test_idx_df = test_idx_df.dropna() #NaNs are due to restriction: range(100)

#save scored data frame
test_idx_df.to_csv(SAVE_PATH + '/test_idx_df_scored_lstm.csv', header=True)

print "computing ranking metrics..."
lstm_mrr_test = compute_mrr(test_idx_df, score_name='lstm_score')
print "lstm MRR (test): ", np.mean(lstm_mrr_test)

lstm_pr1_test = precision_at_k(test_idx_df, K=1, score_name='lstm_score')
print "lstm P@1 (test): ", np.mean(lstm_pr1_test)

lstm_pr5_test = precision_at_k(test_idx_df, K=5, score_name='lstm_score')
print "lstm P@5 (test): ", np.mean(lstm_pr5_test)

lstm_map_test = compute_map(test_idx_df, score_name='lstm_score')
print "lstm map (test): ", np.mean(lstm_map_test)


#generate plots
plt.figure()
plt.plot(training_loss, label='Adam')
plt.title("LSTM Model Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.savefig('../figures/lstm_training_loss.png')

plt.figure()
plt.plot(learning_rate_schedule, label='learning rate')
plt.title("LSTM learning rate schedule")
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.legend()
plt.savefig('../figures/lstm_learning_rate_schedule.png')

"""
plt.figure()
plt.plot(validation_loss, label='Adam')
plt.title("LSTM Model Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.savefig('./figures/lstm_validation_loss.png')
"""

