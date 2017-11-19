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

import ConfigParser
from tqdm import tqdm
from time import time
import cPickle as pickle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(0)
#torch.manual_seed(0)

config = ConfigParser.ConfigParser()
config.readfp(open(r'config.ini'))

SAVE_PATH = config.get('paths', 'save_path')
DATA_FILE_NAME = config.get('paths', 'extracted_data_file_name')
TRAIN_TEST_FILE_NAME = config.get('paths', 'train_test_file_name')
SAVE_NAME = config.get('rnn_params', 'save_name')

MAX_TITLE_LEN = int(config.get('data_params', 'MAX_TITLE_LEN'))
MAX_BODY_LEN = int(config.get('data_params', 'MAX_BODY_LEN'))

data_filename = SAVE_PATH + DATA_FILE_NAME
train_test_filename = SAVE_PATH + TRAIN_TEST_FILE_NAME

print "loading pickled data..."
tic = time()
with open(data_filename) as f:  
    train_text_df, train_idx_df, dev_idx_df, test_idx_df, embeddings, word_to_idx = pickle.load(f)
with open(train_test_filename) as f:
    train_data, val_data, test_data = pickle.load(f)
toc = time()
print "elapsed time: %.2f sec" %(toc - tic)

tokenizer = RegexpTokenizer(r'\w+')
stop = set(stopwords.words('english'))

def compute_mrr(data_frame, score_name='bm25_score'):

    mrr_output = []
    for qidx in range(data_frame.shape[0]):
        retrieved_set = map(int, data_frame.loc[qidx, 'random_id'].split(' '))
        relevant_set = set(map(int, data_frame.loc[qidx, 'similar_id'].split(' ')))
        retrieved_scores = map(float, data_frame.loc[qidx, score_name].split(' '))

        #sort according to scores (higher score is better, i.e. ranked higher)        
        retrieved_set_sorted = [p for p, s in sorted(zip(retrieved_set, retrieved_scores),
                                key = lambda pair: pair[1], reverse=True)]

        rank = 1
        for item in retrieved_set_sorted:
            if item in relevant_set:
                break
            else:
                rank += 1
        #end for
        MRR = 1.0 / rank
        mrr_output.append(MRR)
    #end for
    return mrr_output

def precision_at_k(data_frame, K=5, score_name='bm25_score'):

    pr_output = []
    for qidx in range(data_frame.shape[0]):
        retrieved_set = map(int, data_frame.loc[qidx, 'random_id'].split(' '))
        relevant_set = set(map(int, data_frame.loc[qidx, 'similar_id'].split(' '))) 
        retrieved_scores = map(float, data_frame.loc[qidx, score_name].split(' '))

        #sort according to scores (higher score is better, i.e. ranked higher)        
        retrieved_set_sorted = [p for p, s in sorted(zip(retrieved_set, retrieved_scores),
                                key = lambda pair: pair[1], reverse=True)]

        count = 0
        for item in retrieved_set_sorted[:K]:
            if item in relevant_set:
                count += 1
        #end for
        precision_at_k = count / float(K)
        pr_output.append(precision_at_k)
    #end for
    return pr_output

def compute_map(data_frame, score_name='bm25_score'):

    map_output = []
    for qidx in range(data_frame.shape[0]):
        retrieved_set = map(int, data_frame.loc[qidx, 'random_id'].split(' '))
        relevant_set = set(map(int, data_frame.loc[qidx, 'similar_id'].split(' '))) 
        retrieved_scores = map(float, data_frame.loc[qidx, score_name].split(' '))

        #sort according to scores (higher score is better, i.e. ranked higher)        
        retrieved_set_sorted = [p for p, s in sorted(zip(retrieved_set, retrieved_scores),
                                key = lambda pair: pair[1], reverse=True)]

        AP = 0
        num_relevant = 0
        for ridx, item in enumerate(retrieved_set_sorted):
            if item in relevant_set:
                num_relevant += 1
                #compute precision at K=ridx+1
                count = 0
                for entry in retrieved_set_sorted[:ridx+1]:
                    if entry in relevant_set:
                        count += 1
                #end for
                AP += count / float(ridx+1)
            #end if
        #end for
        if (num_relevant > 0):
            AP = AP / float(num_relevant)
        else:
            AP = 0
        #end for
        map_output.append(AP)
    #end for
    return map_output


#visualize data
f, (ax1, ax2) = plt.subplots(1, 2)
sns.distplot(train_text_df['title_len'], hist=True, kde=True, color='b', label='title len', ax=ax1)
sns.distplot(train_text_df[train_text_df['body_len'] < 256]['body_len'], hist=True, kde=True, color='r', label='body len', ax=ax2)
ax1.axvline(x=MAX_TITLE_LEN, color='k', linestyle='--', label='max len')
ax2.axvline(x=MAX_BODY_LEN, color='k', linestyle='--', label='max len')
ax1.set_title('title length histogram'); ax1.legend(loc=1); 
ax2.set_title('body length histogram'); ax2.legend(loc=1);
plt.savefig('../figures/question_len_hist.png')

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

#training parameters
num_epochs = 32 
batch_size = 16 

#model parameters
embed_dim = embeddings.shape[1] #200
hidden_size = 32 # number of LSTM cells 
weight_decay = 1e-3 
learning_rate = 1e-3 

#RNN architecture
class RNN(nn.Module):
    
    def __init__(self, embed_dim, hidden_size, vocab_size, batch_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding_layer = nn.Embedding(vocab_size, embed_dim) #TODO: make non-trainable
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
        
use_gpu = torch.cuda.is_available()

model = RNN(embed_dim, hidden_size, len(word_to_idx), batch_size)
if use_gpu:
    print "found CUDA GPU..."
    model = model.cuda()
    
print model

#define loss and optimizer
criterion = nn.MultiMarginLoss(p=1, margin=2, size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

training_loss, validation_loss, test_loss = [], [], []
"""
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
        
    for batch in tqdm(train_data_loader):
      
        query_title = Variable(batch['query_title'])
        query_body = Variable(batch['query_body'])
        similar_title = Variable(batch['similar_title'])
        similar_body = Variable(batch['similar_body'])

        random_title_list = []
        random_body_list = []
        for ridx in range(10):  #range(100)
            random_title_name = 'random_title_' + str(ridx)
            random_body_name = 'random_body_' + str(ridx)
            random_title_list.append(Variable(batch[random_title_name]))
            random_body_list.append(Variable(batch[random_body_name]))

        if use_gpu:
            query_title, query_body = query_title.cuda(), query_body.cuda()
            similar_title, similar_body = similar_title.cuda(), similar_body.cuda()
            random_title_list = map(lambda item: item.cuda(), random_title_list)
            random_body_list = map(lambda item: item.cuda(), random_body_list)
        
        optimizer.zero_grad() #TODO: check how often to reset this
        model.hidden = model.init_hidden() #TODO: check how often to reset this
        if use_gpu:
            model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))

        lstm_query_title = model(query_title)
        lstm_query_body = model(query_body)
        lstm_query = (lstm_query_title + lstm_query_body)/2.0

        lstm_similar_title = model(similar_title)
        lstm_similar_body = model(similar_body)
        lstm_similar = (lstm_similar_title + lstm_similar_body)/2.0

        lstm_random_list = []
        for ridx in range(len(random_title_list)):
            lstm_random_title = model(random_title_list[ridx])
            lstm_random_body = model(random_body_list[ridx])
            lstm_random = (lstm_random_title + lstm_random_body)/2.0
            lstm_random_list.append(lstm_random)
           
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
    print "epoch: %4d, training loss: %.4f" %(epoch+1, running_train_loss)
    
    torch.save(model, SAVE_PATH + SAVE_NAME)
#end for
"""
print "loading pre-trained model..."
model = torch.load(SAVE_PATH + SAVE_NAME)
if use_gpu:
    print "found CUDA GPU..."
    model = model.cuda()

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
    for ridx in range(10):  #range(20)
        random_title_name = 'random_title_' + str(ridx)
        random_body_name = 'random_body_' + str(ridx)
        random_title_list.append(Variable(batch[random_title_name]))
        random_body_list.append(Variable(batch[random_body_name]))

    if use_gpu:
        query_title, query_body = query_title.cuda(), query_body.cuda()
        similar_title, similar_body = similar_title.cuda(), similar_body.cuda()
        random_title_list = map(lambda item: item.cuda(), random_title_list)
        random_body_list = map(lambda item: item.cuda(), random_body_list)
        
    model.hidden = model.init_hidden() #TODO: check how often to reset this!!
    if use_gpu:
        model.hidden = tuple(map(lambda item: item.cuda(), model.hidden))

    lstm_query_title = model(query_title)
    lstm_query_body = model(query_body)
    lstm_query = (lstm_query_title + lstm_query_body)/2.0

    lstm_similar_title = model(similar_title)
    lstm_similar_body = model(similar_body)
    lstm_similar = (lstm_similar_title + lstm_similar_body)/2.0

    lstm_random_list = []
    for ridx in range(len(random_title_list)):
        lstm_random_title = model(random_title_list[ridx])
        lstm_random_body = model(random_body_list[ridx])
        lstm_random = (lstm_random_title + lstm_random_body)/2.0
        lstm_random_list.append(lstm_random)
           
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
    
    #save scores to data-frame
    lstm_query_idx = query_idx.cpu().numpy()
    lstm_retrieved_scores = X_scores.cpu().data.numpy()[:,1:] #skip positive score
    for row, qidx in enumerate(lstm_query_idx):
        test_idx_df.loc[test_idx_df['query_id'] == qidx, 'lstm_score'] = " ".join(lstm_retrieved_scores[row,:].astype('str'))
#end for        
    
print "total test loss: ", running_test_loss
print "number of NaN: ", test_idx_df.isnull().sum()
test_idx_df = test_idx_df.dropna() #NaNs are due to restriction: range(100)

print "computing ranking metrics..."
lstm_mrr_test = compute_mrr(test_idx_df, score_name='lstm_score')
print "lstm MRR (test): ", np.mean(lstm_mrr_test)

lstm_pr1_test = precision_at_k(test_idx_df, K=1, score_name='lstm_score')
print "lstm P@1 (test): ", np.mean(lstm_pr1_test)

lstm_pr5_test = precision_at_k(test_idx_df, K=5, score_name='lstm_score')
print "lstm P@5 (test): ", np.mean(lstm_pr5_test)

lstm_map_test = compute_map(test_idx_df, score_name='lstm_score')
print "lstm map (test): ", np.mean(lstm_map_test)



"""
#generate plots
plt.figure()
plt.plot(training_loss, label='Adam')
plt.title("LSTM Model Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.savefig('./figures/training_loss.png')

plt.figure()
plt.plot(validation_loss, label='Adam')
plt.title("LSTM Model Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.savefig('./figures/validation_loss.png')
"""

        

