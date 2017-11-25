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
from ranking_metrics import compute_mrr, precision_at_k, compute_map

np.random.seed(0)
#torch.manual_seed(0)

config = ConfigParser.ConfigParser()
config.readfp(open(r'../src/config.ini'))
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

#training parameters
num_epochs = 16
batch_size = 32 

#model parameters
embed_num = len(word_to_idx)
embed_dim = len(embeddings[0])
kernel_num = 100  #TODO: tune
kernel_sizes = range(2,6)
learning_rate = 1e-3
weight_decay = 1e-3

class  CNN(nn.Module):
    def __init__(self, embed_num, embed_dim, kernel_num, kernel_sizes):
        super(CNN,self).__init__()
        V = embed_num
        D = embed_dim
        Ci = 1            #input channel
        Co = kernel_num   #depth
        Ks = kernel_sizes #height of each filter

        self.embed = nn.Embedding(V, D)
        self.embed.weight.data = torch.from_numpy(embeddings)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

    def forward(self, x):
        x = self.embed(x) # (N,W,D)
        x = x.unsqueeze(1) # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return x

model = CNN(embed_num, embed_dim, kernel_num, kernel_sizes)

use_gpu = torch.cuda.is_available()
if use_gpu:
    print "found CUDA GPU..."
    model = model.cuda()

print model

#define loss and optimizer
criterion = nn.MultiMarginLoss(p=1, margin=2, size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
        
    for batch in tqdm(train_data_loader):
     
        query_title = Variable(batch['query_title'])
        query_body = Variable(batch['query_body'])
        similar_title = Variable(batch['similar_title'])
        similar_body = Variable(batch['similar_body'])

        random_title_list = []
        random_body_list = []
        for ridx in range(NUM_NEGATIVE): #number of random negative examples
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

        cnn_query_title = model(query_title)
        cnn_query_body = model(query_body)
        cnn_query = (cnn_query_title + cnn_query_body)/2.0

        cnn_similar_title = model(similar_title)
        cnn_similar_body = model(similar_body)
        cnn_similar = (cnn_similar_title + cnn_similar_body)/2.0

        cnn_random_list = []
        for ridx in range(len(random_title_list)):
            cnn_random_title = model(random_title_list[ridx])
            cnn_random_body = model(random_body_list[ridx])
            cnn_random = (cnn_random_title + cnn_random_body)/2.0
            cnn_random_list.append(cnn_random)
        #end for
           
        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        score_pos = cosine_similarity(cnn_query, cnn_similar)

        score_list = []
        score_list.append(score_pos)
        for ridx in range(len(cnn_random_list)):
            score_neg = cosine_similarity(cnn_query, cnn_random_list[ridx])
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
model = torch.load(SAVE_PATH)
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
    
    cnn_query_title = model(query_title)
    cnn_query_body = model(query_body)
    cnn_query = (cnn_query_title + cnn_query_body)/2.0

    cnn_similar_title = model(similar_title)
    cnn_similar_body = model(similar_body)
    cnn_similar = (cnn_similar_title + cnn_similar_body)/2.0

    cnn_random_list = []
    for ridx in range(len(random_title_list)):
        cnn_random_title = model(random_title_list[ridx])
        cnn_random_body = model(random_body_list[ridx])
        cnn_random = (cnn_random_title + cnn_random_body)/2.0
        cnn_random_list.append(cnn_random)
    #end for
           
    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    score_pos = cosine_similarity(cnn_query, cnn_similar)

    score_list = []
    score_list.append(score_pos)
    for ridx in range(len(cnn_random_list)):
        score_neg = cosine_similarity(cnn_query, cnn_random_list[ridx])
        score_list.append(score_neg)

    X_scores = torch.stack(score_list, 1) #[batch_size, K=101]
    y_targets = Variable(torch.zeros(X_scores.size(0)).type(torch.LongTensor)) #[batch_size]
    if use_gpu:
        y_targets = y_targets.cuda()
    loss = criterion(X_scores, y_targets) #y_target=0
    running_test_loss += loss.cpu().data[0]        
    
    #save scores to data-frame
    cnn_query_idx = query_idx.cpu().numpy()
    cnn_retrieved_scores = X_scores.cpu().data.numpy()[:,1:] #skip positive score
    for row, qidx in enumerate(cnn_query_idx):
        test_idx_df.loc[test_idx_df['query_id'] == qidx, 'cnn_score'] = " ".join(cnn_retrieved_scores[row,:].astype('str'))
#end for        
    
print "total test loss: ", running_test_loss
print "number of NaN: \n", test_idx_df.isnull().sum()
test_idx_df = test_idx_df.dropna() #NaNs are due to restriction: range(100)

print "computing ranking metrics..."
cnn_mrr_test = compute_mrr(test_idx_df, score_name='cnn_score')
print "cnn MRR (test): ", np.mean(cnn_mrr_test)

cnn_pr1_test = precision_at_k(test_idx_df, K=1, score_name='cnn_score')
print "cnn P@1 (test): ", np.mean(cnn_pr1_test)

cnn_pr5_test = precision_at_k(test_idx_df, K=5, score_name='cnn_score')
print "cnn P@5 (test): ", np.mean(cnn_pr5_test)

cnn_map_test = compute_map(test_idx_df, score_name='cnn_score')
print "cnn map (test): ", np.mean(cnn_map_test)


"""
#generate plots
plt.figure()
plt.plot(training_loss, label='Adam')
plt.title("CNN Model Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.savefig('../figures/cnn_training_loss.png')

plt.figure()
plt.plot(validation_loss, label='Adam')
plt.title("CNN Model Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.savefig('../figures/cnn_validation_loss.png')
"""

        
