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

from ranking_metrics import compute_mrr, precision_at_k, compute_map
from model_network import model_neural_network

import sys

np.random.seed(0)
#torch.manual_seed(0)

config = ConfigParser.ConfigParser()
config.readfp(open(r'../src/config.ini'))
SAVE_PATH = config.get('paths', 'save_path')
DATA_FILE_NAME = config.get('paths', 'extracted_data_file_name')
TRAIN_TEST_FILE_NAME = config.get('paths', 'train_test_file_name')
SAVE_NAME = config.get('cnn_params', 'save_name')
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
num_epochs = 2 #16
batch_size = 32 

#model parameters
embed_num = len(word_to_idx)
embed_dim = len(embeddings[0])
kernel_num = 100  #TODO: tune
kernel_sizes = range(2,6)
learning_rate = 1e-3 
weight_decay = 1e-5

class  CNN(nn.Module):
    def __init__(self, embed_num, embed_dim, kernel_num, kernel_sizes):
        super(CNN,self).__init__()
        V = embed_num
        D = embed_dim
        Ci = 1            #input channel
        Co = kernel_num   #depth
        Ks = kernel_sizes #height of each filter

        self.embed = nn.Embedding(V, D)
        self.embed.requires_grad = False
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
criterion = nn.MultiMarginLoss(p=1, margin=0.4, size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=4, gamma=0.5) #half learning rate every 4 epochs

learning_rate_schedule = [] 
training_loss, validation_loss, test_loss = [], [], []

print "train data size:" + str(len(train_data))
print "val data size:" + str(len(val_data))
print "test data size:" + str(len(test_data))

print "training..."

training_loss, train_idx_df, learning_rate_schedule = model_neural_network(True, num_epochs, train_data, train_idx_df, batch_size, 40,  model, criterion, optimizer, scheduler, 'CNN_train', use_gpu, (SAVE_PATH + SAVE_NAME))
running_test_loss, test_idx_df, _ = model_neural_network(False, num_epochs, test_data, test_idx_df, batch_size, 25,  model, criterion, optimizer, scheduler, 'CNN_test', use_gpu, (SAVE_PATH + SAVE_NAME + "test"))

    
print "total test loss: ", running_test_loss
print "number of NaN: \n", test_idx_df.isnull().sum()
test_idx_df = test_idx_df.dropna() #NaNs are due to restriction: range(100)

#save scored data frame
test_idx_df.to_csv(SAVE_PATH + '/test_idx_df_scored_cnn.csv', header=True)

print "computing ranking metrics..."
cnn_mrr_test = compute_mrr(test_idx_df, score_name='cnn_score')
print "cnn MRR (test): ", np.mean(cnn_mrr_test)

cnn_pr1_test = precision_at_k(test_idx_df, K=1, score_name='cnn_score')
print "cnn P@1 (test): ", np.mean(cnn_pr1_test)

cnn_pr5_test = precision_at_k(test_idx_df, K=5, score_name='cnn_score')
print "cnn P@5 (test): ", np.mean(cnn_pr5_test)

cnn_map_test = compute_map(test_idx_df, score_name='cnn_score')
print "cnn map (test): ", np.mean(cnn_map_test)


#generate plots
plt.figure()
plt.plot(training_loss, label='Adam')
plt.title("CNN Model Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.savefig('../figures/cnn_training_loss1.png')

plt.figure()
plt.plot(learning_rate_schedule, label='learning rate')
plt.title("CNN learning rate schedule")
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.legend()
plt.savefig('../figures/cnn_learning_rate_schedule1.png')

"""
plt.figure()
plt.plot(validation_loss, label='Adam')
plt.title("CNN Model Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.savefig('../figures/cnn_validation_loss.png')
"""

        
