import numpy as np
import pandas as pd
import nltk as nk
from nltk.corpus import stopwords
import cPickle as pickle
from time import time
from nltk.tokenize import RegexpTokenizer
import ConfigParser
#################################################
np.random.seed(0)
#torch.manual_seed(0)

config = ConfigParser.ConfigParser()
config.readfp(open(r'config.ini'))
ROOT_PATH = config.get('paths', 'root_path')
DATA_PATH = config.get('paths', 'data_path')
SAVE_PATH = config.get('paths', 'save_path')
EMBEDDINGS_FILE = config.get('paths', 'embeddings_path')
DATA_FILE_NAME = config.get('paths', 'extracted_data_file_name')
tokenizer = RegexpTokenizer(r'\w+')

###############################################
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

###################################################################
#load data
print "loading data..."
stop = set(stopwords.words('english')) #faster membership checking with a set
tic = time()
train_text_file = DATA_PATH + 'texts_raw_fixed.txt'
train_text_df = pd.read_table(train_text_file, sep='\t', header=None)
train_text_df.columns = ['id', 'title', 'body']
train_text_df = train_text_df.dropna()
train_text_df['title_len'] = train_text_df['title'].apply(lambda words: len(tokenizer.tokenize(str(words))))
train_text_df['body_len'] = train_text_df['body'].apply(lambda words: len(tokenizer.tokenize(str(words))))
train_text_df['title'] = train_text_df['title'].apply(lambda words:' '.join(filter(lambda x: x not in stop,  words.split())))
train_text_df['body'] = train_text_df['body'].apply(lambda words: ' '.join(filter(lambda x: x not in stop,  words.split())))
train_text_df['title_no_stpwrds_len'] = train_text_df['title'].apply(lambda words: len(tokenizer.tokenize(str(words))))
train_text_df['body_no_stpwrds_len'] = train_text_df['body'].apply(lambda words: len(tokenizer.tokenize(str(words))))

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
###################################################################

filename = SAVE_PATH + DATA_FILE_NAME
with open(filename, 'w') as f:
    pickle.dump([train_text_df, train_idx_df, dev_idx_df, embeddings, word_to_idx], f)
