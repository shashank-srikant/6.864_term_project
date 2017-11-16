import torch
import cPickle as pickle
from tqdm import tqdm
import ConfigParser
from time import time
from nltk.tokenize import RegexpTokenizer
###################################################################
config = ConfigParser.ConfigParser()
config.readfp(open(r'config.ini'))

SAVE_PATH = config.get('paths', 'save_path')
DATA_FILE_NAME = config.get('paths', 'extracted_data_file_name')
filename = SAVE_PATH + DATA_FILE_NAME

tic1 = time()
with open(filename) as f:  # Python 3: open(..., 'rb')
    train_text_df, train_idx_df, dev_idx_df, embeddings, word_to_idx = pickle.load(f)
toc1 = time()
print "elapsed time to retrieve extracted data from file: %.2f sec" %(toc1 - tic1)
###################################################################

TRAIN_TEST_FILE_NAME = config.get('paths', 'train_test_file_name')

MAX_TITLE_LEN = int(config.get('data_params', 'MAX_TITLE_LEN'))
MAX_BODY_LEN = int(config.get('data_params', 'MAX_BODY_LEN')) #max number of words per sentence
TRAIN_SAMPLE_SIZE = int(config.get('data_params', 'TRAIN_SAMPLE_SIZE')) # can provide any number less than train_idx_df.shape[0] to test your code. Provide -1 if you want to use all of train_idx_df.shape[0]

if TRAIN_SAMPLE_SIZE == -1:
    TRAIN_SAMPLE_SIZE = train_idx_df.shape[0]

###################################################################
def get_tensor_idx(text, word_to_idx, max_len):
    null_idx = 0  #idx if word is not in the embeddings dictionary
    text_idx = [word_to_idx[x] if x in word_to_idx else null_idx for x in text][:max_len]
    if len(text_idx) < max_len:
        text_idx.extend([null_idx for _ in range(max_len - len(text_idx))])    
    x = torch.LongTensor(text_idx)  #64-bit integer
    return x
###################################################################
tic = time()
train_data = []
tokenizer = RegexpTokenizer(r'\w+')

for idx in tqdm(range(TRAIN_SAMPLE_SIZE)):
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
            if len(random_title) > 0:
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
###################################################################
filename = SAVE_PATH + TRAIN_TEST_FILE_NAME
with open(filename, 'w') as f:
    pickle.dump(train_data, f)