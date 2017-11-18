import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = '/data/vision/fisher/data1/vsmolyakov/nlp_project/data/askubuntu/'

def compute_mrr(data_frame, score_name='bm25_score'):

    mrr_output = []
    for qidx in range(data_frame.shape[0]):
        retrieved_set = map(int, data_frame.loc[qidx, 'random_id'].split(' '))
        relevant_set = set(map(int, data_frame.loc[qidx, 'similar_id'].split(' '))) #TODO: check
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


#load data
print "loading data..."

dev_idx_file = DATA_PATH + '/dev.txt'
dev_idx_df = pd.read_table(dev_idx_file, sep='\t', header=None)
#dev_idx_df.columns = ['query_id', 'similar_id', 'retrieved_id', 'bm25_score']
dev_idx_df.columns = ['query_id', 'similar_id', 'random_id', 'bm25_score']
dev_idx_df = dev_idx_df.dropna()
dev_idx_df = dev_idx_df.reset_index()

test_idx_file = DATA_PATH + '/test.txt'
test_idx_df = pd.read_table(test_idx_file, sep='\t', header=None)
#test_idx_df.columns = ['query_id', 'similar_id', 'retrieved_id', 'bm25_score']
test_idx_df.columns = ['query_id', 'similar_id', 'random_id', 'bm25_score']
test_idx_df = test_idx_df.dropna()
test_idx_df = test_idx_df.reset_index()

print "computing ranking metrics..."

bm25_mrr_dev = compute_mrr(dev_idx_df, score_name='bm25_score')
bm25_mrr_test = compute_mrr(test_idx_df, score_name='bm25_score')
print "bm25 MRR (dev): ", np.mean(bm25_mrr_dev)
print "bm25 MRR (test): ", np.mean(bm25_mrr_test)






