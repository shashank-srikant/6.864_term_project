import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import ConfigParser
import cPickle as pickle

config = ConfigParser.ConfigParser()
config.readfp(open(r'config.ini'))
SAVE_PATH = config.get('paths', 'save_path')

#load all data
filename = SAVE_PATH + 'figures_cnn.dat' 
with open(filename) as f: 
    figures_cnn = pickle.load(f)

filename = SAVE_PATH + 'figures_lstm.dat' 
with open(filename) as f: 
    figures_lstm = pickle.load(f)

filename = SAVE_PATH + 'figures_da_tfidf.dat' 
with open(filename) as f: 
    figures_da_tfidf = pickle.load(f)

filename = SAVE_PATH + 'figures_da_cnn_direct.dat' 
with open(filename) as f: 
    figures_da_cnn_direct = pickle.load(f)

filename = SAVE_PATH + 'figures_da_lstm_direct.dat' 
with open(filename) as f: 
    figures_da_lstm_direct = pickle.load(f)




filename = SAVE_PATH + 'figures_da_doc2vec.dat' 
with open(filename) as f: 
    figures_da_doc2vec = pickle.load(f)

filename = SAVE_PATH + 'figures_da_siamese.dat' 
with open(filename) as f: 
    figures_da_siamese = pickle.load(f)


#generate plots
