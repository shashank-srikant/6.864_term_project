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




