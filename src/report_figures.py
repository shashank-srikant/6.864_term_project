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

filename = SAVE_PATH + 'figures_da_cnn_adversarial.dat' 
with open(filename) as f: 
    figures_da_cnn_adversarial = pickle.load(f)

filename = SAVE_PATH + 'figures_da_lstm_adversarial.dat' 
with open(filename) as f: 
    figures_da_lstm_adversarial = pickle.load(f)

filename = SAVE_PATH + 'figures_da_doc2vec_dm.dat' 
with open(filename) as f: 
    figures_da_doc2vec_dm = pickle.load(f)

filename = SAVE_PATH + 'figures_da_doc2vec_dbow.dat' 
with open(filename) as f: 
    figures_da_doc2vec_dbow = pickle.load(f)

filename = SAVE_PATH + 'figures_da_siamese.dat' 
with open(filename) as f: 
    figures_da_siamese = pickle.load(f)


#generate plots
plt.figure()
plt.plot(figures_lstm['lstm_training_loss'], c='r', lw=2.0, label='LSTM')
plt.plot(figures_cnn['cnn_training_loss'], c='b', lw=2.0, label='CNN')
plt.title("Multi-Margin Training Loss (Adam)")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.savefig('../figures/report_training_loss_part1.png')

plt.figure()
plt.plot(figures_cnn['cnn_learning_rate'], lw=2.0, label='learning rate')
plt.title("learning rate schedule")
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.legend()
plt.savefig('../figures/report_learning_rate_schedule.png')

xlabels = ['value', 'metric', 'model']
ranking_metrics_val_df = pd.DataFrame(columns = xlabels)
ranking_metrics_val_df.loc[0, :] = [figures_lstm['lstm_map_val'][0], 'MAP', 'LSTM (dev)']
ranking_metrics_val_df.loc[1, :] = [figures_lstm['lstm_mrr_val'][0], 'MRR', 'LSTM (dev)']
ranking_metrics_val_df.loc[2, :] = [figures_lstm['lstm_pr1_val'][0], 'P@1', 'LSTM (dev)']
ranking_metrics_val_df.loc[3, :] = [figures_lstm['lstm_pr5_val'][0], 'P@5', 'LSTM (dev)']
ranking_metrics_val_df.loc[4, :] = [figures_cnn['cnn_map_val'][0], 'MAP', 'CNN (dev)']
ranking_metrics_val_df.loc[5, :] = [figures_cnn['cnn_mrr_val'][0], 'MRR', 'CNN (dev)']
ranking_metrics_val_df.loc[6, :] = [figures_cnn['cnn_pr1_val'][0], 'P@1', 'CNN (dev)']
ranking_metrics_val_df.loc[7, :] = [figures_cnn['cnn_pr5_val'][0], 'P@5', 'CNN (dev)']

ranking_metrics_test_df = pd.DataFrame(columns = xlabels)
ranking_metrics_test_df.loc[0, :] = [figures_lstm['lstm_map_test'][0], 'MAP', 'LSTM (test)']
ranking_metrics_test_df.loc[1, :] = [figures_lstm['lstm_mrr_test'][0], 'MRR', 'LSTM (test)']
ranking_metrics_test_df.loc[2, :] = [figures_lstm['lstm_pr1_test'][0], 'P@1', 'LSTM (test)']
ranking_metrics_test_df.loc[3, :] = [figures_lstm['lstm_pr5_test'][0], 'P@5', 'LSTM (test)']
ranking_metrics_test_df.loc[4, :] = [figures_cnn['cnn_map_test'][0], 'MAP', 'CNN (test)']
ranking_metrics_test_df.loc[5, :] = [figures_cnn['cnn_mrr_test'][0], 'MRR', 'CNN (test)']
ranking_metrics_test_df.loc[6, :] = [figures_cnn['cnn_pr1_test'][0], 'P@1', 'CNN (test)']
ranking_metrics_test_df.loc[7, :] = [figures_cnn['cnn_pr5_test'][0], 'P@5', 'CNN (test)']

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), sharex=True)
sns.barplot(x='metric', y='value', hue='model', data=ranking_metrics_val_df, palette='Reds', ax=ax1)
sns.barplot(x='metric', y='value', hue='model', data=ranking_metrics_test_df, palette='Reds', ax=ax2)
ax1.set_title('ranking performance')
ax1.set_ylabel('dev')
ax2.set_ylabel('test')
plt.savefig('../figures/report_ranking_performance.png')













