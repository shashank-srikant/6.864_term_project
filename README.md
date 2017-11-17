# 6.864_term_project
Term project for 6.864 - Advanced NLP

# Installation
Once you've downloaded the repository, you will need to create a config.ini in the /src folder.
This file will contain all parameters that will be picked up at run-time. These include paths to your data folder etc.
Python 2.x provides the ConfigReader module which we use to parse this file.
This is the format of config.ini that you'll have to follow.

```
[paths]
# End each of your paths with a / (on linux-like systems)
data_path = <PATH TO THE DATA FOLDER ON YOUR LOCAL MACHINE>
save_path = <PATH WHERE YOU WANT TO SAVE THE INTERMEDIATE DATA FILES>
root_path = <CWD OF YOUR CODEBASE>
embeddings_path = <PATH TO EMBEDDINGS FILE ON YOUR LOCAL MACHINE>
extracted_data_file_name = data_raw.dat
train_test_file_name = data_train_test.dat

[data_params]
MAX_TITLE_LEN = 10
MAX_BODY_LEN = 100
TRAIN_SAMPLE_SIZE = -1
# -1 if you want to run it on the entire data set extracted in extracted_data_file_name. Use any other small number like 10/100 to prepare a train-set of 10/100 data points and test your workflow.

[rnn_params]
# add your RNN params here

[cnn_params]
# add your CNN params here
```

# Dependencies   
PyTorch 0.2.0_3   
