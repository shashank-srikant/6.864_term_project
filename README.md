# 6.864_term_project
Term project for 6.864 - Advanced NLP

# Dependencies   
PyTorch   

# Installation
Once you've downloaded the repository, you will need to create a config.ini in the /src folder.
This file will contain all parameters that will be picked up at run-time. These include paths to your data folder etc.
Python 2.x provides the ConfigReader module which we use to parse this file.
This is the format of config.ini that you'll have to follow.

```
[paths]
data_path = <PATH TO THE DATA FOLDER ON YOUR LOCAL MACHINE>
save_path = <PATH WHERE YOU WANT TO SAVE THE INTERMEDIATE DATA FILES>
root_path = <CWD OF YOUR CODEBASE>
embeddings_path = <PATH TO EMBEDDINGS FILE ON YOUR LOCAL MACHINE>

[data_params]
MAX_TITLE_LEN = 20
MAX_BODY_LEN = 100

[rnn_params]
# add your RNN params here

[cnn_params]
# add your CNN params here
```
