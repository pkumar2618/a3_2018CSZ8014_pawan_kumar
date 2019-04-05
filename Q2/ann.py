import numpy as np
import pandas as pd
import sys

# print sys.argv[0] # prints python_script.py

path_config= sys.argv[1] # prints var1
# path_train_hot = sys.argv[2] # prints var1
# path_test_hot = sys.argv[3] # prints var2

path_train_hot = "../../../ass3_data/poker-hand-training-true-onehot.data"
path_test_hot= "../../../ass3_data/poker-hand-testing-onehot.data"

train_XY = pd.read_csv(path_train_hot)
train_XY= train_XY.loc[0:100,:]
train_X = train_XY.loc[:,:'C5_13']
train_Y = train_XY['Hand']

# test_XY = pd.read_csv(path_test_hot)
# test_XY= train_XY.loc[0:100,:]
# test_X = train_XY.loc[:,:'C5_13']
# test_Y = train_XY['Hand']
# print(train_Y)

### Reading Configuration file ###
config ={'neu_input_layer': 1, 'neu_output_layer': 1, 'batch_size': 10, 'n_hidden_layers': 1, 'neu_in_layer':[2,3],
         'non_linearity': 'sigmoid', 'learning_rate': 0.04 }

