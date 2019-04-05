import numpy as np
import pandas as pd
import sys
from model_dnn import L_layer_model

# print sys.argv[0] # prints python_script.py

# path_config= sys.argv[1] # prints var1
# path_train_hot = sys.argv[2] # prints var1
# path_test_hot = sys.argv[3] # prints var2

path_train_hot = "../../../ass3_data/poker-hand-training-true-onehot.data"
path_test_hot= "../../../ass3_data/poker-hand-testing-onehot.data"

train_XY = pd.read_csv(path_train_hot)
train_XY= train_XY.loc[:,:]
# train_X = train_XY.loc[0,:'C5_13'].to_numpy(copy=True).reshape(1,-1)
# train_Y = train_XY.loc[0,'Hand_0':'Hand_9'].to_numpy(copy=True).reshape(1,-1)

### gradient descent ##
train_X = train_XY.loc[:,:'C5_13'].to_numpy(copy=True).reshape(1,-1)
train_Y = train_XY.loc[:,'Hand_0':'Hand_9'].to_numpy(copy=True).reshape(1,-1)

### batch gradient descent ##

# test_XY = pd.read_csv(path_test_hot)
# test_XY= train_XY.loc[0:100,:]
# test_X = train_XY.loc[:,:'C5_13']
# test_Y = train_XY['Hand']
# print(train_Y)

### Reading Configuration file ###
config ={'neu_input_layer': 1, 'neu_output_layer': 1, 'batch_size': 10, 'n_hidden_layers': 2, 'neu_in_layers':[85,5,10],
         'non_linearity': 'sigmoid', 'learning_rate': 0.04 }



parameters = L_layer_model(train_X.T, train_Y.T, config['neu_in_layers'], learning_rate = config['learning_rate'], num_iterations = 1000, print_cost = True)

