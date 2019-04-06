import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import sys
from model_dnn import L_layer_model
from ann_utils import predict
import time

# print sys.argv[0] # prints python_script.py

# path_config= sys.argv[1] # prints var1
# path_train_hot = sys.argv[2] # prints var1
# path_test_hot = sys.argv[3] # prints var2

path_config = '/home/dell/Documents/2nd_sem/ml_assig/a3_2018CSZ8014_pawan_kumar/Q2/config_c.txt'
path_train_hot = '/home/dell/Documents/2nd_sem/ml_assig/ass3_data/poker-hand-training-true-onehot.data'
path_test_hot = '/home/dell/Documents/2nd_sem/ml_assig/ass3_data/poker-hand-testing-onehot.data'
question_part = 'c'

train_XY = pd.read_csv(path_train_hot)
train_XY = train_XY.sample(frac=1).reset_index(drop=True)
### use for a single example
# train_X = train_XY.loc[0,:'C5_13'].to_numpy(copy=True).reshape(1,-1)
# train_Y = train_XY.loc[0,'Hand_0':'Hand_9'].to_numpy(copy=True).reshape(1,-1)

### gradient descent for entire batch ##
train_X = train_XY.loc[:,:'C5_13'].to_numpy(copy=True)
train_Y = train_XY.loc[:,'Hand_0':'Hand_9'].to_numpy(copy=True)

test_XY = pd.read_csv(path_test_hot)
test_XY = test_XY.sample(frac=1).reset_index(drop=True)
test_X = test_XY.loc[:,:'C5_13'].to_numpy(copy=True)
test_Y = test_XY.loc[:,'Hand_0':'Hand_9'].to_numpy(copy=True)

### Reading Configuration file ###
config ={'neu_input_layer': 1, 'neu_output_layer': 1, 'batch_size': 100, 'n_hidden_layers': 2, 'neu_in_layers':[85,5,10],
         'non_linearity': 'sigmoid', 'learning_rate': 0.5}
len_config = len(config)
file_config = open(path_config, 'r')

for params in config.keys():
    if params == 'non_linearity':
        config[params] = file_config.readline().strip()
    elif params == 'learning_rate':
        config[params] = float(file_config.readline().strip())
    elif params == 'neu_in_layers':
        temp_string = file_config.readline().split()
        config[params] = [int(x) for x in temp_string]
    else:
        config[params] = int(file_config.readline().strip())
file_config.close()

### learning model for dnn ####
if question_part =='b':
    parameters = L_layer_model(train_X.T, train_Y.T, batch=config['batch_size'] ,layers_dims=config['neu_in_layers'], learning_rate = config['learning_rate'], num_iterations = 800, print_cost = True)

if question_part == 'c':
    acc_train = []
    acc_test = []
    train_time = []
    #for l in range(1,20,1):
    for i in [5, 10, 15, 20, 25]:
        config['neu_in_layers'] = [85, i, 10]
        start = time.time()
        parameters = L_layer_model(train_X.T, train_Y.T, batch=config['batch_size'],
                                   layers_dims=config['neu_in_layers'],
                                   learning_rate=0.04, num_iterations=500, print_cost=True)
        end = time.time()
        train_time.append(end - start)
        acc_tr, p_tr = predict(train_X.T, train_Y.T, parameters)
        acc_train.append(acc_tr)
        acc_te, p_te = predict(test_X.T, test_Y.T, parameters)
        acc_test.append(acc_te)
        # np.argmax(p_te, axis=0)
        print(confusion_matrix(np.argmax(test_Y.T, axis=0), np.argmax(p_te, axis=0)))

        print("training time", train_time)
        print("training Accuracy", acc_train)
        print("test accuracy", acc_test)

