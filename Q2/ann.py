import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
# from pandas import ConfusionMatrix
import matplotlib.pyplot as plt
import sys
from model_dnn import L_layer_model
from model_dnn import adaptive_learning_dnn_model
from model_dnn import adaptive_learning_dnn_model_with_relu
from ann_utils import predict, predict_with_relu

import time

# print sys.argv[0] # prints python_script.py

# path_config= sys.argv[1] # prints var1
# path_train_hot = sys.argv[2] # prints var1
# path_test_hot = sys.argv[3] # prints var2

path_config = '/home/dell/Documents/2nd_sem/ml_assig/a3_2018CSZ8014_pawan_kumar/Q2/config_c.txt'
path_train_hot = '/home/dell/Documents/2nd_sem/ml_assig/ass3_data/poker-hand-training-true-onehot.data'
path_test_hot = '/home/dell/Documents/2nd_sem/ml_assig/ass3_data/poker-hand-testing-onehot.data'
question_part = 'f'

train_XY = pd.read_csv(path_train_hot)
train_XY = train_XY.sample(frac=1).reset_index(drop=True)
### use for a single example
# train_X = train_XY.loc[0,:'C5_13'].to_numpy(copy=True).reshape(1,-1)
# train_Y = train_XY.loc[0,'Hand_0':'Hand_9'].to_numpy(copy=True).reshape(1,-1)
train_X = train_XY.loc[:,:'C5_13'].to_numpy(copy=True)
train_Y = train_XY.loc[:,'Hand_0':'Hand_9'].to_numpy(copy=True)

test_XY = pd.read_csv(path_test_hot)
test_XY = test_XY.sample(frac=0.5).reset_index(drop=True)
test_X = test_XY.loc[:,:'C5_13'].to_numpy(copy=True)
test_Y = test_XY.loc[:,'Hand_0':'Hand_9'].to_numpy(copy=True)

### Reading Configuration file ###
config ={'neu_input_layer': 1, 'neu_output_layer': 1, 'batch_size': 1, 'n_hidden_layers': 2, 'neu_in_layers':[85,5,10],
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
    parameters = L_layer_model(train_X.T, train_Y.T, batch=config['batch_size'] ,layers_dims=config['neu_in_layers'], learning_rate = config['learning_rate'], num_iterations = 400, print_cost = True)

if question_part == 'c':
    acc_train = []
    acc_test = []
    train_time = []
    config['batch_size'] =100
    #for l in range(1,20,1):
    for i in [5, 10, 15, 20, 25]:
        config['neu_in_layers'] = [85, i, 10]
        start = time.time()
        parameters = L_layer_model(train_X.T, train_Y.T, batch=config['batch_size'],
                                   layers_dims=config['neu_in_layers'],
                                   learning_rate=0.1, num_iterations=700, print_cost=True)
        end = time.time()
        train_time.append(end - start)
        acc_tr, p_tr = predict(train_X.T, train_Y.T, parameters)
        acc_train.append(acc_tr)
        acc_te, p_te = predict(test_X.T, test_Y.T, parameters)
        acc_test.append(acc_te)
        # temp = np.argmax(p_te, axis=0)
        hands = np.arange(10).reshape(10,-1)
        # p_te_labels = pd.Series(np.sum(p_te * hands, axis = 0), name='predicted')
        # true_y_labels = pd.Series(np.sum(test_Y.T * hands, axis = 0), name='actual')
        # print(pd.crosstab(true_y_labels, p_te_labels))

        p_te_labels = np.sum(p_te * hands, axis=0)
        true_y_labels = np.sum(test_Y.T * hands, axis=0)
        print(confusion_matrix(true_y_labels, p_te_labels)) #labels=["true", "predicted"]))

        print("training time", train_time)
        print("training Accuracy", acc_train)
        print("test accuracy", acc_test)

    fig1 = plt.figure()
    grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.3)
    ax1 = fig1.add_subplot(grid[0, 0])
    ax2 = fig1.add_subplot(grid[0, 1])

    line1 = ax1.plot([5, 10, 15, 20, 25], acc_train, label='train set accuracy')
    line2 = ax1.plot([5, 10, 15, 20, 25], acc_test, label='test set accuracy')
    ax1.legend()
    ax1.set_xlabel("Neurons in the hidden layer")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy for train and test set")

    line3 = ax2.plot([5, 10, 15, 20, 25], train_time, label='training time in minutes ')
    ax2.legend()
    ax2.set_xlabel("Neurons in the hidden layer")
    ax2.set_ylabel("Training time")
    ax2.set_title("Training time for a single hidden layer with varying units in it.")
    plt.show()

if question_part == 'd':
    ## network with two hidden layers

    print("for double layer:")
    acc_train = []
    acc_test = []
    train_time = []
    for i in [5, 10, 15, 20, 25]:
        config['neu_in_layers'] = [85, i, i, 10]
        start = time.time()
        parameters = L_layer_model(train_X.T, train_Y.T, batch=config['batch_size'],
                                   layers_dims=config['neu_in_layers'],
                                   learning_rate=0.1, num_iterations=700, print_cost=True)
        end = time.time()
        train_time.append(end - start)
        acc_tr, p_tr = predict(train_X.T, train_Y.T, parameters)
        acc_train.append(acc_tr)
        acc_te, p_te = predict(test_X.T, test_Y.T, parameters)
        acc_test.append(acc_te)
        # np.argmax(p_te, axis=0)
        hands = np.arange(10).reshape(10, -1)
        # p_te_labels = pd.Series(np.sum(p_te * hands, axis = 0), name='predicted')
        # true_y_labels = pd.Series(np.sum(test_Y.T * hands, axis = 0), name='actual')
        # print(pd.crosstab(true_y_labels, p_te_labels))

        p_te_labels = np.sum(p_te * hands, axis=0)
        true_y_labels = np.sum(test_Y.T * hands, axis=0)
        print(confusion_matrix(true_y_labels, p_te_labels))  # labels=["true", "predicted"]))

        print("training time", train_time)
        print("training Accuracy", acc_train)
        print("test accuracy", acc_test)

    fig1 = plt.figure()
    grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.3)
    ax1 = fig1.add_subplot(grid[0, 0])
    ax2 = fig1.add_subplot(grid[0, 1])

    line1 = ax1.plot([5, 10, 15, 20, 25], acc_train, label='train set accuracy')
    line2 = ax1.plot([5, 10, 15, 20, 25], acc_test, label='test set accuracy')
    ax1.legend()
    ax1.set_xlabel("Neurons in the two hidden layer")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy for train and test set")

    line3 = ax2.plot([5, 10, 15, 20, 25], train_time, label='training time in minutes ')
    ax2.legend()
    ax2.set_xlabel("Neurons in the two hidden layer")
    ax2.set_ylabel("Training time")
    ax2.set_title("Training time for a two hidden layer with varying units in it.")
    plt.show()


if question_part == 'e':
    # adaptive learning rate model
    # print("for single layer:")
    acc_train = []
    acc_test = []
    train_time = []
    for i in [5, 10, 15, 20, 25]:
        config['neu_in_layers'] = [85, i, 10]
        start = time.time()
        parameters = adaptive_learning_dnn_model(train_X.T, train_Y.T, batch=config['batch_size'],
                                   layers_dims=config['neu_in_layers'],
                                   learning_rate=0.1, num_iterations=700, print_cost=True)
        end = time.time()
        train_time.append(end - start)
        acc_tr, p_tr = predict(train_X.T, train_Y.T, parameters)
        acc_train.append(acc_tr)
        acc_te, p_te = predict(test_X.T, test_Y.T, parameters)
        acc_test.append(acc_te)
        hands = np.arange(10).reshape(10, -1)
        # p_te_labels = pd.Series(np.sum(p_te * hands, axis = 0), name='predicted')
        # true_y_labels = pd.Series(np.sum(test_Y.T * hands, axis = 0), name='actual')
        # print(pd.crosstab(true_y_labels, p_te_labels))

        p_te_labels = np.sum(p_te * hands, axis=0)
        true_y_labels = np.sum(test_Y.T * hands, axis=0)
        print(confusion_matrix(true_y_labels, p_te_labels))  # labels=["true", "predicted"]))

        print("training time", train_time)
        print("training Accuracy", acc_train)
        print("test accuracy", acc_test)

    # print("for double layer:")
    # acc_train = []
    # acc_test = []
    # train_time = []
    for i in [5, 10, 15, 20, 25]:
        config['neu_in_layers'] = [85, i, i, 10]
        start = time.time()
        parameters = adaptive_learning_dnn_model(train_X.T, train_Y.T, batch=config['batch_size'],
                                   layers_dims=config['neu_in_layers'],
                                   learning_rate=0.1, num_iterations=700, print_cost=True)
        end = time.time()
        train_time.append(end - start)
        acc_tr, p_tr = predict(train_X.T, train_Y.T, parameters)
        acc_train.append(acc_tr)
        acc_te, p_te = predict(test_X.T, test_Y.T, parameters)
        acc_test.append(acc_te)
        hands = np.arange(10).reshape(10, -1)
        # p_te_labels = pd.Series(np.sum(p_te * hands, axis = 0), name='predicted')
        # true_y_labels = pd.Series(np.sum(test_Y.T * hands, axis = 0), name='actual')
        # print(pd.crosstab(true_y_labels, p_te_labels))

        p_te_labels = np.sum(p_te * hands, axis=0)
        true_y_labels = np.sum(test_Y.T * hands, axis=0)
        print(confusion_matrix(true_y_labels, p_te_labels))  # labels=["true", "predicted"]))

        print("training time", train_time)
        print("training Accuracy", acc_train)
        print("test accuracy", acc_test)


    fig1 = plt.figure()
    grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.3)
    ax1 = fig1.add_subplot(grid[0, 0])
    ax2 = fig1.add_subplot(grid[0, 1])

    line1 = ax1.plot([5, 10, 15, 20, 25], acc_train[0:5], label='train set accuracy')
    line2 = ax1.plot([5, 10, 15, 20, 25], acc_test[0:5], label='test set accuracy')
    ax1.legend()
    ax1.set_xlabel("Neurons in the hidden layer")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy for train and test set with adaptive learning rate")

    line3 = ax2.plot([5, 10, 15, 20, 25], train_time[0:5], label='training time in minutes ')
    ax2.legend()
    ax2.set_xlabel("Neurons in the the hidden layer")
    ax2.set_ylabel("Training time")
    ax2.set_title("Training time for a single hidden layer with varying units in it.")
    plt.show()

    fig2 = plt.figure()
    grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.3)
    ax3 = fig2.add_subplot(grid[0, 0])
    ax4 = fig2.add_subplot(grid[0, 1])

    line4 = ax3.plot([5, 10, 15, 20, 25], acc_train[5:10], label='train set accuracy')
    line5 = ax3.plot([5, 10, 15, 20, 25], acc_test[5:10], label='test set accuracy')
    ax3.legend()
    ax3.set_xlabel("Neurons in the two hidden layer, having same neurons")
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Accuracy for train and test set with adaptive learning rate")

    line6 = ax4.plot([5, 10, 15, 20, 25], train_time[5:10], label='training time in minutes ')
    ax4.legend()
    ax4.set_xlabel("Neurons in the two hidden layer")
    ax4.set_ylabel("Training time")
    ax4.set_title("Training time for a two hidden layer with varying units in it.")
    plt.show()

if question_part == 'f':
    ## adaptive learning with reLu in the hidden layer
        # adaptive learning rate model
        # print("for single layer_with relu in the hide:")
        acc_train = []
        acc_test = []
        train_time = []
        for i in [5, 10, 15, 20, 25]:
            config['neu_in_layers'] = [85, i, 10]
            start = time.time()
            parameters = adaptive_learning_dnn_model_with_relu(train_X.T, train_Y.T, batch=config['batch_size'],
                                                     layers_dims=config['neu_in_layers'],
                                                     learning_rate=0.1, num_iterations=700, print_cost=True)
            end = time.time()
            train_time.append(end - start)
            acc_tr, p_tr = predict_with_relu(train_X.T, train_Y.T, parameters)
            acc_train.append(acc_tr)
            acc_te, p_te = predict_with_relu(test_X.T, test_Y.T, parameters)
            acc_test.append(acc_te)

            hands = np.arange(10).reshape(10, -1)
            p_te_labels = np.sum(p_te * hands, axis=0)
            true_y_labels = np.sum(test_Y.T * hands, axis=0)
            print(confusion_matrix(true_y_labels, p_te_labels))  # labels=["true", "predicted"]))
            print("training time", train_time)
            print("training Accuracy", acc_train)
            print("test accuracy", acc_test)

    # print("for double layer with relu in the hide:")
        # acc_train = []
        # acc_test = []
        # train_time = []
        for i in [5, 10, 15, 20, 25]:
            config['neu_in_layers'] = [85, i, i, 10]
            start = time.time()
            parameters = adaptive_learning_dnn_model_with_relu(train_X.T, train_Y.T, batch=config['batch_size'],
                                                     layers_dims=config['neu_in_layers'],
                                                     learning_rate=0.1, num_iterations=700, print_cost=True)
            end = time.time()
            train_time.append(end - start)
            acc_tr, p_tr = predict_with_relu(train_X.T, train_Y.T, parameters)
            acc_train.append(acc_tr)
            acc_te, p_te = predict_with_relu(test_X.T, test_Y.T, parameters)
            acc_test.append(acc_te)

            hands = np.arange(10).reshape(10, -1)
            p_te_labels = np.sum(p_te * hands, axis=0)
            true_y_labels = np.sum(test_Y.T * hands, axis=0)
            print(confusion_matrix(np.argmax(test_Y.T, axis=0), np.argmax(p_te, axis=0)))
            print("training time", train_time)
            print("training Accuracy", acc_train)
            print("test accuracy", acc_test)

        fig1 = plt.figure()
        grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.3)
        ax1 = fig1.add_subplot(grid[0, 0])
        ax2 = fig1.add_subplot(grid[0, 1])

        line1 = ax1.plot([5, 10, 15, 20, 25], acc_train[0:5], label='train set accuracy')
        line2 = ax1.plot([5, 10, 15, 20, 25], acc_test[0:5], label='test set accuracy')
        ax1.legend()
        ax1.set_xlabel("Neurons in the hidden layer")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy for train and test set with \n adaptive learning rate and relu-activation")

        line3 = ax2.plot([5, 10, 15, 20, 25], train_time[0:5], label='training time in minutes ')
        ax2.legend()
        ax2.set_xlabel("Neurons in the the hidden layer")
        ax2.set_ylabel("Training time")
        ax2.set_title("Training time for a single hidden layer \n with varying units in it and relu-activation")
        plt.show()

        fig2 = plt.figure()
        grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.3)
        ax3 = fig2.add_subplot(grid[0, 0])
        ax4 = fig2.add_subplot(grid[0, 1])

        line4 = ax3.plot([5, 10, 15, 20, 25], acc_train[5:10], label='train set accuracy')
        line5 = ax3.plot([5, 10, 15, 20, 25], acc_test[5:10], label='test set accuracy')
        ax3.legend()
        ax3.set_xlabel("Neurons in the two hidden layer, having same neurons")
        ax3.set_ylabel("Accuracy")
        ax3.set_title("Accuracy for train and test set with adaptive learning rate \n and relu-activation in the hidden layers")

        line6 = ax4.plot([5, 10, 15, 20, 25], train_time[5:10], label='training time in minutes ')
        ax4.legend()
        ax4.set_xlabel("Neurons in the two hidden layer")
        ax4.set_ylabel("Training time")
        ax4.set_title("Training time for a two hidden layer with \n varying units in it and relu activation.")
        plt.show()