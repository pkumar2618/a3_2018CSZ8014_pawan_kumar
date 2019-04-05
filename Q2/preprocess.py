import numpy as np
import pandas as pd
import sys
# print sys.argv[0] # prints python_script.py



# mat_labels_features = np.zeros((10,4))
# train_XY = pd.read_csv(path_train, delimiter=',')
# test_XY = pd.read_csv(path_test, delimiter=',')
# val_XY = pd.read_csv(path_val, delimiter=',')
question_part  = sys.argv[5]

if question_part == 'a':
    path_train = sys.argv[1] # prints var1
    path_test = sys.argv[2] # prints var2
    path_train_onehot = sys.argv[3] # prints
    path_test_onehot = sys.argv[4]

    # path_train = "../../../ass3_data/poker-hand-training-true.data"
    # path_test = "../../../ass3_data/poker-hand-testing.data"

    train_XY = pd.read_csv(path_train, delimiter=',', header=None, names = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Hand'])
    test_XY = pd.read_csv(path_test, delimiter=',', header=None, names = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Hand'])

    ### using one-hot encoding for catagorical data
    train_XY = pd.get_dummies(train_XY, columns=['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Hand'], dtype=int)
    # train_Y = train_XY.loc[:, 'Hand']

    test_XY = pd.get_dummies(test_XY, columns=['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Hand'], dtype =int)
    # test_Y = test_XY.loc[:, 'Hand']
    ## aligning dummies
    train_XY_onehot, test_XY_onehot = train_XY.align(test_XY, join='left', fill_value=0, axis=1)


    # file_location_train = "../../../ass3_data/poker-hand-training-true-onehot.data"
    # file_location_test = "../../../ass3_data/poker-hand-testing-onehot.data"
    file_location_train = path_train_onehot
    file_location_test = path_test_onehot

    # train_Xhot_Y = pd.concat([train_X_onehot, train_Y], axis = 1)
    # test_Xhot_Y = pd.concat([test_X_onehot, test_Y], axis = 1)
    train_XY_onehot.to_csv(file_location_train, index=False)
    test_XY_onehot.to_csv(file_location_test, index=False)
