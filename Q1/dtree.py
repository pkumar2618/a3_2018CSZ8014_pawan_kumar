import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from node import Node
from node import best_attribute
from tree import Tree
# from tree import grow_tree




# print sys.argv[0] # prints python_script.py

# path_train = sys.argv[1] # prints var1
# path_test = sys.argv[2] # prints var2
# path_val = sys.argv[3] # prints var2
# question_part = sys.argv[4] # prints
path_train = "../../../ass3_data/credit-cards.train.csv"
path_test = "../../../ass3_data/credit-cards.test.csv"
path_val = "../../../ass3_data/credit-cards.val.csv"
question_part = 'a' # prints


# mat_labels_features = np.zeros((10,4))
train_XY = pd.read_csv(path_train, delimiter=',')
test_XY = pd.read_csv(path_test, delimiter=',')
val_XY = pd.read_csv(path_val, delimiter=',')
label_name = train_XY.iloc[0, :]

label_dict = {}
attr_n = 0
for label in list(train_XY):
    label_dict[label] = attr_n
    attr_n += 1

train_XY = train_XY.drop([0], axis =0)
train_XY = train_XY.drop('X0', axis =1)

test_XY = test_XY.drop([0], axis =0)
test_XY = test_XY.drop('X0', axis =1)

val_XY = val_XY.drop([0], axis =0)
val_XY = val_XY.drop('X0', axis =1)



# preprocessing the data based on their labels
cont_attr = ["X1", "X5", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23"]
for label in cont_attr:
    median = train_XY[label].median()
    train_XY[label] = train_XY[label].apply(lambda x: 0 if float(x) < median else 1)
    test_XY[label] = test_XY[label].apply(lambda x: 0 if float(x) < median else 1)
    val_XY[label] = val_XY[label].apply(lambda x: 0 if float(x) < median else 1)



if question_part == 'a':
    dt_fit = Tree()
    n_features = len(list(train_XY))-1 # and Y removed.
    # feature = best_attribute(dataset= pd.DataFrame(), max_features = n_features)
    dt_fit.root = Node().grow_tree(train_XY)

# if question_part == 'b':
#
# if question_part == 'c':

if question_part == 'd':
    """
    using sciki - learn library to grow a decision tree
    """
    train_X = train_XY.loc[:,'X1':'X23'].to_numpy(copy=True)
    train_Y = train_XY.loc[:,'Y'].to_numpy(copy=True)
    test_X = test_XY.loc[:,'X1':'X23'].to_numpy(copy=True)
    test_Y = test_XY.loc[:,'Y'].to_numpy(copy=True)
    val_X = val_XY.loc[:,'X1':'X23'].to_numpy(copy=True)
    val_Y = val_XY.loc[:,'Y'].to_numpy(copy=True)

    acc_min_leaf =[]
    acc_min_split = []
    acc_max_depth = []
    for min_leaf in range(1,100,5):
        clf_config = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=min_leaf, min_samples_split=2,
                                                 max_depth=None)
        decision_tree = clf_config.fit(train_X, train_Y)
        val_Y_pred = decision_tree.predict(val_X)

        acc_min_leaf.append(accuracy_score(val_Y, val_Y_pred)*100)
        # print("accuracy on test data", acc )

    for min_split in range(2, 100, 5):
        clf_config = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=min_split,
                                                 max_depth=None)
        decision_tree = clf_config.fit(train_X, train_Y)
        val_Y_pred = decision_tree.predict(val_X)

        acc_min_split.append(accuracy_score(val_Y, val_Y_pred)*100)

    for max_d in range(2, 1000, 5):
        clf_config = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=2,
                                                 max_depth=max_d)
        decision_tree = clf_config.fit(train_X, train_Y)
        val_Y_pred = decision_tree.predict(val_X)

        acc_max_depth.append(accuracy_score(val_Y, val_Y_pred)*100)

    fig1 = plt.figure()
    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)
    ax1 = fig1.add_subplot(grid[0,0])
    ax2 = fig1.add_subplot(grid[0,1])
    ax3 = fig1.add_subplot(grid[1,:])

    line1 = ax1.plot(range(1,100,5), acc_min_leaf, label = 'min_sample_leaf')
    ax1.legend()
    ax1.set_xlabel("range of values")
    ax1.set_ylabel("accuracy")
    ax1.set_title("accuracy vs min_sample_leaf for validation set")

    line2 = ax2.plot(range(2, 100, 5), acc_min_split, label = 'min_sample_split')
    ax2.legend()
    ax2.set_xlabel("range of values")
    ax2.set_ylabel("accuracy")
    ax2.set_title("accuracy vs min_sample_split for validation set")

    line3 = ax3.semilogx(range(2, 1000, 5), acc_max_depth, label = 'max depth')
    ax3.legend()
    ax3.set_xlabel("range of values")
    ax3.set_ylabel("accuracy")
    ax3.set_title("accuracy vs max_depth for validation set")


    # plt.legend((line1, line2, line3), ('min_sample_leaf', 'min_sample_split', 'max_depth'))
    plt.show()


if question_part == 'e':
    """
    using one-hot encoding for catagorical data and sciki - learn tree
    """
    train_X = pd.get_dummies(train_XY.loc[:,'X1':'X23'])
    train_Y = train_XY.loc[:,'Y'].to_numpy(copy=True)
    test_X = pd.get_dummies(test_XY.loc[:,'X1':'X23'])
    test_Y = test_XY.loc[:,'Y'].to_numpy(copy=True)
    val_X = pd.get_dummies(val_XY.loc[:,'X1':'X23'])
    val_Y = val_XY.loc[:,'Y'].to_numpy(copy=True)
    ## aligning dummies
    train_X_onehot, test_X_onehot= train_X.align(test_X, join='left', fill_value= 0, axis=1)
    train_X_onehot, val_X_onehot = train_X.align(val_X, join='left', fill_value= 0, axis=1)
    clf_config = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=22, min_samples_split=70,
                                                 max_depth=7)

    train_X = train_X_onehot.to_numpy(copy=True)
    test_X = test_X_onehot.to_numpy(copy=True)
    val_X = val_X_onehot.to_numpy(copy=True)

    decision_tree = clf_config.fit(train_X, train_Y)
    train_Y_pred = decision_tree.predict(train_X)
    val_Y_pred = decision_tree.predict(val_X)
    test_Y_pred = decision_tree.predict(test_X)

    acc_train = accuracy_score(train_Y, train_Y_pred)*100
    acc_val = accuracy_score(val_Y, val_Y_pred) * 100
    acc_test = accuracy_score(test_Y, test_Y_pred) * 100
    print("accuracy on train_set, validation_set and test_set are %f, %f, %f  respectively " % (acc_test, acc_val, acc_test) )

if question_part == 'f':
    """
    prediction using Random Forest
    """
    train_X = pd.get_dummies(train_XY.loc[:, 'X1':'X23'])
    train_Y = train_XY.loc[:, 'Y'].to_numpy(copy=True)
    test_X = pd.get_dummies(test_XY.loc[:, 'X1':'X23'])
    test_Y = test_XY.loc[:, 'Y'].to_numpy(copy=True)
    val_X = pd.get_dummies(val_XY.loc[:, 'X1':'X23'])
    val_Y = val_XY.loc[:, 'Y'].to_numpy(copy=True)
    ## aligning dummies
    train_X_onehot, test_X_onehot = train_X.align(test_X, join='left', fill_value=0, axis=1)
    train_X_onehot, val_X_onehot = train_X.align(val_X, join='left', fill_value=0, axis=1)

    train_X = train_X_onehot.to_numpy(copy=True)
    test_X = test_X_onehot.to_numpy(copy=True)
    val_X = val_X_onehot.to_numpy(copy=True)

    acc_n_estimators = []
    acc_max_features = []
    acc_bootstrap10 = []
    for n_tree in range(10, 100, 5):
        clf_config = RandomForestClassifier(n_estimators=n_tree)
        r_forest = clf_config.fit(train_X, train_Y)
        val_Y_pred = r_forest.predict(val_X)

        acc_n_estimators.append(accuracy_score(val_Y, val_Y_pred) * 100)
        # print("accuracy on test data", acc )
    attr_n = train_X.shape[1]
    for n_attr in range(train_X.shape[1]):
        clf_config = RandomForestClassifier(n_estimators = 10, max_features=n_attr+1)
        r_forest = clf_config.fit(train_X, train_Y)
        val_Y_pred = r_forest.predict(val_X)

        acc_max_features.append(accuracy_score(val_Y, val_Y_pred) * 100)

    for bootstrap_state in [True, False]:
        clf_config = RandomForestClassifier(n_estimators=100, max_features=None, bootstrap = bootstrap_state)
        r_forest = clf_config.fit(train_X, train_Y)
        val_Y_pred = r_forest.predict(val_X)

        acc_bootstrap10.append(accuracy_score(val_Y, val_Y_pred) * 100)

    fig1 = plt.figure()
    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)
    ax1 = fig1.add_subplot(grid[0, 0])
    ax2 = fig1.add_subplot(grid[0, 1])
    ax3 = fig1.add_subplot(grid[1, :])

    line1 = ax1.plot(range(10, 100, 5), acc_n_estimators, label='n_estimators')
    ax1.legend()
    ax1.set_xlabel("number of trees in the forest")
    ax1.set_ylabel("accuracy")
    ax1.set_title("accuracy vs n_estimators for validation set")

    line2 = ax2.plot(range(1, train_X.shape[1]+1), acc_max_features, label='max_features')
    ax2.legend()
    ax2.set_xlabel("max. number of features considered for split, while choosing the best attribute")
    ax2.set_ylabel("accuracy")
    ax2.set_title("accuracy vs max_features for validation set")

    line3 = ax3.plot([1,0], acc_bootstrap10, label='1=bootstrapping, 0=no-bootstrapping')
    ax3.legend()
    ax3.set_xlabel("Bootstrapping state")
    ax3.set_ylabel("accuracy")
    ax3.set_title("accuracy vs bootstrapping state for validation set")

    # plt.legend((line1, line2, line3), ('min_sample_leaf', 'min_sample_split', 'max_depth'))
    plt.show()
#
# # root_node = Node(mat_labels_features)
# # root = Tree(root_node)
#
# # print(root_node.entropy())


