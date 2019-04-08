import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from node import Node
import sys
# from node import best_attribute
# from tree import Tree
# from tree import grow_tree

# node_counter = 1


# print sys.argv[0] # prints python_script.py

question_part = sys.argv[1] # prints
path_train = sys.argv[2] # prints var1
path_test = sys.argv[3]# prints var2
path_val = sys.argv[4] # prints var2

# path_train = "../../ass3_data/credit-cards.train.csv"
# path_test = "../../ass3_data/credit-cards.test.csv"
# path_val = "../../ass3_data/credit-cards.val.csv"
# path_train = "../../../ass3_data/credit-cards.train.csv"
# path_test = "../../../ass3_data/credit-cards.test.csv"
# path_val = "../../../ass3_data/credit-cards.val.csv"
# question_part = 'a' # prints


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
# train_XY =train_XY.iloc[0:50,:]

test_XY = test_XY.drop([0], axis =0)
test_XY = test_XY.drop('X0', axis =1)
# test_XY = test_XY.iloc[0:10,:]

val_XY = val_XY.drop([0], axis =0)
val_XY = val_XY.drop('X0', axis =1)
# val_XY = val_XY.iloc[0:5,:]


# preprocessing the data based on their labels
cont_attr = ["X1", "X5", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23"]
for label in cont_attr:
    median = train_XY[label].median()
    train_XY[label] = train_XY[label].apply(lambda x: 0 if float(x) < median else 1)
    test_XY[label] = test_XY[label].apply(lambda x: 0 if float(x) < median else 1)
    val_XY[label] = val_XY[label].apply(lambda x: 0 if float(x) < median else 1)



if question_part == '1':
    # dt_fit = Tree()
    # n_features = len(list(train_XY))-1 # and Y removed.
    # feature = best_attribute(dataset= pd.DataFrame(), max_features = n_features)
    root_node = Node(data=train_XY)
    # root_node.grow_tree()

    node_id_leaf_train = []
    train_accu = []
    train_y_pred_label=pd.Series(np.zeros(train_XY.shape[0]))

    node_id_leaf_test=[]
    test_accu=[]
    test_y_pred_label=pd.Series(np.zeros(test_XY.shape[0]))

    node_id_leaf_val = []
    val_accu=[]
    val_y_pred_label=pd.Series(np.zeros(val_XY.shape[0]))

    test_data=test_XY
    val_data=val_XY


    # root_node.grow_tree_predict(node_counter, node_id_leaf_test, test_accu, test_y_pred_label, node_id_leaf_val, val_accu, val_y_pred_label, test_data, val_data)
    root_node.grow_tree_predict(node_id_leaf_train, train_accu, train_y_pred_label, node_id_leaf_test, test_accu, test_y_pred_label, node_id_leaf_val,
                                val_accu, val_y_pred_label, test_data, val_data)
    fig1  = plt.figure()

    # node

    # node_id_leaf_test = np.unique(node_id_leaf_test, return_counts=False)
    # test_accu = np.unique(test_accu, return_counts=False).reshape((len(node_id_leaf_test), -1))
    #
    # node_id_leaf_val = np.unique(node_id_leaf_val, return_counts=False)
    # val_accu = np.unique(val_accu, return_counts=False).reshape((len(node_id_leaf_val), -1))

    node_id_leaf_train.sort()
    train_accu.sort()

    node_id_leaf_test.sort()
    test_accu.sort()

    node_id_leaf_val.sort()
    val_accu.sort()

    plt.plot(node_id_leaf_train[0:len(train_accu)], train_accu, label="train set accuracy")
    plt.plot(node_id_leaf_test[0:len(test_accu)], test_accu, label="test set accuracy")
    plt.plot(node_id_leaf_val[0:len(val_accu)], val_accu, label="val set accuracy")

    # test_node_acc = np.unique(np.vstack((node_id_leaf_test, test_accu)), axis=1, return_counts=False)
    # val_node_acc = np.unique(np.vstack((node_id_leaf_val, val_accu)), axis= 1, return_counts=False)
    # test_node_acc = np.vstack((node_id_leaf_test, test_accu))
    # test_node_acc = test_node_acc.sort(axis =1)
    # val_node_acc = np.vstack((node_id_leaf_val, val_accu))
    # val_node_acc = val_node_acc.sort(axis=1)
    # plt.plot(test_node_acc[0,:],test_node_acc[1,:], label= "test set accuracy")
    # plt.plot(val_node_acc[0,:],val_node_acc[1,:], label="validation set accuracy")

    plt.legend()
    plt.xlabel("Node count in the tree")
    plt.ylabel("Accuracy")
    plt.title("Val-set and Test-set accuracy with growing d-tree.")
    plt.show()

    # print("Tree complete")

if question_part == '2':
    #Grow the tree first
    root_node = Node(data=train_XY)
    # root_node.grow_tree()

    # build the decision tree for the first time
    root_node.grow_tree()

    # Counting the nodes
    # nodes_in_tree = root_node.node_count()
    # print("node in tree", nodes_in_tree)
    train_data = train_XY.copy(deep=True)
    train_nodes = []
    train_accu = []
    train_y_pred_label = pd.Series(np.zeros(train_XY.shape[0]))

    test_data = test_XY.copy(deep=True)
    test_nodes = []
    test_accu = []
    test_y_pred_label = pd.Series(np.zeros(test_XY.shape[0]))

    val_data = val_XY.copy(deep=True)
    val_nodes = []
    val_accu = []
    val_y_pred_label = pd.Series(np.zeros(val_XY.shape[0]))

    root_node.post_prunning_accu(val_data, val_accu, val_nodes,
                       test_data, test_accu, test_nodes,
                       train_data, train_accu, train_nodes)

    val_accu.sort(reverse=True)
    old_val_acc = 0
        # getting accuracy and pruning
    # pruning of validation set

    while(root_node.leaf_flag == False):
        root_node.prune_tree()
        train_data = train_XY.copy(deep=True)
        train_nodes = []
        train_accu = []
        train_y_pred_label = pd.Series(np.zeros(train_XY.shape[0]))

        test_data = test_XY.copy(deep=True)
        test_nodes = []
        test_accu = []
        test_y_pred_label = pd.Series(np.zeros(test_XY.shape[0]))

        val_data = val_XY.copy(deep=True)
        val_nodes = []
        val_accu = []
        val_y_pred_label = pd.Series(np.zeros(val_XY.shape[0]))

        root_node.post_prunning_accu(val_data, val_accu, val_nodes,
                           test_data, test_accu, test_nodes,
                           train_data, train_accu, train_nodes)
        new_val_acc = val_accu.sort(reverse=True)
        if (new_val_acc < old_val_acc):
            break
        else:
            root_node.prune()



if question_part == '3':

if question_part == '4':
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

    clf_config = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=22, min_samples_split=70,
                                             max_depth=7)
    decision_tree = clf_config.fit(train_X, train_Y)
    val_Y_pred = decision_tree.predict(val_X)
    val_accu = accuracy_score(val_Y, val_Y_pred) * 100

    train_Y_pred = decision_tree.predict(train_X)
    train_accu = accuracy_score(train_Y, train_Y_pred) * 100

    test_Y_pred = decision_tree.predict(test_X)
    test_accu = accuracy_score(test_Y, test_Y_pred) * 100
    print("Trains set Accuracy:", train_accu)
    print("Validation set Accuracy:", val_accu)
    print("Test set Accuracy:", test_accu)

if question_part == '5':
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

if question_part == '6':
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


    clf_config = RandomForestClassifier(n_estimators=90, max_features=33, bootstrap=True)
    r_forest = clf_config.fit(train_X, train_Y)

    val_Y_pred = r_forest.predict(val_X)
    val_accu = accuracy_score(val_Y, val_Y_pred) * 100

    train_Y_pred = r_forest.predict(train_X)
    train_accu = accuracy_score(train_Y, train_Y_pred) * 100

    test_Y_pred = r_forest.predict(test_X)
    test_accu = accuracy_score(test_Y, test_Y_pred) * 100
    print("Trains set Accuracy:", train_accu)
    print("Validation set Accuracy:", val_accu)
    print("Test set Accuracy:", test_accu)
