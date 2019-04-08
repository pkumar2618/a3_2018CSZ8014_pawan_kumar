import numpy as np
import pandas as pd
# import sys
node_counter = 1
# print sys.argv[0] # prints python_script.py
# path_train = sys.argv[1] # prints var1
# path_test = sys.argv[2] # prints var2
# question_part = sys.argv[3] # prints

class Node:

    """
    node to contain attribute on which it splits the data
    """
    def __init__(self, data, feature=None, branches=[], branch_value=[], leaf_flag=False, leaf_label=None, node_id = None):
        """
        :param feature: the attribute on which self is split 
        :param branches: the new branches that would shoot from self. 
        """
        self.feature = feature
        self.leaf_flag = leaf_flag
        self.leaf_label = leaf_label
        self.branches= branches
        self.branch_value =branch_value
        self.data = data
        self.node_id = node_id

    def grow_tree(self):  # =  pd.DataFrame()):
        #  it will take as argument the dataset passed on the the new branch.
        dataset = self.data
        value, counts = np.unique(dataset['Y'].values, return_counts=True)
        # n_rows = len(dataset.index)
        # equal_zeros = pd.Series(np.zeros(n_rows))
        # equal_ones = pd.Series(np.ones(n_rows))
        if len(value)==1:
            # equal_zeros.equals(dataset['Y'])
            if int(value[0]) == 0:
                self.feature = 'Leaf'
                self.leaf_flag = True
                self.leaf_label = 0
                return 0
            elif int(value[0]) == 1:
                # equal_ones.equals(dataset['Y'])
                self.feature = 'Leaf'
                self.leaf_flag = True
                self.leaf_label = 1
                return 1
        # elif len(value) ==2:
        #     if counts
        else:
            # n_multiway = 0
            # branched_data = []
            if len(list(dataset)) - 1 == 0: #there is no attribute except Y in the passed dataset
                return 9
            else:
                attr = best_attribute(dataset)  # if attr is boolean return attribute label and [0,1],
                # else label as well as category list ['a', 'b'] the attributes
                self.feature = attr.name  # storing the best attribute found for this node
                self.branch_value = attr[:]
                new_nodes =[]
                for attr_val in attr[:]:
                    split_row_indices = (dataset[attr.name] == attr_val)
                    temp_branched_data = dataset.loc[split_row_indices, :]
                    branched_data = temp_branched_data.drop([attr.name],axis=1)
                    new_nodes.append(Node(data =branched_data))

                self.branches = new_nodes
                #bfs like growth
                for i in range(len(attr)):
                    self.branches[i].grow_tree()

    def grow_tree_predict(self, node_id_leaf_test=[], test_accu =[], test_y_pred_label=pd.Series(), node_id_leaf_val=[], val_accu=[],
                          val_y_pred_label=pd.Series(), test_data=pd.DataFrame(), val_data=pd.DataFrame()):
    # def grow_tree_predict(self, node_count, test_accu, test_y_pred_label,
    #                       val_accu, val_y_pred_label, test_data, val_data):
        #  it will take as argument the dataset passed on the the new branch.
        #  and the split based on the feature value of test_data
        # and val_data, as it encounters a leaf_node the node_count will be updated,
        # and accuracies will be updates as well
        global node_counter
        dataset = self.data
        test_data = test_data
        val_data = val_data
        test_accu = test_accu
        val_accu = val_accu
        value, counts = np.unique(dataset['Y'].values, return_counts=True)

        # When a leaf node is encountered, the accuracy should update
        if len(value) == 1:
            if int(value[0]) == 0:
                self.feature = 'Leaf'
                self.leaf_flag = True
                self.leaf_label = 0
                #predicting majority class
                test_pred_indices = test_data.loc[pd.to_numeric(test_data['Y'], downcast='signed')==0].index.tolist()
                val_pred_indices = val_data.loc[pd.to_numeric(val_data['Y'], downcast='signed') == 0].index.tolist()
                try:
                    test_y_pred_label.iloc[test_pred_indices] = True
                    # node_id_leaf_test.append(self.node_id)
                    node_id_leaf_test.append(node_counter)
                    temp_test_acc = test_y_pred_label.value_counts(normalize=True, dropna=True)[True]
                    test_accu.append(temp_test_acc)
                except:
                    pass
                try:
                    val_y_pred_label.iloc[val_pred_indices] = True
                    # node_id_leaf_val.append(self.node_id)
                    node_id_leaf_val.append(node_counter)
                    temp_val_acc = val_y_pred_label.value_counts(normalize=True, dropna=True)[True]
                    val_accu.append(temp_val_acc)
                except:
                    pass
                # return 0
            elif int(value[0]) == 1:  # equal_ones.equals(dataset['Y']):
                self.feature = 'Leaf'
                self.leaf_flag = True
                self.leaf_label = 1
                # predicting majority class
                test_pred_indices = test_data.loc[pd.to_numeric(test_data['Y'], downcast='signed') == 1].index.tolist()
                val_pred_indices = val_data.loc[pd.to_numeric(val_data['Y'], downcast='signed') == 1].index.tolist()
                try:
                    test_y_pred_label.iloc[test_pred_indices] = True
                    node_id_leaf_test.append(node_counter)
                    temp_test_acc = test_y_pred_label.value_counts(normalize=True, dropna=True)[True]
                    test_accu.append(temp_test_acc)
                except:
                    pass
                try:
                    val_y_pred_label.iloc[val_pred_indices] = True
                    node_id_leaf_val.append(node_counter)
                    temp_val_acc = val_y_pred_label.value_counts(normalize=True, dropna=True)[True]
                    val_accu.append(temp_val_acc)
                except:
                    pass
                # return 1
        # elif len(value) ==2:
        #     if counts
        else:
            # n_multiway = 0
            # branched_data = []
            if len(list(dataset)) - 1 == 0:  # there is no attribute except Y in the passed dataset
                return 9
            else:
                attr = best_attribute(dataset)  # if attr is boolean return attribute label and [0,1],
                # else label as well as category list ['a', 'b'] the attributes
                self.feature = attr.name  # storing the best attribute found for this node
                self.branch_value = attr[:]
                new_nodes = []
                test_branched_data = []
                val_branched_data = []
                for attr_val in attr[:]:
                    split_row_indices = (dataset[attr.name] == attr_val)
                    temp_branched_data = dataset.loc[split_row_indices, :]
                    branched_data = temp_branched_data.drop([attr.name], axis=1)
                    new_nodes.append(Node(data=branched_data))

                    # Splitting test and validation set data as well
                    try:
                        test_split_row_indices = (test_data[attr.name] == attr_val)
                        temp_branched_data_test = test_data.loc[test_split_row_indices, :]
                        test_branched_data_temp = temp_branched_data_test.drop([attr.name], axis=1)
                        test_branched_data.append(test_branched_data_temp)
                    except KeyError:
                        test_branched_data.append(pd.DataFrame())

                    try:
                        val_split_row_indices = (val_data[attr.name] == attr_val)
                        temp_branched_data_val = val_data.loc[val_split_row_indices, :]
                        val_branched_data_temp = temp_branched_data_val.drop([attr.name], axis=1)
                        val_branched_data.append(val_branched_data_temp)
                    except KeyError:
                        val_branched_data.append(pd.DataFrame())

                self.branches = new_nodes
                # counting the nodes and assigning new id to nodes
                # for i in range(len(attr)):
                #     node_counter = 1 + node_counter


                # bfs like growth
                for i in range(len(attr)):
                    # global node_counter
                    node_counter += 1
                    self.branches[i].node_id = node_counter
                    # self.branches[i].grow_tree_predict(node_counter, node_id_leaf_test, test_accu, test_y_pred_label, node_id_leaf_val, val_accu,
                    #                                    val_y_pred_label, test_branched_data[i], val_branched_data[i])
                    self.branches[i].grow_tree_predict(node_id_leaf_test, test_accu, test_y_pred_label, node_id_leaf_val, val_accu,
                                                       val_y_pred_label, test_branched_data[i], val_branched_data[i])

def entropy_Hy(data = pd.Series()):
    size = data.size
    value, counts = np.unique(data.values, return_counts=True)
    probs = counts / size
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0
    # Compute entropy
    return -(probs * np.log2(probs)).sum()

def entropy_px_Hyx(x_label, dataset = pd.DataFrame()):
    """

    :param x_label: feature x on which dataset will be split
    :param dataset: this dataset will be split into chunk sequal to feature values, and each chunk will return entropy_HY
    :return:
    """
    size = len(dataset.index)
    value, counts = np.unique(dataset[x_label].values, return_counts=True)
    probs_x = counts / size
    entropy_on_chunk = []
    ### splitting into chunks corresponding to values x will take. ####
    for attr_val in value:
        split_row_indices = (dataset[x_label] == attr_val)
        temp_chunk_data = dataset.loc[split_row_indices, :]
        entropy_on_chunk.append(entropy_Hy(temp_chunk_data['Y']))

    n_classes = np.count_nonzero(probs_x)
    if n_classes <= 1:
        return 0
    # Compute entropy

    return (probs_x * entropy_on_chunk).sum()

def best_attribute(dataset= pd.DataFrame()):

        """
        :param max_features: the number of feature to be considered for possible candidate, which will get maximum 
        infromation gain.   
        :return: the best attribute name and its values [0,1] for boolean and [list of values] for catogarical
        """
        list_attr = list(dataset)
        n_features = len(list_attr)-1 # removing column for 'Y'
        # if n_features == 0:
        #     return -1 # leaf node is reached.
        # entropy_vs_attr= []
        entropy_on_y = entropy_Hy(dataset['Y'])
        entropy_pxi_Hyx =  np.array([])
        # label_i = 0
        for label_i in range(n_features):
            current_attr_data = dataset.iloc[:, label_i]
            entropy_pxi_Hyx = np.append(entropy_pxi_Hyx, entropy_px_Hyx(list_attr[label_i], dataset))

        ## find the best atrribute based on information gain.
        info_gain = entropy_on_y - entropy_pxi_Hyx
        attr_i = np.argmax(info_gain)
        values = np.unique(dataset.iloc[:, attr_i].values, return_counts=False)
        feature =  list_attr[attr_i] # the best feature getting maximum information gain
        return pd.Series(values,name=feature)

class Leaf:
    """
    The leaf node which contains only the pure labels.
    """
    def __init__(self, label = None):
        self.label = label