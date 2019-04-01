import numpy as np
import pandas as pd


from node import Node
from tree import Tree

# print sys.argv[0] # prints python_script.py

# path_train = sys.argv[1] # prints var1
# path_test = sys.argv[2] # prints var2
# path_val = sys.argv[3] # prints var2

path_train = "../../../ass3_data/credit-cards.train.csv"
path_test = "../../../ass3_data/credit-cards.test.csv"
path_val = "../../../ass3_data/credit-cards.val.csv"

# question_part = sys.argv[4] # prints

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
test_XY = test_XY.drop([0], axis =0)
val_XY = val_XY.drop([0], axis =0)



# preprocessing the data based on their labels
cont_attr = ["X1", "X5", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23"]
for label in cont_attr:
    median = train_XY[label].median()
    train_XY[label] = train_XY[label].apply(lambda x: 0 if float(x) < median else 1)

def GrowTree(dataset_set = pd.DataFrame([])): # it will take as argument the dataset
    for nth_splits in range(len(list(dataset))):
        n_row = len(dataset_set[nth_split].index)
        equal_zeros = pd.Series(np.zeros((n_row)))
        equal_ones = pd.Series(np.ones((n_row)))
        if equal_zeros.equals(dataset_set[nth_splits]['Y']):
            return Leaf(0)
        if equal_ones.equals(dataset_set[nth_splits]['Y']):
            return Leaf(1)
        else:
            attr = BestAttribute(dataset_set[nth_splits]) # if attr is boolean return attribute lable, else label as well as category of the attributes
            n_multiway =0
            for attr_val in [0,1]: # modify for multiway attribute later
                split_row_indices = (dataset_set[nth_splits][attr] == attr_val).reshape(-1)
                branched_data[n_multiway] = dataset_set[nth_splits].iloc[split_row_indices,:]
                n_multiway += 1
            GrowTree(branced_data)

def BestAttribute(dataset = pd.DataFrame):

if question_part == 'a':

if question_part == 'b':

if question_part == 'c':

if question_part == 'd':
    """
    using sciki - learn library to grow a decision tree
    """

if question_part == 'e':

if question_part == 'f':

# root_node = Node(mat_labels_features)
# root = Tree(root_node)

# print(root_node.entropy())


