import numpy as np
# import sys

# print sys.argv[0] # prints python_script.py
# path_train = sys.argv[1] # prints var1
# path_test = sys.argv[2] # prints var2
# question_part = sys.argv[3] # prints

class Node(object):
    """
    node to contain attribute on which it splits the data
    """
    def __init__(self, label_feature_mat=None, left_node=None, right_node=None):
        self.label_feature_mat = label_feature_mat
        self.left_node = left_node
        self.right_node = right_node


    def entropy(self):
        temp_size = self.label_feature_mat.shape[0]
        return temp_size

class Leaf(object):
    """
    The leaf node which contains only the pure labels.
    """
    def __init__(self, label = None):
        self.label = label