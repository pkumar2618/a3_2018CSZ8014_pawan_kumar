from node import Node, Leaf
from node import best_attribute

class Tree:
    """
    create decision tree with nodes as defined in the module node.py.
    """
    def __init__(self, root = None):
        self.root = root

    # def entropy(self,):
    #     temp = 4
    #     return temp