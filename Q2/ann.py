import numpy as np
import pandas as pd

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
# label_name = train_XY.iloc[0, :]

# label_dict = {}
# attr_n = 0
# for label in list(train_XY):
#     label_dict[label] = attr_n
#     attr_n += 1

# train_XY = train_XY.drop([0], axis =0)
# test_XY = test_XY.drop([0], axis =0)
# val_XY = val_XY.drop([0], axis =0)

if question_part == 'a':


if question_part == 'b':


if question_part == 'c':


if question_part == 'd':


if question_part == 'e':


if question_part == 'f':
