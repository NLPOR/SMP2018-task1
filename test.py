#encoding:utf-8
import numpy as np
from sklearn.cross_validation import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(4, n_folds=3)
for train_index, test_index in kf:
    print("TRAIN:", train_index, "TEST:", test_index)
    print X[train_index], y[train_index]