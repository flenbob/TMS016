from cv2 import boxPoints, threshold
import numpy as np
from sklearn.metrics import confusion_matrix
import math
import h5py
from sklearn.model_selection import StratifiedKFold

# y_prob = clf.predict_proba(X)[:, 1]
# FPR_list, TPR_list, threshold_list = roc_curve(y, y_prob)

# FNR_list = 1 - TPR_list
# dfplot=pd.DataFrame({'Threshold':threshold_list, 
# 'False Positive Rate':FPR_list, 
# 'False Negative Rate': FNR_list})

# ax=dfplot.plot(x='Threshold', y=['False Positive Rate',
# 'False Negative Rate'], figsize=(10,6))
# ax.plot([thresh,thresh],[0,0.2]) #mark selected thresh
# plt.show()

X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 1])

skf = StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
for train_index, test_index in skf.split(X, y):
    print('---------')
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print(X_train, X_test)
    print(y_train, y_test)

# print(np.concatenate((X, Y)))
# y_FP = np.array(len(X)*[0])
# print(y_FP)


# # #X_FP = np.ndarray(shape=(0, 3))
# X_FP = np.empty(shape=(0, 3))
# X = np.array([[1, 2, 3],[1,2,3],[1,2,3]])
# print(np.concatenate((X, X_FP)))

# print(len(X))
# ys = np.array([1, 0, 1, 0])
# y = np.array(len(X)*[0])
# y = y.reshape((-1,1))

# FP_idx = np.where(ys == 0)[0]
# #X_FP.append(X[FP_idx])
# #X_FP = np.append(X_FP, X[FP_idx])
# X_FP = np.append(X_FP, X[FP_idx], axis = 0)
# X_FP = np.append(X_FP, X[np.where(ys == 1)[0]], axis = 0)
# print(X_FP)

# print(y)
# print(np.where(ys == 1)[0])

# file = h5py.File('data/positive_test.hdf5', 'r')
# keys = file.keys()
# keys = list(keys)
