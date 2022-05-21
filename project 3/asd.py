import numpy as np
from sklearn.utils import shuffle


X_FP = np.empty(shape=(0, 2))
X = np.array([[0, 1],[2,3], [4,5]])

X_FP = np.append(X, X_FP, axis=0)
print(X_FP)

