from typing import Any
from PIL import Image, ImageOps
import numpy as np
import os
from skimage.feature import haar_like_feature, haar_like_feature_coord
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


def get_FP_samples(models_reduced: Any, min_FPR: int, file_path: str, last_visited: str, scale: float, delta: float) -> np.ndarray:
    n_f = 162336
    X_FP = np.empty(shape=(0, n_f))

    #Read image file from filepath and begin from last visited
    dirs = os.listdir(file_path)
    if last_visited != None:
        dirs = dirs[dirs.index(last_visited)+1:-1]

    for img_path in dirs:
        img = Image.open(f'{file_path}/{img_path}')
        img = ImageOps.grayscale(img)
        FP_img_ii_list, _ = cascade_scan(models_reduced, img, scale, delta)

        for FP_img_ii in FP_img_ii_list:
            X = haar_like_feature(FP_img_ii, 0, 0, 24, 24)
            X = X.reshape(1, -1)
            X_FP = np.append(X_FP, X, axis=0)
            print(len(X_FP))
            if len(X_FP) >= min_FPR:
                return X_FP, img_path
    return X_FP, 'EOF'

def cross_validate(clf: Any, X: np.ndarray, y: np.ndarray, FNR_target: float, FPR_target: float, n_splits: int) -> list:
    #Input:
    # clf: SC
    # X, y: Dataset and labels
    # FNR_target: target false negative rate [0, 1]
    # FPR_target: target false positive rate [0, 1]
    # n_splits: Number of splits in cross validation
    #
    #Output:
    # clf_best: Best SC with its threshold, and feature specifications
    # flag: True if SC is accepted, False otherwise.
    skf = KFold(n_splits=n_splits)
    FNR_fold = []
    FPR_fold = []
    FNR_best = 1
    FPR_best = 1
    thresh = 0.5 #Initial threshold for SC
    X_KFold, X_test, y_KFold, y_test, = train_test_split(X, y, train_size=0.75, random_state=1, stratify=y)
    i = 0
    for train_idx, test_idx in skf.split(X_KFold):
        i += 1
        print(f'Fold {i}')
        X_train, X_validate = X_KFold[train_idx], X_KFold[test_idx]
        y_train, y_validate = y_KFold[train_idx], y_KFold[test_idx]
        
        #Train SC
        clf.fit(X_train, y_train)

        #Train SC w.r.t selected features of train clf_reduced = [model, threshold, feature_type, feature_coord, feature_idx]
        clf_reduced = adaboost_feature_reduce(clf, X_train, y_train)

        #Tune hyperparameter FPR and FNR of SC (reduced) and then validate
        y_prob = clf_reduced[0].predict_proba(X_validate[:, clf_reduced[4]])[:, -1]
        FPR_list, TPR_list, threshold_list = roc_curve(y_validate, y_prob)
        FNR_list = 1 - TPR_list
        thresh, FPR, FNR = optimize_threshold(FNR_list, FPR_list, threshold_list, FNR_target)
        clf_reduced[1] = thresh
        print(f'Threshold = {thresh} gives FNR = {FNR}, FPR = {FPR}')

        #If feasible and best performing, save model and threshold
        if FPR < FPR_target and FNR < FNR_target and FPR < FPR_best and FNR < FNR_best:
            clf_best = clf_reduced
        
        FNR_fold.append(FNR)
        FPR_fold.append(FPR)

    #If the mean performance of our models is worse than target, then we need more WC:s
    if sum(FNR_fold)/len(FNR_fold) > FNR_target or sum(FPR_fold)/len(FPR_fold) > FPR_target:
        flag = False
        clf_best = None
        return clf_best, flag

    #Run best model and its threshold on test data and check for overfitting
    y_pred = 1*(clf_best[0].predict_proba(X_test[:, clf_best[4]])[:, -1] >= clf_best[1])
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    FNR = FN/(FN+TP)
    FPR = FP/(FP+TN)
    print(f'VALIDATION SET: Threshold: {clf_best[1]} gives: FNR = {FNR}, FPR = {FPR}')

    #Return
    flag = True
    return clf_best, flag
         
   
def read_feature_file(path_read: str) -> tuple[np.ndarray, np.ndarray]:
    #Read feature hdf5 file and return:
    #X: features for each image
    #y: labels for each image 
    file = h5py.File(f'{path_read}', 'r')
    X = file['features'] 
    y = file['labels']
    X = np.array(X)
    y = np.array(y)
    file.close()
    return X, y