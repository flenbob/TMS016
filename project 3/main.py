from typing import Any
from skimage.feature import haar_like_feature
from skimage.transform import integral_image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from time import time
import h5py
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math

def main():
    T = [1, 5, 25, 25, 50, 50, 75]
    FNR_thresh = 0.1
    
    X_pos, y_pos = read_hdf5('data/positive_test.hdf5')
    X_neg, y_neg = read_hdf5('data/negative_test.hdf5')
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    print('Files loaded')
    file_path = 'data/adaboost_classifiers_test.pkl'
    train_adaboost_classifiers(FNR_thresh, T, X, y, file_path)
    
    
def train_adaboost_classifiers(FNR_thresh: float, T: list, X: np.ndarray, y: np.ndarray, file_path: str) -> None:
    # Input:
    # FNR_thresh: Maximum False Negative Ratio threshold [0, 1]
    # T: List of number of weak classifiers (WC) for each layer of strong classifier (SC)
    # X: Matrix of haar features for each sample (image)
    # y: array of correct binary classification
    # file_path: Filepath including filename (ends with .pkl) (eg. folder1/folder2/file.pkl)
    
    # Returns:
    # None, but saves all SC:s and their thresholds to .pkl file
    
    #Set initial value to start while-loop
    TN = 1
    FN = 1
    FP = 1
    
    #Open file
    file = open(file_path, 'wb')
    while (TN and FN and FP) != 0:
        for t in T:
            print(f'Training SC with {t} WC:s....')
            #SC with t WC:s
            clf = AdaBoostClassifier(
                DecisionTreeClassifier(criterion='gini', max_depth=1), 
                algorithm="SAMME.R", 
                n_estimators=t
            )
            #Train SC
            clf.fit(X, y)
            y_pred = clf.predict(X)
            
            #Find probability threshold to reduce FPR to 1%
            print(np.shape(y_pred))
            TN, FP, FN, TP = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
            FNR = FN/(FN+TP)
            
            #If only TP samples remains
            if (FP and FN and TN) == 0:
                break
        
            print(f'Before thresh, FNR: {FNR}')
            #If probability threshold is not satisfied
            thresh = 0.5
            if FNR > FNR_thresh:
                y_prob = clf.predict_proba(X)[:, 1]
                FPR_list, TPR_list, threshold_list = roc_curve(y, y_prob)
                FNR_list = 1 - TPR_list
                thresh = threshold_FNR(FNR_list, threshold_list, FNR_thresh)
            
            #Predict labels with new threshold
            y_pred = 1*(clf.predict_proba(X)[:,1] >= thresh)
            print(f'New threshold: {thresh}')
        
            #Remove samples that are TN
            TN_idx = []
            for i in range(len(y_pred)):
                if (y[i] == 0) and (y_pred[i] == 0):
                    TN_idx.append(i)
        
            X = np.delete(X, TN_idx, axis=0)
            y = np.delete(y, TN_idx)
            y_pred = np.delete(y_pred, TN_idx)
            
            #Check confusion matrix
            TN, FP, FN, TP = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
            print(f'TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}')
            #Append SC and threshold to .pkl file
            pickle.dump([clf, thresh], file)
            
    file.close()
    print(f'Saved all SC:s and thresholds to: {file_path}')
    
    
def threshold_FNR(FNR_list: np.ndarray, threshold_list: np.ndarray, target_FNR: float) -> float:
    #Find threshold that satisfies target false negative rate (FNR)
    idx = 0
    FNR = FNR_list[0]
    #print(f'FNR list: {FNR_list}')
    #print(f'Threshold list: {threshold_list}')
    while FNR >= target_FNR:
        idx += 1
        FNR = FNR_list[idx]
    
    #Linearly approximate target threshold
    left_FNR = FNR_list[idx-1]
    right_FNR = FNR_list[idx]
    left_threshold = threshold_list[idx-1]
    right_threshold = threshold_list[idx]
    
    print(f'Closest FNR: {FNR_list[idx]} at idx = {idx}, and threshold: {threshold_list[idx]}')
    
    ratio = (left_FNR-target_FNR)/(left_FNR-right_FNR)
    target_thresh = left_threshold-(ratio*(left_threshold-right_threshold))
    return target_thresh


def img2haar_features(path_read: str, path_write: str, n_f: int, n_img: int, img_type: str) -> np.ndarray:
    #X: Haar feature for each image
    #y: label for each image
    X = np.empty((n_img, n_f))
    
    if img_type == 'positive':
        y = np.array([1]*n_img)
    elif img_type == 'negative':
        y = np.array([0]*n_img)
    else:
        return print('Wrong input')
    
    #Calculate haar feature and break loop if out of bounds
    for idx, image_id in enumerate(os.listdir(path_read)):
        print(idx)
        img = cv2.imread(f'{path_read}/{image_id}', 0)
        img_ii = integral_image(img)
        feature = haar_like_feature(img_ii, 0, 0, 24, 24)
        X[idx] = feature
        if idx + 1 == n_img: 
            break

    #Save to h5.file 
    file = h5py.File(f'{path_write}/{img_type}.hdf5', 'w')
    file.create_dataset('features', data=X)
    file.create_dataset('labels', data=y)
    file.close()
    
def read_hdf5(path_read: str):
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

if __name__ == "__main__":
    main()