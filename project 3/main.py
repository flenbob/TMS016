from skimage.feature import haar_like_feature
from skimage.transform import integral_image
from skimage import color
from skimage import io
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from os import listdir
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    n_f = 162336
    
    #Load haar features for negative and positive images
    X_pos, y_pos = img2haar_features('../FDDB/test_pos', n_f, 5, 1)
    X_neg, y_neg = img2haar_features('../FDDB/test_neg', n_f, 5, 0)
    
    #Concatenate data and split into test and train data
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25,
                                                    random_state=0,
                                                    stratify=y)
    
    #Use forest classifier to reduce the number of features
    clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                             max_features=100, n_jobs=-1, random_state=0)
    
    #Fit model to training data and then score the test data
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    
    
def img2haar_features(path_read: str, n_f: int, n_img: int, img_type: int) -> np.ndarray:
    #Get haar features for each image (1 = positive, 0 = negative)
    
    #X: Haar feature for each image
    #y: label for each image
    X = np.empty((n_img, n_f))
    y = np.array([img_type]*n_img)
    
    #Calculate haar feature and break loop if out of bounds
    for idx, image_id in enumerate(os.listdir(path_read)):
        print(idx)
        img = cv2.imread(f'{path_read}/{image_id}', 0)
        img_ii = integral_image(img)
        feature = haar_like_feature(img_ii, 0, 0, 24, 24)
        X[idx] = feature
        if idx + 1 == n_img: 
            break
        
    return X, y

if __name__ == "__main__":
    main()