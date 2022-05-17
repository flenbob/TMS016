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
import dataclasses

def main():
    n_f = 162336 # number of features for a 24*24 pixel image
    n_pos = 1
    n_neg = 1

    #Load haar features for negative and positive images
    pos_df = img2haar_features2('../data/test_pos', n_f, n_pos, 1)
    neg_df = img2haar_features2('../data/test_neg', n_f, n_neg, 0)

    print(pos_df.info)

    
    #Concatenate data and split into test and train data
    #X = np.concatenate((X_pos, X_neg))
    #y = np.concatenate((y_pos, y_neg))
    # print(len(X))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25,
    #                                                random_state=0,
    #                                                stratify=y)
    
    
    
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

        print(feature)
        print("size of feature: " + str(len(feature)))
        X[idx] = feature
        if idx + 1 == n_img: 
            break
    return X, y

def img2haar_features2(path_read: str, n_f: int, n_img: int, img_type: int) -> pd.DataFrame:
    #Get haar features for each image (1 = positive, 0 = negative)
    
    column_names = ["img_id", "class", "features", "weight"]
    img_df = pd.DataFrame(columns= column_names)
    
    #Calculate haar feature and break loop if out of bounds
    for idx, image_id in enumerate(os.listdir(path_read)):
        print(idx)
        img = cv2.imread(f'{path_read}/{image_id}', 0)
        img_ii = integral_image(img)
        feature = haar_like_feature(img_ii, 0, 0, 24, 24) 

        new_row = pd.DataFrame({'img_id': image_id, 'class': img_type, 'features': feature.tolist(), 'weight': 1/(2*n_img)})
        img_df = pd.concat([img_df, new_row])
        #print(feature)
        #print("size of feature: " + str(len(feature)))

        if idx + 1 == n_img: 
            break
    return img_df


def start_weights(path_read: str, n_img: int, img_type: int) -> np.ndarray:
    ws = np.full((n_img, 1), 1/(2*n_img))
    return ws

def normalize_weights(ws: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(ws)
    return ws / norm

""""
# T: number of boosting iterations = weak classifiers to be produced
# w: weights of the images
# f: all possible features
# theta: thresholds for each feature
# p: parity for each feature
def boost(T: int, ws: np.ndarray, img_type: int, fs: np.ndarray, thetas: np.ndarray, ps: np.ndarray) -> np.ndarray:

    for t in range(T):
        ws = normalize_weights(ws)
        for f in fs: # iterate through every feature
            err, theta, p = train_WC(ws: np.ndarray, img_type: int, ) FIXFIXFIXFIXFIXFIXFIXFIXFIX

    return 
"""
# weak classifier
def WC(f: int, theta: float, p: int) -> int:
    if p*f < p*theta:
        return 1
    else:
        return 0



if __name__ == "__main__":
    main()