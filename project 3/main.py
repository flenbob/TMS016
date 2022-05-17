from skimage.feature import haar_like_feature
from skimage.transform import integral_image
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import sys
from time import time
import h5py
import pandas as pd
import pickle
from sklearn.metrics import plot_confusion_matrix

def main():
    n_f = 162336
    n_img = 500
    with open('adaboost_classifier.pkl', 'rb') as f:
        bdt = pickle.load(f)
        
    #print(bdt.feature_importances_)
        
    #idx_sorted = np.argsort(bdt.feature_importances_)[::-1]
    print(np.sum(bdt.feature_importances_))
    #print(bdt.feature_importances_[idx_sorted][1:200])
    #print(idx_sorted[1:200])
    
    # X_pos, y_pos = read_hdf5('data', 'positive')
    # X_neg, y_neg = read_hdf5('data', 'negative')
    # X = np.concatenate((X_pos, X_neg))
    # y = np.concatenate((y_pos, y_neg))
    # y_pred = bdt.predict(X)
    # plot_confusion_matrix(bdt, X, y)
    # plt.show()
    
    # #Load haar features for negative and positive images
    # img2haar_features('../pos_imgs_test_2', 'data', n_f, 1000, 'positive')
    # img2haar_features('../neg_imgs_test_2', 'data', n_f, 1000, 'negative')
    # X_pos, y_pos = read_hdf5('data', 'positive')
    # X_neg, y_neg = read_hdf5('data', 'negative')
    
    # #Concatenate data and split into test and train data
    # X = np.concatenate((X_pos, X_neg))
    # y = np.concatenate((y_pos, y_neg))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25,
    #                                                 random_state=0,
    #                                                 stratify=y)
    #print(bdt.score(X_test, y_test))
    # print('saving classifier to file')
    # with open('adaboost_classifier.pkl', 'wb') as f:
    #     pickle.dump(bdt, f)
    
    # bdt.fit(X_train, y_train)
    #print(bdt.score(X, y))
    # idx_sorted = np.argsort(bdt.feature_importances_)[::-1]
    # idx_sorted = idx_sorted[1:100]
    # feature_imps = pd.Series(bdt.feature_importances_[idx_sorted], index=idx_sorted)
    # fig, ax = plt.subplots()
    # feature_imps.plot.bar(ax=ax)
    # ax.set_title("Feature importances using MDI")
    # ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    # print('before plot show')
    # print(bdt.feature_importances_[0])
    # plt.show()
    
def train_adaboost_classifier(X_train, y_train, n_estimators):
    #Train adaboost classifier with n_estimators decision stumps
    bdt = AdaBoostClassifier(
        DecisionTreeClassifier(criterion='gini', max_depth=1), 
        algorithm="SAMME", 
        n_estimators=n_estimators
    )
    t_start = time.time()
    bdt.fit(X_train, y_train)
    print(f'Time to fit model: {time.time() - t_start}s')
    return bdt  

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
    
def read_hdf5(path_read: str, img_type: str):
    #Read feature hdf5 file and return:
    #X: features for each image
    #y: labels for each image 
    file = h5py.File(f'{path_read}/{img_type}.hdf5', 'r')
    X = file['features'] 
    y = file['labels']
    X = np.array(X)
    y = np.array(y)
    file.close()
    return X, y

if __name__ == "__main__":
    main()