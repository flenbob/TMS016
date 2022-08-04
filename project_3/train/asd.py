import numpy as np
import os
from skimage.feature import haar_like_feature
from multiprocessing import Pool
from skimage.feature import haar_like_feature, haar_like_feature_coord
from skimage.transform import integral_image
from time import perf_counter
from functions import *
from sklearn.metrics import confusion_matrix, roc_curve
import timeit
import matplotlib.pyplot as plt
import cv2
import itertools
import constants as c


def main():
    clf = AdaBoostClassifier(
        DecisionTreeClassifier(criterion='gini', max_depth=1), 
        algorithm="SAMME.R", 
        n_estimators=4
    )

    X_pos = samples_load(c.POS_SAMPLES_PATH)[0:200]
    X_neg = samples_load(c.NEG_SAMPLES_PATH)[0:200]
    y_pos = np.array(len(X_pos)*[1])
    y_neg = np.array(len(X_neg)*[0])
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    print('Files loaded')

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size = 0.75, random_state=42)
    clf.fit(X_train, y_train)

    #TRAIN
    y_prob = clf.predict_proba(X_train)[:, -1]
    fpr_list, tpr_list, thresh_list_train = roc_curve(y_train, y_prob)
    fnr_list = 1 - tpr_list

    idx_train = [i for i, (fnr, fpr) in enumerate(zip(fnr_list, fpr_list)) if not (fpr > c.FPR_MAX or fnr > c.FNR_MAX)]
    thresh_train = thresh_list_train[max(min(idx_train)-1, 0): min(max(idx_train)+1, len(thresh_list_train))]
    thresh_train[0] -= 2e-2
    thresh_train[-1] += 2e-2

    #VALIDATE
    y_prob = clf.predict_proba(X_validate)[:, -1]
    fpr_list, tpr_list, thresh_list_validate = roc_curve(y_validate, y_prob)
    fnr_list = 1 - tpr_list

    idx_validate = [i for i, (fnr, fpr) in enumerate(zip(fnr_list, fpr_list)) if not (fpr > c.FPR_MAX or fnr > c.FNR_MAX)]
    thresh_validate = thresh_list_validate[max(min(idx_validate)-1, 0): min(max(idx_validate)+1, len(thresh_list_validate))]
    thresh_validate[0] -= 2e-2
    thresh_validate[-1] += 2e-2

    if max(min(thresh_validate), min(thresh_train)) <= min(max(thresh_validate), max(thresh_train)): # max(thresh_min_validate, thresh_min_train) <= min(thresh_max_validate, thresh_max_train):
        thresh_min = max(min(thresh_validate), min(thresh_train))
        thresh_max = min(max(thresh_validate), max(thresh_train))

        #Select all threshold values which are feasible:
        thresh_vals_train = [thresh for thresh in thresh_train if (thresh_max > thresh > thresh_min)]
        thresh_vals_validate = [thresh for thresh in thresh_validate if (thresh_max > thresh > thresh_min)]
        thresh_vals = thresh_vals_train + thresh_vals_validate
        
        best_error = float('inf')
        for thresh in thresh_vals:
            y_pred = 1*(clf.predict_proba(X_validate)[:, -1] >= thresh)
            tn, fp, fn, tp = confusion_matrix(y_validate, y_pred, labels=[0, 1]).ravel()
            fnr_validate, fpr_validate = fn/(fn+tp), fp/(fp+tn)
            print('-------')
            print(f'Valid: FNR = {fnr_validate:.3f}, FPR = {fpr_validate:.3f}')
        
            y_pred = 1*(clf.predict_proba(X_train)[:, -1] >= thresh)
            tn, fp, fn, tp = confusion_matrix(y_train, y_pred, labels=[0, 1]).ravel()
            fnr_train, fpr_train = fn/(fn+tp), fp/(fp+tn)
            print(f'Train: FNR = {fnr_train:.3f}, FPR = {fpr_train:.3f}')

            if fnr_train + fpr_train + fnr_validate + fpr_validate < best_error:
                best_error = fnr_train + fpr_train + fnr_validate + fpr_validate
                best_thresh = thresh
    else:
        print('Infeasible solution, intervals dont overlap')

    print('BEST SOLUTION:')
    y_pred = 1*(clf.predict_proba(X_validate)[:, -1] >= best_thresh)
    tn, fp, fn, tp = confusion_matrix(y_validate, y_pred, labels=[0, 1]).ravel()
    fnr_validate, fpr_validate = fn/(fn+tp), fp/(fp+tn)
    print(f'Valid: FNR = {fnr_validate:.3f}, FPR = {fpr_validate:.3f}')

    y_pred = 1*(clf.predict_proba(X_train)[:, -1] >= best_thresh)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred, labels=[0, 1]).ravel()
    fnr_train, fpr_train = fn/(fn+tp), fp/(fp+tn)
    print(f'Train: FNR = {fnr_train:.3f}, FPR = {fpr_train:.3f}')

def test():
    clf = AdaBoostClassifier(
        DecisionTreeClassifier(criterion='gini', max_depth=1), 
        algorithm="SAMME.R", 
        n_estimators=3
    )

    X_pos = samples_load(c.POS_SAMPLES_PATH)[0:200]
    X_neg = samples_load(c.NEG_SAMPLES_PATH)[0:200]
    y_pos = np.array(len(X_pos)*[1])
    y_neg = np.array(len(X_neg)*[0])
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    print('Files loaded')

    clf.fit(X, y)

    y_prob = clf.predict_proba(X)[:,-1]
    fpr_list, tpr_list, thresh_list = roc_curve(y, y_prob)
    fnr_list = 1 - tpr_list

    print(f'thresh = {thresh_list}\nfnr = {fnr_list}\nfpr = {fpr_list}')

    y_pred = 1*(clf.predict_proba(X)[:,-1] >= (thresh_list[4]-1e-4))
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    fnr, fpr = fn/(fn+tp), fp/(fp+tn)
    print(f'FNR = {fnr:.3f}, FPR = {fpr:.3f}')


def read_model_file(file_path: str) -> list:
    models = []
    with open(file_path, 'rb') as f:
        try:
            while True:
                model = pickle.load(f)
                #print(f'{model}')
                models.append(model)
        except (EOFError):
            pass
    return models

def testa():
    starttime = timeit.default_timer()
    imgs_path = os.listdir('../negative_imgs')
    imgs_path = [f'{c.NEG_DSET_PATH}/{img}' for img in imgs_path]
    imgs_path = imgs_path[0:8]

    images = []
    for img in imgs_path:
        im = Image.open(img)
        im = ImageOps.grayscale(im)
        images.append(im)

    n_add = 100
    models = read_model_file('project_3/train/ABCLF.pkl')
    partial_args = functools.partial(image_scan, models, n_add)
    with concurrent.futures.ProcessPoolExecutor() as pool:
        sample_splits = list(pool.map(partial_args, images))
    
    for sample_split in sample_splits:
        print(len(sample_split))

    samples = []
    samples = list(itertools.chain.from_iterable(sample_splits))
    print("The time difference is :", timeit.default_timer() - starttime)

def tet():
    a = []
    a += [1,2,3]
    print(a)

if __name__ == "__main__":

    tet()