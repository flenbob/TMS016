from importlib.resources import path
import logging
from typing import Any
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import confusion_matrix
import math
from sklearn.utils import shuffle
from skimage.feature import haar_like_feature
from PIL import Image, ImageOps
from utils import *
import constants as c
from model import Model

def train(X: np.ndarray, y: np.ndarray, T: int) -> tuple[Model, int]:
    model = None
    while not model:
        #Define classifier and train model
        clf = AdaBoostClassifier(
            DecisionTreeClassifier(criterion='gini', max_depth=1), 
            algorithm="SAMME.R", 
            n_estimators=T
            )
        X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size = 0.75, random_state=42)
        clf.fit(X_train, y_train)
        model = model_reduce(clf, X, y)

        #Get ROC-curve and optimize threshold of strong classifier on trained data
        y_prob = model.clf.predict_proba(X_train[:, model.feats_idx])[:, -1]
        roc = roc_curve(y_train, y_prob)
        thresh, FPR, FNR = optimize_threshold(roc, c.FNR_MAX, c.FPR_MAX)
        model.threshold = thresh
        print(f'Threshold = {thresh} gives FNR = {FNR}, FPR = {FPR}')

        #Control target rates on train subset
        if not (FNR < c.FNR_MAX and FPR < c.FPR_MAX):
            return None

        #Control target rates on validaiton subset
        model = vaildate_model(model, X_validate, y_validate, c.FNR_MAX, c.FPR_MAX)
        if not model:
            print(f'SC with {T} WC:s was not enough. Rerun with {T + math.ceil(c.T_RATE*T)} WC:s.')
            T += math.ceil(c.T_RATE*T)
    
    return model, T

def manage_samples(model: Model, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray]:

    #Delete false negative and true negatives
    X, y = delete_samples(model, X, y)

    #Check current share of negative samples. If they are too low, add FP samples
    N = np.count_nonzero(y==0)/len(y)
    if N < c.N_MIN:
        img = Image.open(c.PATH_NEGATIVE_IMGS)
        img = ImageOps.grayscale(img)
        img_ii_list = add_samples(list(model), img, c.SCALE, c.DELTA)

        #Calculate haar-like features for all FP samples
        X_FP = np.empty(shape=(0, 162336))
        for img_ii in img_ii_list:
            X = haar_like_feature(img_ii, 0, 0, 24, 24)
            X = X.reshape(1, -1)
            X_FP = np.append(X_FP, X, axis=0)
        
        #Add FP samples and labels to dataset and shuffle them
        y_FP = np.array(len(X_FP)*[0])
        X = np.concatenate((X, X_FP))
        y = np.concatenate((y, y_FP))
        X, y = shuffle(X, y)

    #Calculate current rates:
    _, FPR_curr = get_rates(X, y, model)
    return X, y, FPR_curr

def train_model(X: np.ndarray, y: np.ndarray, model_path: str, imgs_path: str) -> None:
    """Trains a cascade of Adaboost classifiers and saves to a .pkl file in model_path

    Args:
        X (np.ndarray): Initial dataset consisting of negative and positive images
        y (np.ndarray): Labels to dataset
        model_path (str): Filepath where .pkl model saved
        imgs_path (str): Folder path to create false positive samples from
    """
    #GETTING OLD

    models_reduced = []
    last_visited = None
    FPR_curr = 1
    t = c.T_INIT
    with open(model_path, 'wb') as f:
        while FPR_curr > c.FPR_TERMINATE: #While we've not cleared enough false positives, keep building layers
            print(f'Current FPR: {FPR_curr}. A total of {len(y)} samples, with {np.count_nonzero(y==0)} negative, and {np.count_nonzero(y==1)} positive.')

            #Define classifier and train model
            clf = AdaBoostClassifier(
                DecisionTreeClassifier(criterion='gini', max_depth=1), 
                algorithm="SAMME.R", 
                n_estimators=t
                )
            X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size = 0.75, random_state=42)
            clf.fit(X_train, y_train)
            model = model_reduce(clf, X, y)

            #Get ROC-curve and optimize threshold of strong classifier on trained data
            y_prob = model.clf.predict_proba(X_train[:, model.feats_idx])[:, -1]
            roc = roc_curve(y_train, y_prob)
            thresh, FPR, FNR = optimize_threshold(roc, c.FNR_MAX, c.FPR_MAX)
            model.threshold = thresh
            print(f'Threshold = {thresh} gives FNR = {FNR}, FPR = {FPR}')

            #Control target rates on train subset
            if not (FNR < c.FNR_MAX and FPR < c.FPR_MAX):
                return None

            #Control target rates on validaiton subset
            model = vaildate_model(model, X_validate, y_validate, c.FNR_MAX, c.FPR_MAX)
            if not model:
                print(f'SC with {t} WC:s was not enough. Rerun with {t + math.ceil(c.T_RATE*t)} WC:s.')
                t += math.ceil(c.T_RATE*t)
                continue

            #Todo: 
            #Add function: 'add_samples'
    


            #If FPR is too low, add more from negative dataset
            _, FP, _, TP = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
            if c.FPR_MIN > FP/(FP+TP) and last_visited != 'EOF':
                X_FP, last_visited = get_FP_samples(models_reduced, c.min_FPR*FP/(FP+TP), imgs_path, last_visited, c.SCALE, c.DELTA)                
                y_FP = np.array(len(X_FP)*[0])
                
                #Add FP samples and labels to dataset and shuffle them
                X = np.concatenate((X, X_FP))
                y = np.concatenate((y, y_FP))
                X, y = shuffle(X, y)
                print(f'FP/TP ratio = {FP/TP} was too low. Added {len(X_FP)} FP samples. Last visited image: {last_visited}')

            elif c.FPR_MIN > FP/(FP+TP) and last_visited == 'EOF':
                print(f'No more FP samples to add')

            # y_pred = 1*(clf_best.predict_proba(X[:, feature_idx])[:,-1] >= thresh)
            # _, FP, _, TP = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
            # FPR_curr = FP/(TP+FP)
            # print(f'Current false positive rate is: {FPR_curr}')
            # #Append SC and threshold to .pkl file
            # print(f'Successfully generated layer: {len(models_reduced)}. SC with {t} WC:s')
            # print('----------------------')
            
            #Save SC to file
            pickle.dump(models_reduced, f)

def run():
    pass

if __name__ == "__main__":

    #READ A DATAFILE CONSISTING OF POSITIVE IMAGES
    #READ A DATAFILE CONSISTING OF NEGATIVE IMAGES
    X = 1
    y = 1
    
    #GENERATE MODEL FOR EACH LAYER
    T = c.T_INIT
    FPR_curr = 1

    with open(c.PATH_MODEL, 'wb') as f:
        while FPR_curr > c.FPR_TERMINATE:
            model, T = train(X, y, T)
            pickle.dump(model, f)
            X, y, FPR_curr = manage_samples(model, X, y)
