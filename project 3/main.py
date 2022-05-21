from typing import Any
from skimage.feature import haar_like_feature
from skimage.transform import integral_image
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
import h5py
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold,StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
import math

def main():

    #img2haar_features('../neg_imgs', 'data', 162336, 1430, 'negative')
    
    X_pos, y_pos = read_feature_file('data/positive_test.hdf5')
    X_neg, y_neg = read_feature_file('data/negative_test.hdf5')
    
    X_pos = X_pos
    y_pos = y_pos
    
    #X_neg = X_neg[751:1000]
    #y_neg = y_neg[751:1000]
    
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    print('files loaded')
    
    FNR_target = 0.01
    FPR_target = 0.4
    file_path = 'abclf_test.pkl'
    n_splits = 4
    
    train_adaboost_classifiers(FNR_target, FPR_target, X, y, file_path, n_splits)
      
def cascade_classifier(img: np.ndarray, models: list) -> int:
    # Input:
    # features: Array of features for an image
    # models: List of models for each layer in the cascade
    
    # Returns:
    # 1 if all layers predicts over the set threshold
    # 0 otherwise
    for model in models:
        #Load strong classifier and its threshold
        img = img.reshape(1,-1)
        SC = model[0]
        threshold = model[1]
        print(SC.predict_proba(img))
        y_pred = 1*(SC.predict_proba(img)[:,-1] >= threshold)
        if y_pred == 0:
            return 0
    return 1

def train_adaboost_classifiers(FNR_target: float, FPR_target: float, X: np.ndarray, \
                               y: np.ndarray, file_path: str, n_splits: int) -> None:
    # Input:
    # FNR_thresh: Maximum False Negative Ratio threshold [0, 1]
    # T: List of number of weak classifiers (WC) for each layer of strong classifier (SC)
    # X: Matrix of haar features for each sample (image)
    # y: array of correct binary classification
    # file_path: Filepath including filename (ends with .pkl) (eg. folder1/folder2/file.pkl)
    
    # Returns:
    # None, but saves all SC:s and their thresholds to .pkl file
    
    #Set initial value to start while-loop
    models = [] #Initialize empty array of (classifier, threshold)
    t = 1 #Initial amount of WC:s in first SC
    t_rate = 0.41 #Constant which controls amount of WC:s to add to SC if it fails
    score = 0
    score_thresh = 0.98 #Score threshold for termination
    last_visited = None #Initialize last visited dataset in negative dataset
    min_FP_ratio = 0.75 #min FP/TP share required for training of each SC
    FPR_target = 0.4 #Target false positive rate for each SC
    FNR_target = 0.01 #Target false negative rate for each SC
    with open(file_path, 'wb') as f:
        while True:
                #SC and WC:s
                clf = AdaBoostClassifier(
                    DecisionTreeClassifier(criterion='gini', max_depth=1), 
                    algorithm="SAMME.R", 
                    n_estimators=t
                )
                print(f'Training SC with {t} WC:s....')
                clf, thresh, flag = cross_validate(clf, X, y, FNR_target, FPR_target, n_splits)
                if not flag:
                    #If we do not achieve rates, add WC:s to SC and train new model
                    print(f'SC with {t} WC:s was not enough. Rerun with {t + math.ceil(t_rate)} WC:s.')
                    t += math.ceil(t_rate)
                    continue
                
                #Else SC is accepted, append to list of models
                models.append((clf, thresh))
                
                #Run SC on dataset and delete TN and FN
                y_pred = 1*(clf.predict_proba(X)[:,-1] >= thresh)
                delete_idx = []
                for i in range(len(y_pred)):
                    if ((y[i] == 0) and (y_pred[i] == 0)) or \
                        ((y[i] == 1) and (y_pred[i] == 0)):
                        delete_idx.append(i)

                print(f'Deleted {round(100*len(delete_idx)/len(y_pred), 2)}% of samples')
                X = np.delete(X, delete_idx, axis=0)
                y = np.delete(y, delete_idx)
                y_pred = np.delete(y_pred, delete_idx)
        
                #If the ratio of FP is too low, add more from negative dataset
                _, FP, _, TP = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
                print(f'FP/TP = {FP/TP}')
                if min_FP_ratio > FP/TP:
                    X_FP, last_visited = generate_FP_samples(min_FP_ratio*TP, models, file_path, last_visited)                
                    y_FP = np.array(len(X_FP)*[0])
                    
                    #Add FP samples and labels to dataset
                    X = np.concatenate((X, X_FP))
                    y = np.concatenate((y, y_FP))
                    print(f'FP/TP ratio = {FP/TP} was too low. Added {len(X_FP)} new negative samples.')
                
                #Append SC and threshold to .pkl file
                print(f'Successfully generated layer: {len(models)}. SC with {t} WC:s')
                print('----------------------')
                
                pickle.dump([clf, thresh], f)

def generate_FP_samples(n_samples_req: int, models: list, file_path: str, last_visited: str) -> np.ndarray:
    n_f = 162336
    X_FP = np.empty(shape=(0, n_f))
    try:
        file = h5py.File(file_path, 'r')
    except:
        print(f'Could not read negaive db file {file_path}')
        return X_FP, 'empty'
    
    #Slice list from the dataset we last visited
    datasets = list(file.keys())
    if last_visited != None:
        datasets = datasets[datasets.index(last_visited)+1:-1]
    
    #Iterate through datasets in file
    for dataset in datasets:
        X = file[dataset]
        
        #Run cascade classifier:
        for model in models:
            clf = model[0]
            thresh = model[1]
            
            #Predict and delete true negatives
            y_pred = 1*(clf.predict_proba(X)[:,1] >= thresh)
            TN_idx = np.where(y_pred == 0)[0]
            X = np.delete(X, TN_idx, axis=0)
        
        #Add FP:s to set
        X_FP = np.concatenate(X_FP, X)
        if len(X_FP) > n_samples_req:
            #Satisfied required amount, return samples and last visited dataset
            file.close()
            return X_FP, dataset
        
    #If required amount cant be supplied, we've reached end of file. Send what's left
    file.close()
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
    # clf_best: Best SC
    # thresh: threshold for SC
    # flag: True if SC is accepted, False otherwise.
    skf = StratifiedKFold(n_splits=n_splits)
    FNR_fold = []
    FPR_fold = []
    d_min = 3 #Initial minimum distance to target rate
    thresh = 0.5 #Initial threshold for SC
    
    #Divide dataset into Kfold (train, test) and validation (train, test)
    i = 0
    X_KFold, X_validate, y_KFold, y_validate, = train_test_split(X, y, train_size=0.75, shuffle=True)
    for train_idx, test_idx in skf.split(X_KFold, y_KFold):
        i += 1
        print(f'Fold {i}')
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        #Train SC
        clf.fit(X_train, y_train)
        
        #FPR and FNR of SC
        TN, FP, FN, TP = confusion_matrix(y_test, clf.predict(X_test), labels=[0,1]).ravel()
        print(f'TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}')

        FPR = FP/(FP+TN)
        FNR = FN/(FN+TP)
        if math.isnan(FNR):
            FNR = 0
        if math.isnan(FPR):
            FPR = 0
        FNR_fold.append(FNR)
        FPR_fold.append(FPR)
        
        #Distance to target rates
        d = (FNR - FNR_target) + (FPR - FPR_target)
        print(f'Distance: {d} vs min dist {d_min}')
        
        #If FNR and FPR is best: Save current SC as best SC
        if d_min > d:
            d_min = d
            clf_best = clf
    
    #If mean of FNR and FPR from KFold are worse than target rates:
    FNR_fold = np.array(FNR_fold)
    FPR_fold = np.array(FPR_fold)
    if (np.mean(FNR_fold) > FNR_target) or (np.mean(FPR_fold) > FPR_target):
        y_prob = clf.predict_proba(X)[:, -1]
        FPR_list, TPR_list, threshold_list = roc_curve(y, y_prob)
        FNR_list = 1 - TPR_list
        thresh = optimize_threshold(FNR_list, FPR_list, threshold_list, FNR_target)
        
        #Use threshold to calculate FPR:
        y_pred = 1*(clf_best.predict_proba(X_validate)[:,-1] >= thresh)
        TN, FP, FN, TP = confusion_matrix(y_validate, y_pred, labels=[0,1]).ravel()
        FPR = FP/(FP+TN)
        FNR = FN/(FN+TP)
        if math.isnan(FPR):
            FPR = 0
        if math.isnan(FNR):
            FNR = 0
                    
        if FPR > FPR_target:
            #Infeasible solution, does not achieve target rates.
            #Stop cross validation and add WC to SC instead
            flag = False
            return clf_best, thresh, flag
        
    #Validate SC on validation set
    y_pred = 1*(clf_best.predict_proba(X_validate)[:,-1] >= thresh)
    TN, FP, FN, TP = confusion_matrix(y_validate, y_pred, labels=[0,1]).ravel()
    FNR = FN/(FN+TP)
    FPR = FP/(FP+TN)
    if math.isnan(FNR):
        FNR = 0
    if math.isnan(FPR):
        FPR = 0
    
    if FPR > FPR_target and FNR > FNR_target:
        #Infeasible solution. Validation does not achieve target rates.
        #Stop cross validation and add WC to SC instead
        
        flag = False
        return clf_best, thresh, flag
    
    #If we pass all the tests, the SC is accepted
    flag = True
    return clf_best, thresh, flag
        
def optimize_threshold(FNR_list: np.ndarray, FPR_list: np.ndarray, threshold_list: np.ndarray, target_FNR: float) -> float:
    #Find threshold that satisfies target false negative rate (FNR)
    idx = 0
    FNR = FNR_list[0]
    while FNR >= target_FNR:
        idx += 1
        FNR = FNR_list[idx]
    
    #Linearly approximate target threshold
    left_FNR = FNR_list[idx-1]
    right_FNR = FNR_list[idx]
    left_threshold = threshold_list[idx-1]
    right_threshold = threshold_list[idx]
    
    ratio = (left_FNR-target_FNR)/(left_FNR-right_FNR)
    target_thresh = left_threshold-(ratio*(left_threshold-right_threshold))
    
    # if plot:
    #     #Plot threshold and acquired FPR for that target threshold
    #     dfplot = pd.DataFrame({'Threshold':threshold_list, 
    #         'False Positive Rate':FPR_list, 
    #         'False Negative Rate': FNR_list})

    #     ax=dfplot.plot(x='Threshold', y=['False Positive Rate',
    #     'False Negative Rate'], figsize=(10,6))
    #     ax.plot([target_thresh, target_thresh], [0,1]) #mark selected thresh
    #     plt.show()
        
    return target_thresh

def read_model_file(file_path:str) -> list:
    models = []
    with open(file_path, 'rb') as f:
        try:
            while True:
                model = pickle.load(f)
                print(f'{model}')
                models.append(model)
        except (EOFError):
            pass
    return models   
   
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

def img2haar_features(path_read: str, path_write: str, n_f: int, n_img: int, img_type: str) -> None:
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
    file = h5py.File(f'{path_write}/{img_type}_school.hdf5', 'w')
    file.create_dataset('features', data=X)
    file.create_dataset('labels', data=y)
    file.close()

if __name__ == "__main__":
    main()