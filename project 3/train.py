from typing import Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
import h5py
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import math
from sklearn.utils import shuffle
from classify import cascade_classifier
from classify import read_model_file

def main():
    #File path for negative dataset to supply FP:s, and the negative + positive imgs at start.
    file_path_neg = '../negative_imgs_datasets.hdf5'
    X_pos, y_pos = read_feature_file('../positive_imgs.hdf5')
    X_neg, y_neg = read_feature_file('../negative_imgs.hdf5')
    X_pos = X_pos[0:600]
    y_pos = y_pos[0:600]
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    X, y = shuffle(X, y)
    print('files loaded')
    print(f'dataset size: {len(y)}, with {len(y_pos)} positive and {len(y_neg)} negative')

    #Define parameters for training
    FNR_target = 1
    FPR_target = 1
    file_path = 'abclf_2.pkl'
    n_splits = 4
    
    #Train
    train_adaboost_classifiers(FNR_target, FPR_target, X, y, file_path, file_path_neg, n_splits)

def train_adaboost_classifiers(FNR_target: float, FPR_target: float, X: np.ndarray, \
                               y: np.ndarray, file_path: str, file_path_neg: str, n_splits: int) -> None:
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
    t = 8 #Initial amount of WC:s in first SC
    t_rate = 0.41 #Constant which controls amount of WC:s to add to SC if it fails
    last_visited = None #Initialize last visited dataset in negative dataset
    min_FP_ratio = 0.75 #min FP/TP share required for training of each SC
    FPR_target = 0.65 #Maximum false positive rate for each SC
    FNR_target = 0.02 #Maximum false negative rate for each SC
    curr_FP = 1
    with open(file_path, 'wb') as f:
        while curr_FP > 0.03: #While we've not cleared enough false positives, keep building layers
            print(f'Current FPR: {curr_FP}. A total of {len(y)} samples, with {np.count_nonzero(y==0)} negative, and {np.count_nonzero(y==1)} positive.')
            #SC and WC:s
            clf = AdaBoostClassifier(
                DecisionTreeClassifier(criterion='gini', max_depth=1), 
                algorithm="SAMME.R", 
                n_estimators=t
            )
            print(f'Training SC with {t} WC:s....')
            clf_best, thresh, flag = cross_validate(clf, X, y, FNR_target, FPR_target, n_splits)
            if not flag:
                #If we do not achieve rates, add WC:s to SC and train new model
                print(f'SC with {t} WC:s was not enough. Rerun with {t + math.ceil(t_rate*t)} WC:s.')
                t += math.ceil(t_rate*t)
                continue
            
            #Else SC is accepted, append to list of models
            models.append((clf_best, thresh))
            
            #Run SC on dataset and delete TN and FN
            y_pred = 1*(clf_best.predict_proba(X)[:,-1] >= thresh)
            TN, FP, FN, TP = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
            FNR = FN/(FN+TP)
            FPR = FP/(FP+TN)
            print(f'FNR: {FNR}, FPR: {FPR}')

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
            if min_FP_ratio > FP/TP and last_visited != 'EOF':
                X_FP, last_visited = generate_FP_samples(min_FP_ratio*TP, models, file_path_neg, last_visited)                
                y_FP = np.array(len(X_FP)*[0])
                
                #Add FP samples and labels to dataset and shuffle them
                X = np.concatenate((X, X_FP))
                y = np.concatenate((y, y_FP))
                X, y = shuffle(X, y)
                print(f'FP/TP ratio = {FP/TP} was too low. Added {len(X_FP)} FP samples. Last visited dataset: {last_visited}')

            elif min_FP_ratio > FP/TP and last_visited == 'EOF':
                print(f'No more FP samples to add from file.')

            y_pred = 1*(clf_best.predict_proba(X)[:,-1] >= thresh)
            TN, FP, FN, TP = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
            curr_FP = FP/TP
            print(f'Current false positive rate is: {curr_FP}')
            #Append SC and threshold to .pkl file
            print(f'Successfully generated layer: {len(models)}. SC with {t} WC:s')
            print('----------------------')
            
            #Save SC to file
            pickle.dump([clf_best, thresh], f)
            
def generate_FP_samples(min_FP: int, models: list, file_path: str, last_visited: str) -> np.ndarray:
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
        print(datasets)
    
    #Iterate through datasets in file
    for dataset in datasets:
        print(f'Reading dataset {dataset}')
        X = file[dataset]
        
        #Run cascade classifier:
        for model in models:
            clf = model[0]
            thresh = model[1]
            
            #Predict and delete TN
            y_pred = 1*(clf.predict_proba(X)[:,1] >= thresh)
            TN_idx = np.where(y_pred == 0)[0]
            X = np.delete(X, TN_idx, axis=0)
        
        #Add FP:s to set
        print(f'Got {len(X)} FP:s from dataset with size {500}')
        X_FP = np.append(X_FP, X, axis=0)
        if len(X_FP) >= min_FP:
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

        #Tune hyperparameter FPR and FNR of SC and then validate
        y_prob = clf.predict_proba(X_validate)[:, 1]
        FPR_list, TPR_list, threshold_list = roc_curve(y_validate, y_prob)
        FNR_list = 1 - TPR_list
        thresh, FPR, FNR = optimize_threshold(FNR_list, FPR_list, threshold_list, FNR_target)
        print(f'Threshold = {thresh} gives FNR = {FNR}, FPR = {FPR}')

        #If feasible and best performing, save model and threshold
        if FPR < FPR_target and FNR < FNR_target and FPR < FPR_best and FNR < FNR_best:
            clf_best = clf
            thresh_best = thresh
        
        FNR_fold.append(FNR)
        FPR_fold.append(FPR)

    #If the mean performance of our models is worse than target, then we need more WC:s
    if sum(FNR_fold)/len(FNR_fold) > FNR_target or sum(FPR_fold)/len(FPR_fold) > FPR_target:
        flag = False
        clf_best = None
        thresh_best = None
        return clf_best, thresh_best, flag

    #Run best model and its threshold on test data and check for overfitting
    y_pred = 1*(clf_best.predict_proba(X_test)[:, -1] >= thresh_best)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
    FNR = FN/(FN+TP)
    FPR = FP/(FP+TN)
    if math.isnan(FNR):
        FNR = 0
    if math.isnan(FPR):
        FPR = 0
    print(f'VALIDATION SET: Threshold: {thresh_best} gives: FNR = {FNR}, FPR = {FPR}')

    #Return
    flag = True
    return clf_best, thresh_best, flag
        
def optimize_threshold(FNR_list: np.ndarray, FPR_list: np.ndarray, threshold_list: np.ndarray, target_FNR: float) -> list:
    #Find threshold that satisfies target false negative rate (FNR)
    idx = 0
    FNR = FNR_list[0]
    while FNR >= target_FNR:
        idx += 1
        FNR = FNR_list[idx]
    
    print(f'FNR LIST: {FNR_list}')
    print(f'Threshold list: {threshold_list}')
    FPR = FPR_list[idx]
    thresh = threshold_list[idx]-(1e-3)
    return thresh, FPR, FNR 
   
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

if __name__ == "__main__":
    main()