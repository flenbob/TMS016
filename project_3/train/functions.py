import constants as c
from model import Model
import pickle
import numpy as np
import os
import math
import concurrent.futures
import functools
import itertools
from PIL import Image, ImageOps
from skimage.feature import haar_like_feature, haar_like_feature_coord
from skimage.transform import integral_image
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split


#MODEL RELATED FUNCTIONS
def model_fit(X: np.ndarray, y: np.ndarray, n_wc: int) -> tuple[Model, int]:
    """Fits a model consisting of an Adaboost classifier with weak classifiers (decision stumps).

    Args:
        X (np.ndarray): Sample features
        y (np.ndarray): Sample labels
        n_wc (int): Number of weak classifiers

    Returns:
        Model: Trained model and number of weak classifiers used
    """
    model = None
    while not model:
        #Split datasets
        X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size = 0.75, random_state=42)
        
        #Train classifier and place in model object for simpler handling
        clf = AdaBoostClassifier(
            DecisionTreeClassifier(criterion='gini', max_depth=1), 
            algorithm="SAMME.R", 
            n_estimators=n_wc
            )
        clf.fit(X_train, y_train) 
        model = model_reduce(clf, X_train, y_train)

        #Optimize threshold of model that fulfills error rates for both datasets
        model = model_threshold(model, X_train, X_validate, y_train, y_validate)
        if not model:
            print(f'Train: Model with {n_wc} WC:s was not enough. Rerun with {n_wc + math.ceil(c.WC_RATE*n_wc)} WC:s.')
            n_wc += math.ceil(c.WC_RATE*n_wc)
            continue
    return model, n_wc

def model_reduce(clf: AdaBoostClassifier, X: np.ndarray, y: np.ndarray) -> Model:
    """Converts classifier to a model which uses a subset of the most important features.

    Args:
        clf (AdaBoostClassifier): Strong classifier
        X (np.ndarray): Sample features
        y (np.ndarray): Sample labels

    Returns:
        Model: Model object with threshold and haar-like features of interest.
    """

    n_estimators = clf.n_estimators
    feature_coord, feature_type = haar_like_feature_coord(24, 24)
    feature_coord_SC = np.empty(n_estimators, dtype=object)
    feature_type_SC = np.empty(n_estimators, dtype=object)
    feature_indicies = []
    
    #Iterate through weak classifiers in the strong classifier
    for j, wc in enumerate(clf):
        #Feature index
        feature_idx = wc.tree_.feature[0]
        feature_indicies.append(feature_idx)

        #Feature type
        feat_t = np.empty(len([feature_type[feature_idx]]), dtype=object)
        feat_t[:] = feature_type[feature_idx]
        feature_type_SC[j] = feat_t[0]

        #Feature coordinate
        feat_c = np.empty(len([feature_coord[feature_idx]]), dtype=object)
        feat_c[:] = [feature_coord[feature_idx]]
        feature_coord_SC[j] = feat_c[0]
    
    #Train new model on selected feature indicies with same threshold:
    clf_reduced = AdaBoostClassifier(
                DecisionTreeClassifier(criterion='gini', max_depth=1), 
                algorithm="SAMME.R", 
                n_estimators=n_estimators
        )
    clf_reduced.fit(X[:, feature_indicies], y)
    
    #Store as a Model object
    model_reduced = Model(clf_reduced, 0.5, feature_type_SC, feature_coord_SC, feature_indicies)
    return model_reduced

def model_threshold(model: Model, X_train: np.ndarray, X_validate: np.ndarray, y_train: np.ndarray, y_validate: np.ndarray) -> Model:
    """Finds threshold for which yields lowest sum of false negative and false positive rates but is within constraints.

    Args:
        model (Model): Trained model
        X_train (np.ndarray): Training sample features
        X_validate (np.ndarray): Validation sample features
        y_train (np.ndarray): Training sample labels
        y_validate (np.ndarray): Validation sample labels

    Returns:
        Model: Trained model with set threshold
    """
    #Get feasible thresholds from ROC-curve
    thresh_train = roc_thresholds(model, X_train, y_train)
    thresh_validate = roc_thresholds(model, X_validate, y_validate)
    if len(thresh_validate) == 0 or len(thresh_train) == 0:
        return None 

    #Check if there exists overlapping interval of feasible thresholds between both datasets
    thresh_min = max(min(thresh_validate), min(thresh_train))
    thresh_max = min(max(thresh_validate), max(thresh_train))
    if thresh_min <= thresh_max:
        #Select all feasible thresholds for both datasets
        thresh_list = [thresh for thresh in thresh_train if (thresh_max >= thresh >= thresh_min)] + \
            [thresh for thresh in thresh_validate if (thresh_max >= thresh >= thresh_min)]

        #Find the threshold with the lowest error (sum of false negative and false positive rates)
        best_error = float('inf')
        for thresh in thresh_list:
            #Calculate model error for selected threshold
            model.threshold = thresh
            fnr_train, fpr_train = model_error(model, X_train, y_train)
            fnr_validate, fpr_validate = model_error(model, X_validate, y_validate)

            #Save threshold with lowest error
            error = fnr_train + fpr_train + fnr_validate + fpr_validate
            if error < best_error:
                best_error = error
                best_thresh = thresh
    else:
        print('No feasible thresholds exist.')
        return None

    #Set threshold of model and return it
    model.threshold = best_thresh

    #temporary
    fnr_train, fpr_train = model_error(model, X_train, y_train)
    fnr_validate, fpr_validate = model_error(model, X_validate, y_validate)
    print(f'Threshold: {best_thresh:.3f}')
    print(f'TRAIN: FNR: {100*fnr_train:.3f}%    FPR: {100*fpr_train:.3f}%')
    print(f'VALID: FNR: {100*fnr_validate:.3f}%    FPR: {100*fpr_validate:.3f}%')
    return model

def model_error(model: Model, X: np.ndarray, y: np.ndarray) -> tuple[float]:
    """Calculates false positive and false negative error rates for model on given dataset

    Args:
        model (Model): Trained model
        X (np.ndarray): Sample features
        y (np.ndarray): Sample labels

    Returns:
        tuple[float]: False negative rate and false positive rate
    """
    y_pred = 1*(model.clf.predict_proba(X[:, model.feats_idx])[:,-1] >= model.threshold)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    fnr, fpr = fn/(fn+tp), fp/(fp+tn)

    return fnr, fpr

def roc_thresholds(model: Model, X: np.ndarray, y: np.ndarray) -> list[float]:
    """Uses ROC-curve to find thresholds that satisfy error rate constraints

    Args:
        model (Model): Trained model
        X (np.ndarray): Sample features
        y (np.ndarray): Sample labels

    Returns:
        list[float]: List of thresholds
    """
    y_prob = model.clf.predict_proba(X[:, model.feats_idx])[:,-1]
    fpr_list, tpr_list, thresh_list = roc_curve(y, y_prob)
    fnr_list = 1 - tpr_list
    thresholds = [thresh for (fnr, fpr, thresh) in zip(fnr_list, fpr_list, thresh_list) if not (fpr > c.FPR_MAX or fnr > c.FNR_MAX)]
    return thresholds

#SAMPLES RELATED FUNCTIONS
def samples_load(file_path: str) -> tuple[np.ndarray]:
    """Loads samples from .pkl file

    Args:
        file_path (str): Path to file

    Returns:
        tuple[np.ndarray]: Sample features
    """
    with open(file_path, 'rb') as f:
        samples = pickle.load(f)
        return samples

def samples_del(model: Model, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deletes false negative and true negative samples

    Args:
        model (Model): Trained model
        X (np.ndarray): Sample features
        y (np.ndarray): Sample labels

    Returns:
        tuple[np.ndarray, np.ndarray]: Updated sample features and sample labels
    """
    
    #Save true positives and false positives, remove true negatives and false negatives
    y_pred = 1*(model.clf.predict_proba(X[:, model.feats_idx])[:,-1] >= model.threshold)
    n_prev = len(y)
    X = [x for (x, true, pred) in zip(X, y, y_pred) if ((true == 1 and pred == 1) or (true == 0 and pred == 1))]
    y = [true for (true, pred) in zip(y, y_pred) if ((true == 1 and pred == 1) or (true == 0 and pred == 1))]

    print(f'Deleted {n_prev-len(y)} samples ({round(100*len(y)/len(y_pred), 2)}%).')
    return X, y

def samples_add(models: list[Model], X: np.ndarray, y: np.ndarray, img_list: list[str]) -> tuple[np.ndarray, np.ndarray, str]:
    """Adds false negative samples from the negative dataset.

    Args:
        models (list[Model]): Trained models
        X (np.ndarray): Sample features
        y (np.ndarray): Sample labels
        img_list (list[str]): List of paths to not yet exhausted images in the negative dataset

    Returns:
        tuple[np.ndarray, np.ndarray, str]: Updated sample features, sample labels and path to last selected image.
    """

    #Check current ratio, return if sufficient
    n_neg = y.count(0)
    n_pos = y.count(1)
    neg_ratio = n_neg/(n_neg+n_pos)
    if neg_ratio > c.NEG_MIN_RATIO:
        return X, y, img_list

    #Number of negative samples to add
    n_add_tot = int((c.NEG_MIN_RATIO*(n_pos+n_neg) - n_neg)/(1-c.NEG_MIN_RATIO))
    n_add = n_add_tot

    #List images path if first time
    if img_list == None:
        imgs_path = os.listdir(c.NEG_DSET_PATH)
        img_list = [f'{c.NEG_DSET_PATH}/{img}' for img in imgs_path]

    #Iterate though list of images and add samples
    samples = []
    while True:
        #Select images to scan on each process
        n_cpu = os.cpu_count()
        img_list_select = img_list[0:n_cpu]
        n_add_process = math.ceil(n_add/n_cpu)

        img_process_list = []
        for img_select in img_list_select:
            img = Image.open(img_select)
            img = ImageOps.grayscale(img)
            img_process_list.append(img)

        #Multiprocess selected images
        partial_args = functools.partial(image_scan, models, n_add_process)
        with concurrent.futures.ProcessPoolExecutor() as pool:
            samples_process_list = list(pool.map(partial_args, img_process_list))
        
        #Collect samples, if an image returns too few, the image is exhausted: remove from list of images
        for (img_select, sample_process) in zip(img_list_select, samples_process_list):
            if len(sample_process) < n_add_process:
                img_list = [img for img in img_list if not img == img_select]
        samples += list(itertools.chain.from_iterable(samples_process_list))
        print(f'Collected {len(samples)}/{n_add_tot} samples...')

        #Check if enough samples have been added
        if len(samples) - n_add >= 0:
            break
        else:
            n_add -= len(samples)
            
    #Add negative samples and labels to dataset
    X = np.concatenate((X, samples))
    y = np.concatenate((y, np.array(len(samples)*[0])))
    print(f'Added {len(samples)} samples.')
    print(f'{len(img_list)} negative images remain.')
    return X, y, img_list

def image_scan(models: list[Model], n_add: int, img: Image) -> list[np.ndarray]:
    """Scans image and returns haar-like features if accepted by every layer of models.

    Args:
        models (list[Model]): List of models for each layer
        img (Image): Input grayscale image
        n_add (int): Desired number of images to add
        pos_shift (tuple[int]): Parameters determining subsection coordinates

    Returns:
        list: Accepted samples consisting of 162336 haar-like features each.
    """
    size_subwindow = 24
    w, h = img.size
    size_img = min(w,h)
    samples = []
    x_pos = 0
    y_pos = 0

    #Scan through image
    while size_subwindow <= size_img:
        while y_pos+size_subwindow <= h:
            while x_pos+size_subwindow <= w:
                #Get subwindow and resize
                subwindow_bounds = (x_pos, y_pos, x_pos+size_subwindow-1, y_pos+size_subwindow-1)
                subwindow = img.crop(subwindow_bounds)
                
                #Resize and standardize, discard if standard deviation is too low (homogeneous subwindow)
                subwindow = subwindow.resize((24, 24))
                subwindow = np.array(subwindow)
                subwindow_std = np.std(subwindow)
                if subwindow_std < 1:
                    x_pos += c.DELTA
                    continue
                subwindow = 255*(subwindow - np.mean(subwindow))/subwindow_std

                #Create integral image and iterate through each layer of models:
                subwindow_ii = integral_image(subwindow)
                for model in models:
                    features = haar_like_feature(subwindow_ii, 0, 0, 24, 24, feature_type=model.feat_types, feature_coord=model.feats)
                    y_pred = 1*(model.clf.predict_proba(features.reshape(1, -1))[:, 1] >= model.threshold)

                    #If it fails in a layer, then discard it
                    if y_pred == 0:
                        break

                    #Calculate the 162336 haar-like features and append if it passes through all layers
                    elif model == models[-1]:
                        samples.append(haar_like_feature(subwindow_ii, 0, 0, 24, 24))
                        if len(samples) >= n_add:
                            return samples

                #New column: increment column index
                x_pos += c.DELTA
            
            #New row: increment row index, reset column index
            y_pos += c.DELTA
            x_pos = 0
        
        #All rows and columns scanned: increment subwindow scale, reset row and column index
        size_subwindow *= c.SCALE
        size_subwindow = int(size_subwindow)
        x_pos, y_pos = 0, 0
    return samples