import constants as c
from model import Model
import pickle
import numpy as np
import os
import math
import concurrent.futures
import functools
from PIL import Image, ImageOps
from skimage.feature import haar_like_feature, haar_like_feature_coord
from skimage.transform import integral_image
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split


#MODEL RELATED FUNCTIONS
def model_fit(X: np.ndarray, y: np.ndarray, n_wc: int) -> Model:
    """Fits a model consisting of an Adaboost classifier with weak classifiers (decision stumps).

    Args:
        X (np.ndarray): Sample features
        y (np.ndarray): Sample labels
        n_wc (int): Number of weak classifiers

    Returns:
        Model: Trained model
    """
    model = None
    while not model:
        #Define classifier and train model on training data
        clf = AdaBoostClassifier(
            DecisionTreeClassifier(criterion='gini', max_depth=1), 
            algorithm="SAMME.R", 
            n_estimators=n_wc
            )
        X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size = 0.75, random_state=42)
        clf.fit(X_train, y_train)

        model = model_reduce(clf, X_train, y_train)
        model = model_threshold(model, X_train, X_validate, y_train, y_validate)
        if not model:
            print(f'Train: Model with {n_wc} WC:s was not enough. Rerun with {n_wc + math.ceil(c.WC_RATE*n_wc)} WC:s.')
            n_wc += math.ceil(c.WC_RATE*n_wc)
            continue

        #Validate model on validation dataset
        # model = model_validate(model, X_validate, y_validate)
        # if not model:
        #     print(f'Validate: Model with {n_wc} WC:s was not enough. Rerun with {n_wc + math.ceil(c.WC_RATE*n_wc)} WC:s.')
        #     n_wc += math.ceil(c.WC_RATE*n_wc)
        #     continue
    return model

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
    #Get feasible training data thresholds
    y_prob = model.clf.predict_proba(X_train[:, model.feats_idx])[:,-1]
    fpr_list, tpr_list, thresh_list_train = roc_curve(y_train, y_prob)
    fnr_list = 1 - tpr_list
    thresh_train = [thresh for (fnr, fpr, thresh) in zip(fnr_list, fpr_list, thresh_list_train) if not (fpr > c.FPR_MAX or fnr > c.FNR_MAX)]
    # print(f'Train: FNR = {fnr_list}, FPR = {fpr_list}, thresh = {thresh_list_train}')
    # print(f'thresh train: {thresh_train}')

    #Get feasible validation data thresholds
    y_prob = model.clf.predict_proba(X_validate[:, model.feats_idx])[:,-1]
    fpr_list, tpr_list, thresh_list_validate = roc_curve(y_validate, y_prob)
    fnr_list = 1 - tpr_list
    thresh_validate = [thresh for (fnr, fpr, thresh) in zip(fnr_list, fpr_list, thresh_list_validate) if not (fpr > c.FPR_MAX or fnr > c.FNR_MAX)]
    # print(f'Validate: FNR = {fnr_list}, FPR = {fpr_list}, thresh = {thresh_list_validate}')
    # print(f'thresh validation: {thresh_validate}')

    if len(thresh_validate) == 0 or len(thresh_train) == 0:
        return None 

    #Check if there exists feasible thresholds for both datasets
    if max(min(thresh_validate), min(thresh_train)) <= min(max(thresh_validate), max(thresh_train)):
        thresh_min = max(min(thresh_validate), min(thresh_train))
        thresh_max = min(max(thresh_validate), max(thresh_train))

        # print(f'Thresh min = {thresh_min}, thresh_max = {thresh_max}')
        # print(f'Thresh train = {thresh_train}')
        # print(f'Thresh validate = {thresh_validate}')

        #Select all feasible thresholds
        thresh_vals_train = [thresh for thresh in thresh_train if (thresh_max >= thresh >= thresh_min)]
        thresh_vals_validate = [thresh for thresh in thresh_validate if (thresh_max >= thresh >= thresh_min)]
        thresh_vals = thresh_vals_train + thresh_vals_validate
        
        # print(f'Thresh vals = {thresh_vals}')

        #Find the threshold with the lowest error (sum of false negative and false positive rates)
        best_error = float('inf')
        for thresh in thresh_vals:
            y_pred = 1*(model.clf.predict_proba(X_train[:, model.feats_idx])[:,-1] >= thresh)
            tn, fp, fn, tp = confusion_matrix(y_train, y_pred, labels=[0, 1]).ravel()
            fnr_train, fpr_train = fn/(fn+tp), fp/(fp+tn)

            y_pred = 1*(model.clf.predict_proba(X_validate[:, model.feats_idx])[:, -1] >= thresh)
            tn, fp, fn, tp = confusion_matrix(y_validate, y_pred, labels=[0, 1]).ravel()
            fnr_validate, fpr_validate = fn/(fn+tp), fp/(fp+tn)
        
            error = fnr_train + fpr_train + fnr_validate + fpr_validate
            if error < best_error:
                best_error = error
                best_thresh = thresh
    else:
        return None

    model.threshold = best_thresh
    return model

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
    
    #Check true and predicted values and remove true negative and false negatives.
    y_pred = 1*(model.clf.predict_proba(X[:, model.feats_idx])[:,-1] >= model.threshold)
    n_prev = len(y)
    X = [x for (x, true, pred) in zip(X, y, y_pred) if not ((true == 0 and pred == 0) or (true == 1 and pred == 0))]
    y = [true for (true, pred) in zip(y, y_pred) if ((true == 1 and pred == 1) or (true == 0 and pred == 1))]

    #OLD:
    # for i in range(len(y_pred)):
    #     if ((y[i] == 0) and (y_pred[i] == 0)) or \
    #         ((y[i] == 1) and (y_pred[i] == 0)):
    #         del_idx.append(i)

    #Delete indicies
    # X = np.delete(X, del_idx, axis=0)
    # y = np.delete(y, del_idx)

    print(f'Deleted {n_prev-len(y)} samples ({round(100*len(y)/len(y_pred), 2)}%).')
    return X, y

def samples_add(models: list[Model], X: np.ndarray, y: np.ndarray, last_visited: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Adds false negative samples from the negative dataset.

    Args:
        models (list[Model]): Trained model
        X (np.ndarray): Sample features
        y (np.ndarray): Sample labels
        last_visited (str): Path to last visited image in the negative dataset

    Returns:
        tuple[np.ndarray, np.ndarray, str]: Updated sample features, sample labels and image path.
    """

    #Check current ratio, return if sufficient
    n_neg = y.count(0)
    n_pos = y.count(1)
    neg_ratio = n_neg/(n_neg+n_pos)
    if neg_ratio > c.NEG_MIN_RATIO:
        return X, y, last_visited

    #Number of negative samples to add
    n_add = int((c.NEG_MIN_RATIO*(n_pos+n_neg) - n_neg)/(1-c.NEG_MIN_RATIO))

    #List images path
    imgs = os.listdir(c.NEG_DSET_PATH)
    imgs = [f'{c.NEG_DSET_PATH}/{img}' for img in imgs]


    #Slice at last visited image
    if last_visited != None:
        imgs = imgs[imgs.index(last_visited)+1:-1]

    #Iterate though images and add samples
    samples = []
    i = 0
    while True:
        #Read image
        last_visited = imgs[i]
        print(f'Adding {n_add} samples from {imgs[i]}')
        img = Image.open(imgs[i])
        img = ImageOps.grayscale(img)
        i += 1

        #Split image into sections
        w, h = img.size
        n_cpu = os.cpu_count()
        if w >= h:
            pos_shift = [(c.DELTA*i, 0, n_cpu*c.DELTA, c.DELTA)  for i in range(n_cpu)]
        else:
            pos_shift = [(0, c.DELTA*i, c.DELTA, n_cpu*c.DELTA) for i in range(n_cpu)]

        #Multiprocess
        partial_args = functools.partial(image_scan, models, img, int(n_add/n_cpu))
        with concurrent.futures.ProcessPoolExecutor() as pool:
            sample_splits = pool.map(partial_args, pos_shift)
        
        #Append result from each process
        for sample_split in sample_splits:
            for sample in sample_split:
                samples.append(sample)

        #Terminate if enough samples are appended
        if len(samples) >= n_add:
            break
        
    #Add negative samples and labels to dataset
    X = np.concatenate((X, samples))
    y = np.concatenate((y, np.array(len(samples)*[0])))
    return X, y, last_visited

def image_scan(models: list[Model], img: Image, n_add: int, pos_shift: tuple[float]) -> list[np.ndarray]:
    """Scans through subsection of input image and returns haar-like features if accepted by every layer of models.

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
    x_pos = pos_shift[0]
    y_pos = pos_shift[1]

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
                    x_pos += pos_shift[2]
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
                        if len(samples) > n_add:
                            return samples

                x_pos += pos_shift[2]
            y_pos += pos_shift[3]
            x_pos = pos_shift[0]
        size_subwindow *= c.SCALE
        size_subwindow = int(size_subwindow)
        x_pos = pos_shift[0]
        y_pos = pos_shift[1]
    return samples