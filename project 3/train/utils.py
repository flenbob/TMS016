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

def fit_model(X: np.ndarray, y: np.ndarray, n_wc: int) -> Model:
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
        #Define classifier and train model
        clf = AdaBoostClassifier(
            DecisionTreeClassifier(criterion='gini', max_depth=1), 
            algorithm="SAMME.R", 
            n_estimators=n_wc
            )
        X_train, X_validate, y_train, y_validate = train_test_split(X, y, train_size = 0.75, random_state=42)
        clf.fit(X_train, y_train)
        model = model_reduce(clf, X, y)

        #Select threshold for model
        model, FNR, FPR = select_threshold(model, X_train, y_train)
        
        #Control target rates on train subset
        if not (FNR < c.FNR_MAX and FPR < c.FPR_MAX):
            print(f'Train: SC with {n_wc} WC:s was not enough. Rerun with {n_wc + math.ceil(c.WC_RATE*n_wc)} WC:s.')
            model = None
            n_wc += math.ceil(c.WC_RATE*n_wc)
            continue

        #Control target rates on validaiton subset
        model = vaildate_model(model, X_validate, y_validate)
        if not model:
            print(f'Validate: SC with {n_wc} WC:s was not enough. Rerun with {n_wc + math.ceil(c.WC_RATE*n_wc)} WC:s.')
            n_wc += math.ceil(c.WC_RATE*n_wc)
            continue
    return model

def read_sample_file(file_path: str) -> tuple[np.ndarray]:
    with open(file_path, 'rb') as f:
        samples = pickle.load(f)
        return samples

def delete_samples(model: Model, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deletes false negative and true negative samples

    Args:
        model (Model): Trained model
        X (np.ndarray): Sample features
        y (np.ndarray): Sample labels

    Returns:
        tuple[np.ndarray]: Sample features and sample labels
    """
    delete_idx = []
    y_pred = 1*(model.clf.predict_proba(X[:, model.feats_idx])[:,-1] >= model.threshold)

    #If sample is true negative or false negative, remove it
    for i in range(len(y_pred)):
        if ((y[i] == 0) and (y_pred[i] == 0)) or \
            ((y[i] == 1) and (y_pred[i] == 0)):
            delete_idx.append(i)

    print(f'Deleted {len(delete_idx)} samples ({round(100*len(delete_idx)/len(y_pred), 2)}%).')
    X = np.delete(X, delete_idx, axis=0)
    y = np.delete(y, delete_idx)
    return X, y

def select_threshold(model: Model, X: np.ndarray, y: np.ndarray) -> tuple[Model, float, float]:
    """Selects threshold which achieves accepted FNR and FPR, and minimizes FNR as much as possible.

    Args:
        model (Model): Trained model
        X (np.ndarray): Sample features
        y (np.ndarray): Sample labels

    Returns:
        tuple[Model, float, float]: Model with selected threshold, and given FNR and FPR.
    """
    #Generate ROC-curve
    y_prob = model.clf.predict_proba(X[:, model.feats_idx])[:, -1]
    roc = roc_curve(y, y_prob)
    FPR_list = roc[0]
    FNR_list = 1 - roc[1]
    threshold_list = roc[2]

    #Find index which maximizes FPR (and minimizes FNR), but is within constraints.
    FPR = FPR_list[0]
    idx = 0
    while FPR <= c.FPR_MAX:
        idx += 1
        FPR = FPR_list[idx]
    
    FPR = FPR_list[idx-1]
    FNR = FNR_list[idx-1]
    thresh = threshold_list[idx-1]

    #Set model threshold
    print(f'Selected threshold: {thresh:.4f}')
    print(f'Train: FNR = {FNR:.4f} and FPR = {FPR:.4f}')
    model.threshold = thresh
    return model, FNR, FPR

def vaildate_model(model: Model, X: np.ndarray, y: np.ndarray) -> Model:
    """Validates that trained model yields lower than maximum accepted false negative and false positive rates for validation data.

    Args:
        model (Model): Trained model.
        X (np.ndarray): Validation sample features.
        y (np.ndarray): Validation sample labels.

    Returns:
        Model: Validated model or None if validation is not accepted.
    """

    #Control target rates on validation subset
    y_pred = 1*(model.clf.predict_proba(X[:, model.feats_idx])[:,-1] >= model.threshold)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    fnr = fn/(fn+tp)
    fpr = fp/(fp+tn)

    print(f'Validate: FNR = {fnr:.4f} and FPR = {fpr:.4f}')
    if not (fnr < c.FNR_MAX and fpr < c.FPR_MAX):
        return None
    return model

def model_reduce(clf: AdaBoostClassifier, X: np.ndarray, y: np.ndarray) -> Model:
    """Converts classifier to a model which takes into account a reduced number of features.

    Args:
        clf (AdaBoostClassifier): Strong classifier
        X (np.ndarray): Sample features
        y (np.ndarray): Sample labels

    Returns:
        Model: Model object with threshold and haar-like features of interest.
    """

    n_estimators = np.count_nonzero(clf.feature_importances_ != 0)
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

def cascade_scan(models: list[Model], img: Image, n_add: int, pos_shift: tuple[int]) -> list:
    size_subwindow = 24
    w, h = img.size
    size_img = min(w,h)
    samples = []
    x_pos = pos_shift[0]
    y_pos = pos_shift[1]

    y_pred_avg = np.zeros(7)
    while size_subwindow <= size_img:
        while y_pos+size_subwindow <= h:
            while x_pos+size_subwindow <= w:
                #Get subwindow and resize
                subwindow_bounds = (x_pos, y_pos, x_pos+size_subwindow-1, y_pos+size_subwindow-1)
                subwindow = img.crop(subwindow_bounds)
                
                #Resize and standardize
                subwindow = subwindow.resize((24, 24))
                subwindow = np.array(subwindow)
                subwindow_std = np.std(subwindow)

                #If std is too low, then discard
                if subwindow_std < 1:
                    x_pos += pos_shift[2]
                    continue

                subwindow = 255*(subwindow - np.mean(subwindow))/subwindow_std
                #Integral image
                subwindow_ii = integral_image(subwindow)

                #Cascade layers
                for i, model in enumerate(models):
                    features = haar_like_feature(subwindow_ii, 0, 0, 24, 24, feature_type=model.feat_types, feature_coord=model.feats)
                    y_pred = 1*(model.clf.predict_proba(features.reshape(1,-1))[:, 1] >= model.threshold)
                    y_pred_avg[i] = y_pred

                    #If it fails in a layer, then discard it
                    if y_pred == 0:
                        break

                    #If it passes through cascade, then calculate all of its 162336 haar like features
                    elif model == models[-1]:
                        samples.append(haar_like_feature(subwindow_ii, 0, 0, 24, 24))
                        if len(samples) > n_add/7:
                            return samples

                x_pos += pos_shift[2]
            y_pos += pos_shift[3]
            x_pos = pos_shift[0]
        size_subwindow *= c.SCALE
        size_subwindow = int(size_subwindow)
        x_pos = pos_shift[0]
        y_pos = pos_shift[1]
    return samples

def add_samples(models: list[Model], X: np.ndarray, y: np.ndarray, last_visited: str) -> tuple[np.ndarray, np.ndarray, str]:
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
    n_neg = np.count_nonzero(y==0)
    n_pos = np.count_nonzero(y==1)
    neg_ratio = n_neg/(n_neg+n_pos)
    if neg_ratio > c.NEG_MIN_RATIO:
        return X, y, last_visited

    #Number of negative samples to add
    n_add = int((c.NEG_MIN_RATIO*(n_pos+n_neg) - n_neg)/(1-c.NEG_MIN_RATIO))
    print(f'Adding {n_add} samples...')

    #List images path
    imgs = os.listdir(c.NEG_DSET_PATH)
    imgs = [f'{c.NEG_DSET_PATH}/{img}' for img in imgs]

    #Slice at last visited
    if last_visited != None:
        imgs = imgs[imgs.index(last_visited)+1:-1]

    #Iterate though images and add samples
    samples = []
    i = 0
    while True:
        #Read image
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
        partial_args = functools.partial(cascade_scan, models, img, n_add)
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