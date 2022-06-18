import numpy as np
import os
import concurrent.futures
import numpy as np
import functools
import pickle
from PIL import Image, ImageOps
from skimage.feature import haar_like_feature, haar_like_feature_coord
from skimage.transform import integral_image
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, KFold
from model import Model

def read_sample_file(file_path: str) -> tuple[np.ndarray]:
    pass

def read_model_file(file_path: str) -> list:
    #For old model version generated
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

def delete_samples(model: Model, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray]:
    """Deletes false negative and true negative samples

    Args:
        model (Model): Trained model
        X (np.ndarray): Sample features
        y (np.ndarray): Sample labels

    Returns:
        tuple[np.ndarray]: Subset of sample features, sample labels and sample label predicitons
    """
    delete_idx = []
    y_pred = 1*(model.clf.predict_proba(X[:, model.feats_idx])[:,-1] >= model.threshold)

    #If sample is true negative or false negative, remove it
    for i in range(len(y_pred)):
        if ((y[i] == 0) and (y_pred[i] == 0)) or \
            ((y[i] == 1) and (y_pred[i] == 0)):
            delete_idx.append(i)

    print(f'Deleted {round(100*len(delete_idx)/len(y_pred), 2)}% of samples')
    X = np.delete(X, delete_idx, axis=0)
    y = np.delete(y, delete_idx)

    return X, y

def get_rates(X: np.ndarray, y: np.ndarray, model: Model=None, y_pred: np.ndarray=None) -> tuple:
    """Get false negative and false positive rates for samples X and labels y for either model or label prediction array y_pred

    Args:
        X (np.ndarray): Sample features.
        y (np.ndarray): Sample labels.
        model (Model, optional): Trained model. Defaults to None.
        y_pred (np.ndarray, optional): Label prediction array. Defaults to None.

    Raises:
        Exception: Error if neither or both model and y_pred are not None.

    Returns:
        tuple: false negative and false positive rates.
    """
    try:
        if Model:
            y_pred = 1*(model.clf.predict_proba(X)[:,-1] >= model.threshold)
            tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        elif y_pred:
            tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    except:
        raise Exception("Incorrect input, allows either model or y_pred. Not neither, or both.")
        
    fnr = fn/(fn+tp)
    fpr = fp/(fp+tn)

    return fnr, fpr

def optimize_threshold(roc: tuple[np.ndarray], FNR_max: float, FPR_max: float) -> tuple[float]:
    """Uses ROC-curve to find threshold which is below maximum accepted false negative and false positive rates, and optimizes further if possible.

    Args:
        roc (tuple[np.ndarray]): arrays returned from ROC, consisting of false positive rates, false negative rates and thresholds.
        FNR_max (float): Maximum accepted false negative rate.
        FPR_max (float): Maximum accepted false positive rate.

    Returns:
        tuple[float]: Optimal threshold and given false positive and false negative rates
    """

    FPR_list = roc[0]
    FNR_list = 1 - roc[1]
    threshold_list = roc[2]
    #Find first index which achieves accepted false negative rate
 
    
    idx = 0
    FNR = FNR_list[0]
    while FNR >= FNR_max:
        idx += 1
        FNR = FNR_list[idx]
    
    FPR = FPR_list[idx]
    thresh = threshold_list[idx]

    #If we achieve both rates, try to optimize further
    if FPR < FPR_max:
        #Slice indicies which have an accepted FNR
        FNR_list = FNR_list[idx:-1]
        FPR_list = FPR_list[idx:-1]
        threshold_list = threshold_list[idx:-1]

        idx_min = np.argmin(np.abs(FNR_list - FNR_max) + np.abs(FPR_list - FPR_max))
        FNR = FNR_list[idx_min]
        FPR = FPR_list[idx_min]
        thresh = threshold_list[idx_min]

    return thresh, FNR, FPR

def vaildate_model(model: Model, X_validate: np.ndarray, y_validate: np.ndarray, FNR_max: float, FPR_max: float) -> Model:
    """Validates that trained model yields lower than maximum accepted false negative and false positive rates for validation data.

    Args:
        model (Model): Trained model.
        X_validate (np.ndarray): Validation sample features.
        y_validate (np.ndarray): Validation sample labels.
        FNR_max (float): Maximum accepted false negative rate.
        FPR_max (float): Maximum accepted false positive rate.

    Returns:
        Model: Validated model or None if validation is not accepted.
    """

    #Control target rates on validation subset
    FNR, FPR = get_rates(X_validate, y_validate, model=model)
    if not (FNR < FNR_max and FPR < FPR_max):
        return None

    return model

def cross_validate(clf: AdaBoostClassifier, X: np.ndarray, y: np.ndarray, FNR_target: float, FPR_target: float, n_splits: int) -> list:
    # OLD ???
    skf = KFold(n_splits=n_splits)
    FNR_fold = []
    FPR_fold = []
    FNR_best = 1
    FPR_best = 1
    X_KFold, X_test, y_KFold, y_test, = train_test_split(X, y, train_size=0.75, random_state=42, stratify=y)
 
    for i, (train_idx, test_idx) in enumerate(skf.split(X_KFold)):
        print(f'Fold {i}')
        X_train, X_validate = X_KFold[train_idx], X_KFold[test_idx]
        y_train, y_validate = y_KFold[train_idx], y_KFold[test_idx]
        
        #Train SC
        clf.fit(X_train, y_train)

        #Train SC w.r.t selected features of train clf_reduced
        model = model_reduce(clf, X_train, y_train)

        #Tune hyperparameter FPR and FNR of SC (reduced) and then validate
        y_prob = model.clf.predict_proba(X_validate[:, model.feats_idx])[:, -1]
        FPR_list, TPR_list, threshold_list = roc_curve(y_validate, y_prob)
        FNR_list = 1 - TPR_list
        thresh, FPR, FNR = optimize_threshold(FNR_list, FPR_list, threshold_list, FNR_target)
        model.threshold = thresh
        print(f'Threshold = {thresh} gives FNR = {FNR}, FPR = {FPR}')

        #If feasible and best performing, save model and threshold
        if FPR < FPR_target and FNR < FNR_target and FPR < FPR_best and FNR < FNR_best:
            model_best = model
        
        FNR_fold.append(FNR)
        FPR_fold.append(FPR)

    #If the mean performance of our models is worse than target, then we need more WC:s
    if sum(FNR_fold)/len(FNR_fold) > FNR_target or sum(FPR_fold)/len(FPR_fold) > FPR_target:
        flag = False
        clf_best = None
        return clf_best, flag

    #Run best model and its threshold on test data and check for overfitting
    y_pred = 1*(clf_best[0].predict_proba(X_test[:, clf_best[4]])[:, -1] >= clf_best[1])
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    FNR = FN/(FN+TP)
    FPR = FP/(FP+TN)
    print(f'VALIDATION SET: Threshold: {clf_best[1]} gives: FNR = {FNR}, FPR = {FPR}')

    #Return
    flag = True
    return clf_best, flag

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

def cascade_scan(models: list[Model], img: Image, size_subwindow: int, scale: float, pos_shift: int) -> list:
    w, h = img.size
    size_img = min(w,h)
    img_ii_list = []
    subwindow_bounds_list = []
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
                    #OLD MODEL VERSION, CHANGE SOON
                    SC = model[0]
                    threshold = model[1]
                    feature_types = model[2]
                    feature_coords = model[3]

                    features = haar_like_feature(subwindow_ii, 0, 0, 24, 24, feature_type=feature_types, feature_coord=feature_coords)
                    y_pred = 1*(SC.predict_proba(features.reshape(1,-1))[:, 1] >= threshold)
                    y_pred_avg[i] = y_pred

                    #If it fails in a layer, then discard it
                    if y_pred == 0:
                        break
                    elif model == models[-1]:
                        #If it passes through cascade, then save it
                        img_ii_list.append(subwindow_ii)
                        subwindow_bounds_list.append([subwindow_bounds, np.mean(y_pred_avg)])
                x_pos += pos_shift[2]
            y_pos += pos_shift[3]
            x_pos = pos_shift[0]
        size_subwindow *= scale
        size_subwindow = int(size_subwindow)
        x_pos = pos_shift[0]
        y_pos = pos_shift[1]
    return img_ii_list, subwindow_bounds_list

def add_samples(models: list[Model], img: Image, scale: float, delta: int) -> list[np.ndarray]:
    """Generates list of integral images for negatively labeled samples from subwindows of the input image.
    Args:
        models (list[Model]): List of all model layers.
        img (Image): Grayscale image from which subwindows are generated.
        scale (float): Subwindow size growth rate for each round
        delta (int): Distances between each subwindow in pixels.

    Returns:
        list[np.ndarray]: List of integral images. 
    """

    imgs_list = []
    size_subwindow = 24
    w, h = img.size

    N = os.cpu_count()
    if w >= h:
        pos_shift = [(delta*i, 0, N*delta, delta)  for i in range(N)]
    else:
        pos_shift = [(0, delta*i, delta, N*delta) for i in range(N)]

    partial_args = functools.partial(cascade_scan, models, img, size_subwindow, scale)
    with concurrent.futures.ProcessPoolExecutor() as pool:
        results = list(pool.map(partial_args, pos_shift))

    for result in results:
        imgs_list.append(result)

    return imgs_list
