import pickle
import numpy as np
from model import Model
from skimage.feature import  haar_like_feature
from skimage.transform import integral_image
from PIL import Image, ImageOps
import constants as c

def read_model_file(file_path: str) -> list:
    models = []
    with open(file_path, 'rb') as f:
        try:
            while True:
                model = pickle.load(f)
                models.append(model)
        except (EOFError):
            pass
    return models

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

def cascade_classifier(models: list[Model], X: np.ndarray) -> np.ndarray:
    """Runs classifier from each layer in a cascade and predicts the data label

    Args:
        models (list[Model]): Trained models
        X (np.ndarray): Sample features

    Returns:
        np.ndarray: Predicted sample labels
    """
    y = len(X)*[1]
    
    for model in models:
        y = [y_i if y_i == 0 else 1*(model.clf.predict_proba(X_i[model.feats_idx].reshape(1, -1))[:, -1] >= model.threshold) for X_i, y_i in zip(X, y)]
    return y

def image_scan(models: list[Model], img: Image, pos_shift: tuple[float]) -> list[np.ndarray]:
    """Scans through subsection of input image and returns haar-like features if accepted by every layer of models.

    Args:
        models (list[Model]): List of models for each layer
        img (Image): Input grayscale image
        n_add (int): Desired number of images to add
        pos_shift (tuple[int]): Parameters determining subsection coordinates: 
            (x_pos: Initial x-position in image,
             y_pos: Initial y-position in image,
             x_pos_delta: x-position shift parameter
             y_pos_delta: y-position shift parameter).
             
    Returns:
        list: Accepted samples consisting of 162336 haar-like features each.
    """
    size_subwindow = 24
    w, h = img.size
    size_img = min(w,h)
    boxes = []
    x_pos = pos_shift[0]
    y_pos = pos_shift[1]

    #Scan through image
    while size_subwindow <= size_img:
        while y_pos+size_subwindow <= h:
            while x_pos+size_subwindow <= w:
                #Get subwindow and resize
                subwindow_box = (x_pos, y_pos, x_pos+size_subwindow-1, y_pos+size_subwindow-1)
                subwindow = img.crop(subwindow_box)
                
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
                        #samples.append(haar_like_feature(subwindow_ii, 0, 0, 24, 24))
                        # if len(samples) > n_add:
                        #     return samples

                x_pos += pos_shift[2]
            y_pos += pos_shift[3]
            x_pos = pos_shift[0]
        size_subwindow *= c.SCALE
        size_subwindow = int(size_subwindow)
        x_pos = pos_shift[0]
        y_pos = pos_shift[1]
    return boxes