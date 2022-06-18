from typing import Any
import numpy as np
import pickle
import h5py
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from skimage.feature import draw_haar_like_feature, haar_like_feature
from skimage.transform import integral_image
from PIL import Image, ImageOps
import cv2
import cProfile, pstats
import time


def main():
    models = read_model_file('abclf_4.pkl')
    models = models[-1]

    img = Image.open('../test/img2.jpg')
    img = ImageOps.grayscale(img)
    start = time.perf_counter()
    with cProfile.Profile() as pr:
        img_ii, subw = cascade_scan(models, 2, 3, 24, img)
        print(len(img_ii))
    stats = pstats.Stats(pr).sort_stats('cumtime')
    stats.print_stats()
    finish = time.perf_counter()
    print(f'Finished in: {round(finish-start, 2)} seconds')


def plot_box_img(file_path: str, slice: float, img_path: str) -> None:
    #Save image with boxes that were written to the .h5 file by 'write_scanbox_file()'
    hf = h5py.File(file_path, 'r')
    boxes = hf['data']
    boxes = [box for box in boxes]
    hf.close()

    #Slice the least important boxes:
    boxes = boxes[0:int(slice*len(boxes))]

    #Read image and plot boxes to it
    img = cv2.imread(img_path)
    for box_coords in boxes:
        left = box_coords[0]
        top = box_coords[1]
        right = box_coords[2]
        bottom = box_coords[3]
        img = cv2.rectangle(img, (left, top), (right, bottom), color=(255, 0, 0), thickness=2)


    cv2.imwrite('boxes_photo.jpg', img)
    
def write_scanbox_file(models: Any, img_path: str, scale: float, delta: float) -> None:
    #Run cascade scan through image and write coordinate of subwindows which are
    #passed as positive, sorted by confidence (high -> low)
    img = Image.open(img)
    img = ImageOps.grayscale(img)
    _, subwindow_bounds_list = cascade_scan(models, img, scale, delta)
    img_size_sort = []

    proba = [el[1] for el in subwindow_bounds_list]
    img_size = [el[0] for el in subwindow_bounds_list]
    idx_sorted = sorted(range(len(proba)), key=lambda k: proba[k])
    idx_sorted = idx_sorted[::-1]

    for i in idx_sorted:
        img_size_sort.append(img_size[i])

    hf = h5py.File('face_boxes.h5', 'w')
    g1 = hf.create_dataset('data', data=img_size_sort)
    hf.close()

def plot_error_rate(models: Any, save_path: str, X: np.ndarray, y: np.ndarray) -> None:
    ERR = []
    FNR = []
    FPR = []
    curr_model = []
    for model in models:
        curr_model.append(model)
        y_pred = cascade_classifier(X, curr_model)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        ERR.append((fp+fn)/(tn+fp+fn+tp))
        FNR.append(fn/(fn+tp))
        FPR.append(fp/(fp+tn))

    x = range(1, len(ERR)+1)
    plt.plot(x, ERR, label='Error rate')
    plt.plot(x, FNR, label='False negative rate')
    plt.plot(x, FPR, label='False positive rate')
    plt.scatter(x, ERR)
    plt.scatter(x, FNR)
    plt.scatter(x, FPR)
    plt.xlabel('Number of layers')
    plt.ylabel('Error rate')
    plt.legend()
    plt.savefig(f'{save_path}')
    plt.close()

def plot_feature_importances(models: Any, img_path: str, save_path:str) -> None:
    #Given a model
    image = cv2.imread(img_path, 0)
    fig, axes = plt.subplots(1, 7)
    for idx, ax in enumerate(axes.ravel()):
        feature_coords = models[idx][3]
        img = image
        img = draw_haar_like_feature(img, 0, 0, 24, 24, feature_coord = feature_coords[3:6])
    
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

    _ = fig.suptitle('Subset of selected features for each layer')
    plt.savefig(f'{save_path}')
    plt.close()

def cascade_scan(models: Any, scale: float, delta: float, size_subwindow: int, img: Image) -> list:
    """Image scan with cascade classifier applied on each subwindow, returning positively classified subwindows

    Args:
        models (Any): Set of models for each layer of the cascade
        img (np.ndarray): Input grayscale image to scan
        scale (float): Scaling rate of subwindow size
        delta (float): Pixel step size for each subwindow
        size_subwindow (int): Initial subwindow size (which always will be downscaled to 24x24)

    Returns:
        list: Positively classified subwindows and their integral image
    """
    img_ii_positive_list = []
    subwindow_bounds_list = []
    w, h = img.size
    size_img = min(w, h)
    y_pos = 0
    x_pos = 0
    y_pred_avg = np.zeros(7)
    
    while size_subwindow <= size_img:
        while y_pos+size_subwindow <= h:
            while x_pos+size_subwindow <= w:
                #Get subwindow and resize
                subwindow = img.crop((x_pos, y_pos, x_pos+size_subwindow, y_pos+size_subwindow))
                subwindow_bounds = [x_pos, y_pos, x_pos+size_subwindow, y_pos+size_subwindow]
                
                #Resize and standardize
                subwindow = subwindow.resize((24, 24))
                subwindow = np.array(subwindow)
                subwindow = subwindow - np.mean(subwindow)
                subwindow_std = np.std(subwindow)

                #If std is too low, then discard
                if subwindow_std < 1:
                    x_pos += int(delta*scale*size_subwindow/24)
                    continue
                subwindow = 255*subwindow/subwindow_std

                #Integral image
                subwindow_ii = integral_image(subwindow)

                #Cascade layers
                for i, model in enumerate(models):
                    SC = model[0]
                    threshold = model[1]
                    feature_types = model[2]
                    feature_coords = model[3]

                    features = haar_like_feature(subwindow_ii, 0, 0, 24, 24, feature_type=feature_types, feature_coord=feature_coords)
                    y_pred = 1*(SC.predict_proba(features.reshape(1,-1))[:, 1] >= threshold)
                    y_pred_avg[i] = y_pred

                    #If it fails in a layer, then discard it
                    if y_pred == 0:
                        #y_pred_avg = np.zeros(7)
                        break
                    elif model == models[-1]:
                        #If it passes through cascade, then save it
                        img_ii_positive_list.append(subwindow_ii)
                        subwindow_bounds_list.append([subwindow_bounds, np.mean(y_pred_avg)])

                x_pos += delta
            y_pos += delta
            x_pos = 0
        size_subwindow *= scale
        x_pos = 0
        y_pos = 0
        size_subwindow = int(size_subwindow)
    return img_ii_positive_list, subwindow_bounds_list

def cascade_classifier(X: np.ndarray, models: list) -> np.ndarray:
    # Input:
    # features: Array of features for an image
    # models: List of models for each layer in the cascade
    
    # Returns:
    # 1 if all layers predicts over the set threshold
    # 0 otherwise
    y = np.ones(len(X), dtype=int)

    for model in models:
        #Load strong classifier and its threshold
        SC = model[0]
        threshold = model[1]
        feature_type = model[2]
        feature_coord = model[3]
        feature_idx = model[4]

        print(feature_idx)
        #Calculate features of subset for each SC
        #features = haar_like_feature(subwindow_ii, 0, 0, 24, 24, feature_type=feature_types, feature_coord=feature_coords)
        X_pos_samples = X[y==1]
        y_pred = 1*(SC.predict_proba(X_pos_samples[:, feature_idx])[:, 1] >= threshold)
        
        y_idx = np.where(y == 1)
        #y_pred = 1*(SC.predict_proba(X[y == 1])[:, 1] >= threshold)
        y[y_idx] = y_pred
    return y

def read_model_file(file_path:str) -> list:
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
   
def read_feature_file(path_read: str) -> list:
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