from typing import Any
import numpy as np
from classify import read_model_file
from skimage.transform import integral_image
from skimage.feature import haar_like_feature, haar_like_feature_coord
from PIL import Image, ImageOps
import cProfile, re

def main():
    models = read_model_file('abclf.pkl')
    
    # print(models[0][0])
    # print(type(models[0][0]))
    # print(models[0][0][1].tree_.feature)
    #img = cv2.imread('../negative_imgs/images/0000051.jpg', cv2.IMREAD_GRAYSCALE)
    # img = Image.open('../negative_imgs/images/0000051.jpg')
    # img = ImageOps.grayscale(img)
    # img = img.crop((0, 0, 64, 64))
    # scale = 1.25
    # delta = 1
    # #cProfile.run(cascade_scan(models, img, scale, delta))
    # with cProfile.Profile() as pr:
    #     positives = cascade_scan(models, img, scale, delta)
    # pr.print_stats()

def cascade_scan(models: Any, img: np.ndarray, scale: float, delta: float):
    
    features_idx_list = []
    for model in models:
        SC = model[0]
        n_features = np.count_nonzero(SC.feature_importances_ != 0)
        features_idx = np.argsort(SC.feature_importances_)[::-1][0:n_features]
        features_idx_list.append(features_idx)
        
    
    #Split image into subwindows
    feature_array = np.empty(shape=(1,162336))
    feature_types_set = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']
    feature_coord, feature_types = haar_like_feature_coord(24, 24, feature_type=feature_types_set)
    
    positives = []
    h, w = img.size
    size_img = min(h,w)
    size_subwindow = 24
    
    while size_subwindow <= size_img:
        y_pos = 0
        while y_pos+size_subwindow <= h:
            x_pos = 0
            while x_pos+size_subwindow <= w:
                #Get subwindow and resize
                subwindow = img.crop((x_pos, y_pos, x_pos+size_subwindow, y_pos+size_subwindow))
                #subwindow = img[x_pos:(x_pos+size_subwindow), y_pos:(y_pos+size_subwindow)]
                subwindow_bounds = subwindow
                #Standardize image
                std = np.std(subwindow)
                
                if std < 1:
                    print('std < 1')
                    continue
                mean = np.mean(subwindow)
                subwindow = (subwindow - mean)/std
                
                #Integral image
                subwindow_ii = integral_image(subwindow)
                #Cascade
                for i, model in enumerate(models):
                    SC = model[0]
                    threshold = model[1]
                    features = haar_like_feature(subwindow_ii, 0, 0, 24, 24, feature_coord=feature_coord[features_idx_list[i]], feature_type=feature_types[features_idx_list[i]])
                    feature_array[:,features_idx_list[i]] = features
                    y_pred = 1*(SC.predict_proba(feature_array)[:, 1] >= threshold)

                    if y_pred == 0:
                        break
                    elif model == models[-1]:
                        #If it passes through cascade, then save it
                        positives.append((subwindow, subwindow_bounds))

                x_pos += int(delta*scale*size_subwindow/24)
            y_pos += int(delta*scale*size_subwindow/24)
        size_subwindow *= scale
        size_subwindow = int(size_subwindow)
    return positives

if __name__ == "__main__":
    main()