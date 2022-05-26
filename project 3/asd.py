from typing import Any
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from classify import read_model_file
from skimage.transform import integral_image
from skimage.feature import haar_like_feature, haar_like_feature_coord
from PIL import Image, ImageOps
import cProfile, re

def main():
    models = read_model_file('abclf.pkl')
    img = Image.open('../negative_imgs/images/0000051.jpg')
    img = ImageOps.grayscale(img)
    img = img.crop((0, 0, 24, 24))
    
    # feature_coord, feature_type = haar_like_feature_coord(24,24)
    # img_ii = integral_image(img)
    
    # # for i, tree in enumerate(models[0][0]):
    # #     print(tree.tree_.feature[0])
    # #     print(features)
    
    
    scale = 1.25
    delta = 1
    # #cProfile.run(cascade_scan(models, img, scale, delta))
    # with cProfile.Profile() as pr:
    #     positives = cascade_scan(models, img, scale, delta)
    # pr.print_stats()
    positives = cascade_scan(models, img, scale, delta)
    

def cascade_scan(models: Any, img: np.ndarray, scale: float, delta: float):
    feature_coord_SC = np.empty(5, dtype=object)
    feature_type_SC = np.empty(5, dtype=object)
    
    feature_coord, feature_type = haar_like_feature_coord(24, 24)
    img_ii = integral_image(np.array(img))
    
    #Extract feature type and coord for given index in model
    #For every strong classifier SC, iterate through its weak classifiers WC
    for i, model in enumerate(models[:-1]):
        SC = model[0]
        feature_coord_WC = np.empty(np.count_nonzero(SC.feature_importances_ != 0), dtype=object)
        feature_type_WC = np.empty(np.count_nonzero(SC.feature_importances_ != 0), dtype=object)
        for j, WC in enumerate(SC):
            feature_idx = WC.tree_.feature[0]
            WC.tree_.feature[0] = j
            
            #Feature type
            feat_t = np.empty(len([feature_type[feature_idx]]), dtype=object)
            feat_t[:] = feature_type[feature_idx]
            feature_type_WC[j] = feat_t[0]
            #Feature coordinate
            feat_c = np.empty(len([feature_coord[feature_idx]]), dtype=object)
            feat_c[:] = [feature_coord[feature_idx]]
            feature_coord_WC[j] = feat_c[0]
        
        feature_coord_SC[i] = feature_coord_WC
        feature_type_SC[i] = feature_type_WC            

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
                subwindow_ii = integral_image(np.array(subwindow))
                #Cascade
                for i, model in enumerate(models):
                    SC = model[0]
                    threshold = model[1]
                    features = haar_like_feature(subwindow_ii, 0, 0, 24, 24, feature_type=feature_type_SC[i], feature_coord=feature_coord_SC[i])
                    features = features.reshape(1, -1)
                    y_pred = 1*(SC.predict_proba(features)[:, 1] >= threshold)

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