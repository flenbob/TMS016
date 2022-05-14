from skimage.feature import haar_like_feature
from skimage.transform import integral_image
from skimage import color
from skimage import io
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from os import listdir
from sklearn.model_selection import train_test_split

def main():
    n_f = 162336
    n_img = 4495
    img_type = 'positive'
    path_read = '../FDDB/faces'
    path_write = '../FDDB'
    img2haar_features(path_read, path_write, n_f, n_img, img_type)
    
def img2haar_features(path_read, path_write, n_f, n_img, img_type):
    #Get haar features for each of the 24 x 24 images (1 = positive, 0 = negative)
    # and save to h5 file
    if img_type == 'positive':
        cl = 1
    elif img_type == 'negative':
        cl = 0
    else:
        return print('img_type must be either ''positive'' or ''negative''')
    
    haar_f = np.empty((n_img,n_f), float)
    for idx, image_id in enumerate(os.listdir(path_read)):
        img = cv2.imread(path_read+image_id, 0)
        img_ii = integral_image(img)
        feature = haar_like_feature(img_ii, 0, 0, 24, 24)
        haar_f[idx] = feature
        
    df = pd.DataFrame(haar_f)
    df['class'] = cl
    df.to_hdf(f'{path_write}/{img_type}.h5', key=img_type, mode='w')

if __name__ == "__main__":
    main()