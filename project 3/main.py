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

def main():
    n_f = 162336
    haar_f = np.empty((4495,n_f), float)
    data_path = 'data/faces/'
    for idx, image_path in enumerate(os.listdir(data_path)):
        print(image_path)
        img = cv2.imread(data_path+image_path, 0)
        img_ii = integral_image(img)
        feature = haar_like_feature(img_ii, 0, 0, 24, 24)
        haar_f[idx] = feature
        
    df = pd.DataFrame(haar_f)
    df['class'] = 1
    df.to_hdf('data/data.h5', key='positive_images', mode='w')

if __name__ == "__main__":
    main()