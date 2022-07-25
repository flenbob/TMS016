from skimage.feature import haar_like_feature
from skimage.transform import integral_image
from PIL import Image, ImageOps
import numpy as np
import cv2
import os
import concurrent.futures
import pickle
import functools
import constants as c

#SAMPLE COLLECTING FUNCTIONS
def pos_samples(path_read: str, path_write: str, N=float('inf')) -> None:
    """Create 162336 haar-like features for N positive samples and store in path_write.pkl

    Args:
        path_read (str): Folder containing positive samples
        path_write (str): Folder where .pkl file is saved
        N (int, optional): Number of positive samples for which to generate features. Defaults to all samples.
    """
    #List imgs path
    imgs = os.listdir(path_read)
    imgs = [f'{path_read}/{img}' for img in imgs]
    imgs = imgs[0:min(N, len(imgs))]

    #Multiprocess splits of samples
    samples = []
    imgs = np.array_split(imgs, os.cpu_count())

    with concurrent.futures.ProcessPoolExecutor() as executor:
        sample_splits = executor.map(pos_features, imgs)

        for sample_split in sample_splits:
            for sample in sample_split:
                samples.append(sample)
    
    #Save samples
    with open(f'{path_write}.pkl', 'wb') as f:
        print(f'Saved {len(samples)} samples to {path_write}.pkl')
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

def neg_samples(path_read: str, path_write: str, N=float('inf')) -> None:
    """Create 162336 haar-like features for N negative samples and store in path_write.pkl

    Args:
        path_read (str): Folder containing positive samples.
        path_write (str): Folder where .pkl file is saved.
        N (int, optional): Number of positive samples for which to generate features. Defaults to all samples.
    """
    #List images path
    imgs = os.listdir(path_read)
    imgs = [f'{path_read}/{img}' for img in imgs]

    #Iterate though images
    samples = []
    for img in imgs:
        #Read image
        img = Image.open(img)
        img = ImageOps.grayscale(img)
        
        #Split image into sections
        w, h = img.size
        N_cpu = os.cpu_count()

        if w >= h:
            pos_shift = [(c.DELTA*i, 0, N_cpu*c.DELTA, c.DELTA)  for i in range(N_cpu)]
        else:
            pos_shift = [(0, c.DELTA*i, c.DELTA, N_cpu*c.DELTA) for i in range(N_cpu)]

        #Multiprocess each section of image
        partial_args = functools.partial(neg_features, img)
        with concurrent.futures.ProcessPoolExecutor() as pool:
            sample_splits = list(pool.map(partial_args, pos_shift))

        for sample_split in sample_splits:
            for sample in sample_split:
                samples.append(sample)
        
        print(f'{len(samples)} collected samples...')
        
        #Stop reading more images and save samples to .pkl file
        if len(samples) >= N:
            break
    
    with open(f'{path_write}.pkl', 'wb') as f:
        print(f'Saved {len(samples)} samples to {path_write}.pkl')
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

#MULTIPROCESSING FUNCTIONS
def pos_features(imgs_path: str) -> np.ndarray:
    """Get haar-like features of a set of images which are normalized

    Args:
        imgs (str): List of image paths

    Returns:
        np.ndarray: Features for all images in imgs
    """

    features = []
    #Read image, downsample to 24x24, normalize, calculate integral image and then extract haar like features
    for img_path in imgs_path:
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (24,24), interpolation=cv2.INTER_AREA)
        img = 255*(img - np.mean(img))/np.std(img)
        img_ii = integral_image(img)
        feature = haar_like_feature(img_ii, 0, 0, 24, 24)
        features.append(feature)
    return features

def neg_features(img: Image, pos_shift: tuple[int]) -> np.ndarray:
    size_subwindow = 24
    w, h = img.size
    size_img = min(w,h)
    x_pos = pos_shift[0]
    y_pos = pos_shift[1]
    features = []
    while size_subwindow <= size_img:
        while y_pos+size_subwindow <= h:
            while x_pos+size_subwindow <= w:
                #Crop subwindow of image
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

                #Integral image and haar feature
                subwindow = 255*(subwindow - np.mean(subwindow))/subwindow_std
                subwindow_ii = integral_image(subwindow)
                feature = haar_like_feature(subwindow_ii, 0, 0, 24, 24)
                features.append(feature)
                x_pos += pos_shift[2]
            y_pos += pos_shift[3]
            x_pos = pos_shift[0]
        size_subwindow *= c.SCALE
        size_subwindow = int(size_subwindow)
        x_pos = pos_shift[0]
        y_pos = pos_shift[1]
    return features