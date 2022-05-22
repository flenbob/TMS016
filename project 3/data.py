from cv2 import meanStdDev
from skimage.feature import haar_like_feature
from skimage.transform import integral_image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from math import sqrt, sin, cos
import cv2
import os
import h5py
from PIL import Image, ImageOps

def main():
    #Create face images of dim pixels
    #Copy face_coords.txt and place in same dir as folders '2002' and '2003'
    #where face_coords.txt are all .txt files concatenated
    path_read = '../negative_images_verify'
    path_write = '../negative_imgs_verify.hdf5'
    dim = (24,24)
    n_f = 162336
    dset_size = 500
    n_dsets = 10

    img2haar_features(path_read, path_write, n_f, 357, 'negative')
    #neg_img2haar_features(path_read, path_write, n_f, dset_size, n_dsets)
    #positive_images_school_photos(path_read, path_write, dim)
    #positive_images_FDDB(path_read, path_write, dim)
    #negative_images_FDDB('../negative_dataset/images', '../negative_images', dim)

def neg_img2haar_features(path_read: str, path_write: str, n_f: int, dset_size: int, n_dsets: int) -> None:
    #Load negative images and create file partitioned into datasets of size dset_size (used for generating FP:s in adaboost training)
    X = np.empty((dset_size, n_f))
    dset_cnt = 0
    idx = 0

    with h5py.File(path_write, 'w') as f:
        for image_id in os.listdir(path_read):
            img = cv2.imread(f'{path_read}/{image_id}', 0)
            img_ii = integral_image(img)
            X[idx] = haar_like_feature(img_ii, 0, 0, 24, 24)
            idx += 1
            print(idx)
            if (idx) % dset_size == 0:
                dset_cnt += 1
                idx = 0
                f.create_dataset(f'negative_dataset_{dset_cnt}', data=X)
                X = np.empty((dset_size, n_f))
                print(f'Dataset {dset_cnt} created')

            if n_dsets == dset_cnt:
                return
            
def img2haar_features(path_read: str, path_write: str, n_f: int, n_img: int, img_type: str) -> None:
    #X: Haar feature for each image
    #y: label for each image
    X = np.empty((n_img, n_f))
    file = h5py.File(f'{path_write}', 'w')

    if img_type == 'positive':
        y = np.array([1]*n_img)
    elif img_type == 'negative':
        y = np.array([0]*n_img)
    else:
        return print('Wrong input')
    
    #Calculate haar feature and break loop if out of bounds
    for idx, image_id in enumerate(os.listdir(path_read)):
        print(idx)
        img = cv2.imread(f'{path_read}/{image_id}', 0)
        img_ii = integral_image(img)
        feature = haar_like_feature(img_ii, 0, 0, 24, 24)
        X[idx] = feature
        # if idx + 1 == n_img: 
        #     break

    #Save to h5.file
    file.create_dataset('features', data=X)
    file.create_dataset('labels', data=y)
    file.close()

def negative_images_FDDB(path_read: str, path_write: str, dim: tuple[int, int]) -> None:
    #Find dimension of each image, get all possible 24x24 crops
    for image_path in os.listdir(path_read):
        print(f'{path_read}/{image_path}')
        img = cv2.imread(f'{path_read}/{image_path}', 0)
        h, w = img.shape

        m = int(h/dim[0])
        n = int(w/dim[1])
        norm = np.zeros(dim)

        for i in range(m):
            for j in range(n):
                img_cropped = img[(dim[0]*i):(dim[0]+dim[0]*i), (dim[1]*j):(dim[1]+dim[1]*j)]
                #img_cropped = cv2.normalize(img_cropped, norm, 0, 255, cv2.NORM_MINMAX)

                _, std = cv2.meanStdDev(img_cropped)
                if std > 15:
                    cv2.imwrite(f'{path_write}/{image_path}_{i}_{j}.png', img_cropped)

def positive_images_school_photos(path_read: str, path_write: str, dim: tuple[int, int]) -> None:
    for image_path in os.listdir(path_read):
        img = Image.open(f'{path_read}/{image_path}')
        img = ImageOps.grayscale(img)
        img = img.resize(dim, Image.ANTIALIAS)
        img = np.array(img)
        norm = np.zeros(dim)
        img = cv2.normalize(img, norm, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f'{path_write}/{image_path}.png', img)

def positive_images_FDDB(path_read: str, path_write: str, dim: tuple[int, int]) -> None:
    # Read text document in filepath path_read
    # for each image:
    #   for each face:
    #       crop face, resize to dim pixels and save to filepath path_write
    
    face_coords = open(f'{path_read}/face_coords.txt','r')
    lines = face_coords.readlines()
    for idx, line in enumerate(lines):
        #If image on current line
        if line[0:4] == ('2002' or '2003'):
            path = f'{path_read}/{line[:-1]}.jpg'
            print(f'{path_read}/{line[:-1]}.jpg')
            img = cv2.imread(path, 0)
            n_faces = int(lines[idx+1])
            if n_faces > 3: #If too many faces in same image, then skip it (bad results in those cases)
                continue
            masks = lines[idx+2:idx+2+n_faces]
            img_id = line[line.index('_')+1:-1]
        
            #Crop, resize and save all faces in image
            for idx, mask in enumerate(masks):
                masks[idx] = [float(i) for i in mask.split()[:-1]]
                x, y, w, h = ellipse2rect(masks[idx])
                img_crop = img[y:y+h, x:x+w]
                new_img = cv2.resize(img_crop, dim, interpolation = cv2.INTER_AREA)
                
                #Normalize image
                norm = np.zeros(dim)
                new_img = cv2.normalize(new_img, norm, 0, 255, cv2.NORM_MINMAX)
                
                #Save image
                cv2.imwrite(f'{path_write}/face_img_{img_id}_{idx}.png', new_img)
         
def ellipse2rect(ellipse_mask) -> list:
    #Convert ellipse mask to rectangle
    a, b, theta, x_o, y_o = ellipse_mask
    
    x = sqrt(a**2*cos(theta)**2+b**2*sin(theta)**2)
    y = sqrt(a**2*sin(theta)**2+b**2*cos(theta)**2)
    m = (x+y)/2
    
    #Get top left (x,y) and width, height (w, h)
    x_tl = int(abs(x_o-m))
    y_tl = int(abs(y_o-m))
    w = int(2*m)
    h = int(2*m)
    
    return x_tl, y_tl, w, h

    
if __name__ == "__main__":
    main()