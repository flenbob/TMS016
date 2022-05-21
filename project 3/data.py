import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from math import sqrt, sin, cos
import cv2
import os
from PIL import Image, ImageOps

def main():
    #Create face images of dim pixels
    #Copy face_coords.txt and place in same dir as folders '2002' and '2003'
    #where face_coords.txt are all .txt files concatenated
    path_read = '../faces_dataset/M'
    path_write = '../faces_dataset/faces_cropped'
    dim = (24,24)
    
    #positive_images_school_photos(path_read, path_write, dim)
    #positive_images_FDDB(path_read, path_write, dim)
    negative_images_FDDB('../negative_imgs/images', '../neg_imgs', dim)


def negative_images_FDDB(path_read: str, path_write: str, dim: tuple[int, int]) -> None:
    #Crop 2 square regions from a given image and then resize to 24x24
    for image_path in os.listdir(path_read):
        print(f'{path_read}/{image_path}')
        #img = cv2.imread(f'{path_read}/{image_path}', 0)
        img = Image.open(f'{path_read}/{image_path}')
        img = ImageOps.grayscale(img)
        
        #Resize images
        img = img.resize(dim, Image.ANTIALIAS)
        img = np.array(img)
        
        #Normalize images
        norm = np.zeros(dim)
        img = cv2.normalize(img, norm, 0, 255, cv2.NORM_MINMAX)

        #Save images
        cv2.imwrite(f'{path_write}/{image_path}.png', img)

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
                img = cv2.normalize(new_img, norm, 0, 255, cv2.NORM_MINMAX)
                
                #Save image
                cv2.imwrite(f'{path_write}/face_img_{img_id}_{idx}.png', img)
         
def ellipse2rect(ellipse_mask):
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