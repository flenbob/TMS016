import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, sin, cos
import cv2
import os

def main():
    #Create face images of dim pixels
    path_read = 'data/face_coords.txt'
    path_write = 'data/faces'
    dim = (24,24)
    generate_face_data(path_read, path_write, dim)
    
def generate_face_data(path_read: str, path_write: str, dim: tuple[int, int]) -> None:
    # Read text document in path_read
    # for each image:
    #   for each face:
    #       crop face, resize to dim pixels and save to path_write
    
    img_cnt = 0
    face_coords = open(path_read,'r')
    lines = face_coords.readlines()
    #Read through lines of text document
    for idx, line in enumerate(lines):
        #If image on current line
        if line[0:4] == ('2002' or '2003'):
            
            path = 'data/'+line[:-1]+'.jpg'
            img = cv2.imread(path, 0)
            n_faces = int(lines[idx+1])
            masks = lines[idx+2:idx+2+n_faces]
            
            #Crop, resize and save all faces in image
            for idx, mask in enumerate(masks):
                img_cnt += 1
                masks[idx] = [float(i) for i in mask.split()[:-1]]
                x, y, w, h = ellipse2rect(masks[idx])
                img_crop = img[y:y+h, x:x+w]
                dim = (24,24)
                new_img = cv2.resize(img_crop, dim)
                cv2.imwrite(f'{path_write}/face_img_{img_cnt}.png', new_img)
         
def ellipse2rect(ellipse_mask):
    #Convert ellipse mask to rectangle
    a, b, theta, x_o, y_o = ellipse_mask
    x = sqrt(a**2*cos(theta)**2+b**2*sin(theta)**2)
    y = sqrt(a**2*sin(theta)**2+b**2*cos(theta)**2)
    
    #Get top left (x,y) and width, height (w, h)
    x_tl = int(abs(x_o-x))
    y_tl = int(abs(y_o-y))
    w = int(2*x)
    h = int(2*y)
    
    return x_tl, y_tl, w, h
    
if __name__ == "__main__":
    main()