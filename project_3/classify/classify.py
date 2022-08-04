import numpy as np
from functions import *
import constants as c
import cv2
from sklearn.metrics import confusion_matrix

def main():
    #Load model and files
    models = read_model_file('ABCLF_best_2.pkl')
    X_pos = samples_load(c.POS_SAMPLES_PATH)
    X_neg = samples_load(c.NEG_SAMPLES_PATH)
    y_pos = np.array(len(X_pos)*[1])
    y_neg = np.array(len(X_neg)*[0])
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    print('Validation datasets loaded')

    # #Perform cascade classification on dataset
    # y_pred = cascade_classifier(models, X)

    # #Evaluate performance
    # tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    # fnr, fpr = fn/(fn+tp), fp/(fp+tn)
    # print(f'fnr = {fnr}, fpr = {fpr}')

    #'../data/big_photo.jpg'
    img = Image.open('../scan_test/scan_img1.jpg')
    img = ImageOps.grayscale(img)
    boxes = image_scan(models, img, pos_shift=(0, 0, c.DELTA, c.DELTA))
    with open(f'boxes.pkl', 'wb') as f:
        pickle.dump(boxes, f, protocol=pickle.HIGHEST_PROTOCOL)

def test():
    with open('boxes.pkl', 'rb') as f:
        boxes = pickle.load(f)
    
    img = cv2.imread('../scan_test/scan_img1.jpg')

    for box in boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)
    cv2.imwrite('boxes_img.jpg', img)

        
if __name__ == "__main__":
    test()