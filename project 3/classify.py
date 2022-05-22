import numpy as np
import pickle
import h5py
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main():


    #Read trained model and features
    models = read_model_file('abclf.pkl')
    X_pos, y_pos = read_feature_file('../positive_imgs.hdf5')
    X_neg, y_neg = read_feature_file('../negative_imgs_verify.hdf5')
    
    #Run verification on positive images which were on trained on:
    X_pos = X_pos[600:956]
    y_pos = y_pos[600:956]
    y = np.concatenate((y_pos, y_neg))
    X = np.concatenate((X_pos, X_neg))

    #Classify each image (,n_f)-array of haar features
    y_pred = []
    for img in X:
        y_pred.append(cascade_classifier(img, models))
    
    #Generate confusion matrix
    cm = confusion_matrix(y, y_pred, labels = [1,0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

def cascade_classifier(img: np.ndarray, models: list) -> int:
    # Input:
    # features: Array of features for an image
    # models: List of models for each layer in the cascade
    
    # Returns:
    # 1 if all layers predicts over the set threshold
    # 0 otherwise
    for model in models:
        #Load strong classifier and its threshold
        img = img.reshape(1,-1)
        SC = model[0]
        threshold = model[1]
        #print(SC.predict_proba(img))
        y_pred = 1*(SC.predict_proba(img)[:,-1] >= threshold)
        if y_pred == 0:
            return 0
    return 1

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
   
def read_feature_file(path_read: str) -> tuple[np.ndarray, np.ndarray]:
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