from imageio import save
import numpy as np
import pickle
import h5py
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from skimage.feature import draw_haar_like_feature
from skimage.feature import haar_like_feature_coord
import cv2


def main():

    #Read trained model and features
    X_pos1, y_pos1 = read_feature_file('../positive_imgs.hdf5')
    X_neg, y_neg = read_feature_file('../negative_imgs_verify.hdf5')
    #Run verification on positive images
    X_pos1 = X_pos1[-400:-1]
    y_pos1 = y_pos1[-400:-1]
    y1 = np.concatenate((y_pos1, y_neg))
    X1 = np.concatenate((X_pos1, X_neg))

    X_pos2, y_pos2 = read_feature_file('../positive_imgs_school_test.hdf5')
    X2 = np.concatenate((X_pos2, X_neg))
    y2 = np.concatenate((y_pos2, y_neg))
    print('loaded files')

    model_path = 'abclf_2.pkl'
    img_path = '../face_test.png'
    save_path = 'data/feature_importances'

    # models = read_model_file(model_path)
    # for model in models:
    #     print(model)

    #plot_feature_importances(model_path, img_path, save_path)
    #plot_feature_importances_diagram(model_path, save_path)
    plot_error_rate(model_path, 'data/error_rate_both_asdfss', X1, y1, X2, y2)

def plot_error_rate(model_path: str, save_path: str, X1: np.ndarray, y1: np.ndarray, X2, y2) -> None:
    scores1 = []
    scores2 = []
    models = read_model_file(model_path)
    print(len(models))
    curr_model = []
    for model in models:
        print('model run')
        curr_model.append(model)
        print(curr_model)
        y_pred1 = cascade_classifier(X1, curr_model)
        tn, fp, fn, tp = confusion_matrix(y1, y_pred1).ravel()
        scores1.append((fp+fn)/(tn+fp+fn+tp))
        y_pred2 = cascade_classifier(X2, curr_model)
        tn, fp, fn, tp = confusion_matrix(y2, y_pred2).ravel()
        scores2.append((fp+fn)/(tn+fp+fn+tp))
    
    plt.plot(range(len(scores1)), scores1, label='FDDB')
    plt.scatter(range(len(scores1)), scores1)
    plt.plot(range(len(scores2)), scores2, label='School photo dataset')
    plt.scatter(range(len(scores2)), scores2)
    plt.xlabel('Number of layers')
    plt.legend()
    plt.ylabel('Error rate')
    plt.savefig(f'{save_path}')
    plt.close()

def plot_feature_importances(model_path: str, img_path: str, save_path:str) -> None:
    models = read_model_file(model_path)
    image = cv2.imread(img_path, 0)
    idx_sorted_list = []
    for model in models:
        SC = model[0]
        idx_sorted_list.append(np.argsort(SC.feature_importances_)[::-1])
    
    feature_types = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']
    feature_coord, feature_type = haar_like_feature_coord(width=image.shape[1], height=image.shape[0],
                            feature_type=feature_types)

    for i, idx_sorted in enumerate(idx_sorted_list):
        fig, axes = plt.subplots(3, 2)
        for idx, ax in enumerate(axes.ravel()):
            img = draw_haar_like_feature(image, 0, 0,
                                            image.shape[1],
                                            image.shape[0],
                                            [feature_coord[idx_sorted[idx]]])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

        _ = fig.suptitle('The most important features')
        plt.savefig(f'{save_path}_layer{i}')
        plt.close()

def plot_feature_importances_diagram(model_path: str, save_path: str) -> None:
    #Plots a diagram of feature importances vs index for each layer
    models = read_model_file(model_path)
    sorted_lists = []
    x_range_list = []
    for model in models:
        SC = model[0]
        x_range = np.count_nonzero(SC.feature_importances_ != 0)
        sorted_lists.append(np.sort(SC.feature_importances_)[::-1][0:x_range])
        x_range_list.append(x_range)

    for (x_range, sorted_list) in zip(x_range_list, sorted_lists):
        plt.plot(range(x_range), sorted_list)

    plt.show()

def cascade_classifier(X: np.ndarray, models: list) -> int:
    # Input:
    # features: Array of features for an image
    # models: List of models for each layer in the cascade
    
    # Returns:
    # 1 if all layers predicts over the set threshold
    # 0 otherwise
    y = np.ones(len(X), dtype=int)

    for model in models:
        #Load strong classifier and its threshold
        SC = model[0]
        threshold = model[1]
        #print(SC.predict_proba(img))

        y_idx = np.where(y == 1)
        y_pred = 1*(SC.predict_proba(X[y == 1])[:, 1] >= threshold)
        y[y_idx] = y_pred
    return y

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
   
def read_feature_file(path_read: str) -> list:
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