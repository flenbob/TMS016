import numpy as np
import pickle
from functions import *
import constants as c

def train():
    #Load positive and negative samples
    X_pos = samples_load(c.POS_SAMPLES_PATH)
    X_neg = samples_load(c.NEG_SAMPLES_PATH)
    y_pos = np.array(len(X_pos)*[1])
    y_neg = np.array(len(X_neg)*[0])
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    print('Files loaded')
    
    #Initialize parameters
    n_wc = c.WC_INIT
    layer_cnt = 0
    img_list = None
    fpr_acc = 1
    
    #Fit a model for each layer and save to .pkl file
    models = []
    with open(c.MODEL_PATH, 'wb') as m:
        while fpr_acc > c.FPR_TERMINATE:
            #Fit model and save it to .pkl file
            print(f'-----------------\nLayer {layer_cnt}: Fitting model with {n_wc} WC:s')
            model, n_wc = model_fit(X, y, n_wc)
            models.append(model)
            pickle.dump(model, m)
            print(f'Layer {layer_cnt}: Successfully fitted model with {n_wc} WC:s!')
            layer_cnt += 1

            #Update accumulated fnr and fpr
            _, fpr = model_error(model, X, y)
            if len(models) > 1:
                fpr_acc *= fpr
            else:
                fpr_acc = fpr
            print(f'Accumulated fpr = {fpr_acc:.7f}')
            
            #Delete fn and tn samples, then add new fp samples
            X, y = samples_del(model, X, y)
            X, y, img_list = samples_add(models, X, y, img_list)

if __name__ == "__main__":
    train()