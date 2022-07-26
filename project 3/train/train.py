import numpy as np
import pickle
from functions import *
import constants as c

def train():
    #Load positive and negative samples
    X_pos = samples_load(c.POS_SAMPLES_PATH)[0:300]
    X_neg = samples_load(c.NEG_SAMPLES_PATH)[0:500]
    y_pos = np.array(len(X_pos)*[1])
    y_neg = np.array(len(X_neg)*[0])
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    print('Files loaded')
    
    #Initialize parameters
    n_wc = c.WC_INIT
    fpr = c.FPR_INIT
    layer_cnt = 0
    last_visited = None
    
    #Fit a model for each layer and save to .pkl file
    models = []
    with open(c.MODEL_PATH, 'wb') as f:
        while fpr > c.FPR_TERMINATE:
            #Fit model and save it to .pkl file
            print(f'-----------------\nLayer {layer_cnt}: Fitting model with {n_wc} WC:s')
            layer_cnt += 1
            model = model_fit(X, y, n_wc)
            models.append(model)
            pickle.dump(model, f)

            #Update number of weak classifiers
            n_wc = model.clf.n_estimators

            #Delete false negative and true negative samples
            X, y = samples_del(model, X, y)

            #Add new negative samples if requirements are not met
            X, y, last_visited = samples_add(models, X, y, last_visited)

            #Update FPR to check for termination criterion
            y_pred = 1*(model.clf.predict_proba(X[:, model.feats_idx])[:,-1] >= model.threshold)
            tn, fp, _, _ = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
            fpr = fp/(fp+tn)

if __name__ == "__main__":
    train()