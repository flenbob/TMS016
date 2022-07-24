import numpy as np
import pickle
from utils import *
import constants as c

def train():
    #Load positive and negative samples
    X_pos = read_sample_file(c.POS_SAMPLES_PATH)
    X_neg = read_sample_file(c.NEG_SAMPLES_PATH)
    y_pos = np.array(len(X_pos)*[1])
    y_neg = np.array(len(X_neg)*[0])
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    print('Files loaded')
    
    #Fit a model for each layer and save to .pkl file
    models = []
    n_wc = c.WC_INIT
    fpr = c.FPR_INIT
    cnt = 0
    last_visited = None

    with open(c.MODEL_PATH, 'wb') as f:
        while fpr > c.FPR_TERMINATE:
            #Fit model and save it to .pkl file
            print(f'-----------------\nFitting a model for layer {cnt} with {n_wc} WC:s')
            model = fit_model(X, y, n_wc)
            n_wc = model.clf.n_estimators
            models.append(model)
            pickle.dump(model, f)

            #Delete false negative and true negative samples
            X, y = delete_samples(model, X, y)

            #Add new samples if requirements are not met
            X, y, last_visited = add_samples(models, X, y, last_visited)

            #Update FPR to check for termination criterion
            y_pred = 1*(model.clf.predict_proba(X[:, model.feats_idx])[:,-1] >= model.threshold)
            tn, fp, _, _ = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
            fpr = fp/(fp+tn)
            
            cnt += 1

if __name__ == "__main__":
    train()