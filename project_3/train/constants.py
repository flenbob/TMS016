##Training constants

#Weak classifiers parameters
WC_INIT = 3
WC_RATE = 0.41

#Model scoring constraints
NEG_MIN_RATIO = 0.5
FPR_INIT = float('inf')
FPR_TERMINATE = 1e-7
FPR_MAX = 0.5
FNR_MAX = 0.005

#Parameters for image scan
SCALE = 1.25
DELTA = 1

#Path to model file
MODEL_PATH = 'project_3/train/ABCLF.pkl'

#Path to negative and positive samples
NEG_SAMPLES_PATH = '../data/neg_samples.pkl'
POS_SAMPLES_PATH = '../data/pos_samples.pkl'

#Path to negative images
NEG_DSET_PATH = '../negative_imgs'