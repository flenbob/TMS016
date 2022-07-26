from functions import *
import constants as c

def run():
    #Get positive and negative samples saved to .pkl file
    pos_samples(c.POS_PATH_READ, c.POS_PATH_WRITE, c.POS_N_SAMPLES)
    neg_samples(c.NEG_PATH_READ, c.NEG_PATH_WRITE, c.NEG_N_SAMPLES)

if __name__ == "__main__":
    run()