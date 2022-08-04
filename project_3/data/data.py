from functions import *
import constants as c

def main():
    #Get positive and negative samples saved to .pkl file
    #pos_samples(c.POS_PATH_READ, c.POS_PATH_WRITE, c.POS_N_SAMPLES)
    #neg_samples(c.NEG_PATH_READ, c.NEG_PATH_WRITE, c.NEG_N_SAMPLES)

    pos_samples('../positive_imgs_validate', '../data/pos_samples_validate', N=600)
    neg_samples('../negative_imgs_validate', '../data/neg_samples_validate', N=600)

if __name__ == "__main__":
    main()