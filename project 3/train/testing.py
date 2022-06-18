from utils import *
from PIL import Image, ImageOps
import time

def test(x):
    x = x+1
    return print(x)

if __name__ == "__main__":
    models = read_model_file('abclf_4.pkl')
    models = models[-1]
    img = Image.open('../scan_test/scan_img3.jpg')
    img = ImageOps.grayscale(img)
    scale = 2
    delta = 3

    add_samples(models, img, scale, delta)
    test(5)
    time.sleep(5)
    add_samples(models, img, scale, delta)