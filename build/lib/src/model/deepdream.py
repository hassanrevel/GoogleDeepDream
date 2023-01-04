from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from src.Model.utils import dream
from src.Model.vgg import vgg
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-ip", "--image_path",
    type=str,
    required=False
)
arg = parser.parse_args()

# Doing prediction on image and displaying it
img = Image.open(
    arg.image_path
)

orig_size = np.array(img.size)
new_size = np.array(img.size)*0.5
img = img.resize(new_size.astype(int))
layer = list( vgg.features.modules() )[27]

# Execute our Deep Dream Function
img = dream(img, vgg, layer, 20, 1)

img = img.resize(orig_size)
plt.figure(figsize = (10 , 10))
plt.imshow(img)
plt.show()