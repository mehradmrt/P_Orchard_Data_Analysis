# %%

import numpy as np
import cv2 
from PIL import Image

root = 'C:/Users/Students/Box/IoT-4Ag -Data/Multispectral_UAV/Pistachio/Pistachio_8-12-22_4cm/map'
root_color = root + '/index_map_color/GNDVI_local.tif'
root_idx = root + '/index_map/GNDVI.tif'

# img_color = cv2.imread(root_color)
# img_idx = cv2.imread(root_idx)

# cv2.imshow('Image',img_color)



im = Image.open(root_color) # Can be many different formats.
pix = im.load()
print(im.size)  # Get the width and hight of the image for iterating over
print(pix[1,1])  # Get the RGBA Value of the a pixel of an image


    # %%
