#%%

# CVAT file analysis
import numpy as np
from matplotlib import image
import os
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import skimage.io as io

# from Image_extract_test import Json_root

Json_root = 'C:/Users/Students/Downloads/Test_Sensor_trees/images/instances_default_neww.json'
# img_root = 'C:/Users/Students/Downloads/Test_Sensor_trees/images'
# img_root_idx = 'C:/Users/Students/Downloads/Test_Sensor_trees/images/NDVI_idx.tif'

# Json_root = ''

img_root = 'C:/Users/Students/Box/Research/IoT4ag/Project_ Water Stress' \
    + '/Data Collection/Pistachio/Multispectral/'

Dict = {'T1': '06_07_22', 'T2': '06_21_22', 'T3': '07_05_22', 'T4': '07_13_22', \
            'T5': '07_26_22', 'T6': '08_02_22', 'T7': '08_12_22'}

extn = '/index_map'

TRAIN_IMAGES_DIRECTORY = img_root
TRAIN_ANNOTATIONS_PATH = Json_root

coco = COCO(TRAIN_ANNOTATIONS_PATH)
img_ids = coco.getImgIds()
annotation_ids = coco.getAnnIds(img_ids)
annotations = coco.loadAnns(annotation_ids)

def index_capture(idx_file, idx_type):
    print("{} is being extracted for all trees...".format(idx_file))
    index_all = list([])
    for i in range(len(annotations)):
        print(i)
        # image_id = coco.loadImgs(annotations[i]["id"])[0]   
        # im_id = image_id["id"]
        im_id = i+1
        entity_id = annotations[i]["category_id"]
        entity = coco.loadCats(entity_id)[0]["name"]
        print("idx={}: image {}: {}".format(i, im_id ,entity))

        image_path = img_root + Dict[entity] + extn + idx_file  
        # print(image_path)
        # image_meta = coco.loadImgs(annotations[i]["image_id"])[0]   
        # image_path = os.path.join(TRAIN_IMAGES_DIRECTORY, image_meta["file_name"])

        masks = coco.annToMask(annotations[i])
        segmentation = np.where(masks == True)
        print(len(segmentation[0]))

        I = io.imread(image_path)
        index_array = I[(segmentation[0],segmentation[1])]

        temp_dict = {'idx': i, 'image_id': im_id, 'Test_Number': entity, 'pixel_array': index_array}
        index_all.append([temp_dict])

        # print(Index_a)
        # return Index_array
    return index_all
        
NDVI = index_capture('/NDVI.tif', 'NDVI')



# I = io.imread(image_path)
# plt.imshow(I)
# coco.showAnns(annotations)
# plt.show()

#%%

# from PIL import Image
# import cv2

# # Get the masks
# masks = coco.annToMask(annotations[0])

# # Pick an item to mask
# item_mask = masks
# im = cv2.imread(image_path)

# # Get the true bounding box of the mask (not the same as the bbox prediction)
# segmentation = np.where(item_mask == True)
# x_min = int(np.min(segmentation[1]))
# x_max = int(np.max(segmentation[1]))
# y_min = int(np.min(segmentation[0]))
# y_max = int(np.max(segmentation[0]))
# print(x_min, x_max, y_min, y_max)

# # Create a cropped image from just the portion of the image we want
# cropped = Image.fromarray(im[y_min:y_max, x_min:x_max, :])

