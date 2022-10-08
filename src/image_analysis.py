#%%

# CVAT file analysis
import numpy as np
from matplotlib import image
import os
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import skimage.io as io

# from Image_extract_test import Json_root

Dict = {'T1': '06_07_22', 'T2': '06_21_22', 'T3': '07_05_22', 'T4': '07_13_22', \
            'T5': '07_26_22', 'T6': '08_02_22', 'T7': '08_12_22'}

def json_enumerator(idx_file, idx_type):

    Json_files = 'C:/Users/Students/Box/Research/IoT4ag/Project_ Water Stress' \
        +'/Data Collection/Pistachio/Multispectral/Json_files/'

    Img_files = 'C:/Users/Students/Box/Research/IoT4ag/Project_ Water Stress' \
    + '/Data Collection/Pistachio/Multispectral/'

    index_all = list([])

    for i in range(len(Dict)):
        NDVI = index_capture(idx_file, idx_type, Img_files, Json_files, list(Dict.keys())[i])
        index_all.append(NDVI)

    return index_all

def index_capture(idx_file, idx_type, Img_files, Json_files, Tx):

    Json_path = Json_files + Tx + '.json'
    image_path = Img_files + Dict[Tx] + '/index_map' + idx_file

    coco = COCO(Json_path)
    img_ids = coco.getImgIds()
    annotation_ids = coco.getAnnIds(img_ids)
    annotations = coco.loadAnns(annotation_ids) 
    print("{} is being extracted for trees in {}...".format(idx_type, Tx))
    
    index_all_date = list([])
    for i in range(len(annotations)):
        # image_id = coco.loadImgs(annotations[i]["id"])[0]   
        # im_id = image_id["id"]
        im_id = annotations[i]["id"]
        entity_id = annotations[i]["category_id"]
        entity = coco.loadCats(entity_id)[0]["name"]
        print("idx={}: image {}: {}".format(i, im_id ,entity))

          
        # print(image_path)
        # image_meta = coco.loadImgs(annotations[i]["image_id"])[0]   
        # image_path = os.path.join(TRAIN_IMAGES_DIRECTORY, image_meta["file_name"])

        masks = coco.annToMask(annotations[i])
        segmentation = np.where(masks == True)
        print(len(segmentation[0]))

        I = io.imread(image_path)
        index_array = I[(segmentation[0],segmentation[1])]

        temp_dict = {'idx': i, 'image_id': im_id, 'Test_Number': entity, 'pixel_array': index_array}
        index_all_date.append([temp_dict])

        # print(Index_a)
        # return Index_array
    return index_all_date
        
#### Calling Functions        
# NDVI = json_enumerator('/NDVI.tif', 'NDVI')
GNDVI = json_enumerator('/GNDVI.tif', 'GNDVI')


# I = io.imread(image_path)
# plt.imshow(I)
# coco.showAnns(annotations)
# plt.show()

#%%

#### For Cropping the photos

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


#%%


