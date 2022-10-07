# %%

# import numpy as np
# import cv2 
# from PIL import Image

# root = 'C:/Users/Students/Box/IoT-4Ag -Data/Multispectral_UAV/Pistachio/Pistachio_8-12-22_4cm/map'
# root_color = root + '/index_map_color/NDVI_local.tif'
# root_idx = root + '/index_map/NDVI.tif'

# img_color = cv2.imread(root_color)
# img_idx = cv2.imread(root_idx)

# cv2.imshow('Image',img_color)



# im = Image.open(root_idx) # Can be many different formats.
# pix = im.load()
# print(im.size)  # Get the width and hight of the image for iterating over
# pix_val = []
# matrix = np.array(im)
# print(matrix)
# # max(matrix)

# %%

# CVAT file analysis
import numpy as np
from matplotlib import image
import os
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import skimage.io as io


Json_root = 'C:/Users/Students/Downloads/Test_Sensor_trees/images/instances_default.json'
img_root = 'C:/Users/Students/Downloads/Test_Sensor_trees/images'
img_root_idx = 'C:/Users/Students/Downloads/Test_Sensor_trees/images/NDVI_idx.tif'

TRAIN_IMAGES_DIRECTORY = img_root
TRAIN_ANNOTATIONS_PATH = Json_root

coco = COCO(TRAIN_ANNOTATIONS_PATH)
img_ids = [1]
annotation_ids = coco.getAnnIds(img_ids)
annotations = coco.loadAnns(annotation_ids)

for i in range(len(annotations)):
    entity_id = annotations[i]["category_id"]
    entity = coco.loadCats(entity_id)[0]["name"]
    print("{}: {}".format(i, entity))

image_meta = coco.loadImgs(annotations[i]["image_id"])[0]
image_path = os.path.join(TRAIN_IMAGES_DIRECTORY, image_meta["file_name"])

I = io.imread(image_path)
# plt.imshow(I)
# coco.showAnns(annotations)
# plt.show()

# %%
from PIL import Image
import cv2

# Get the masks
masks = coco.annToMask(annotations[0])

# Pick an item to mask
item_mask = masks
im = cv2.imread(image_path)

# Get the true bounding box of the mask (not the same as the bbox prediction)
segmentation = np.where(item_mask == True)
x_min = int(np.min(segmentation[1]))
x_max = int(np.max(segmentation[1]))
y_min = int(np.min(segmentation[0]))
y_max = int(np.max(segmentation[0]))
print(x_min, x_max, y_min, y_max)

# Create a cropped image from just the portion of the image we want
cropped = Image.fromarray(im[y_min:y_max, x_min:x_max, :])
cropped.save("test.tif")

# Create a PIL image out of the mask
mask = Image.fromarray((item_mask * 255).astype('uint8'))

# Crop the mask to match the cropped image
cropped_mask = mask.crop((x_min, y_min, x_max, y_max))
cropped_mask.save("test.tif")

# Load in a background image and choose a paste position
background = Image.fromarray(im, mode='RGB')
paste_position = (300, 150)

# Create a new foreground image as large as the composite and paste the cropped image on top
new_fg_image = Image.new('RGB', background.size)
new_fg_image.paste(cropped, paste_position)

# Create a new alpha mask as large as the composite and paste the cropped mask
new_alpha_mask = Image.new('L', background.size, color = 0)
new_alpha_mask.paste(cropped_mask, paste_position)

# Compose the foreground and background using the alpha mask
composite = Image.composite(new_fg_image, background, new_alpha_mask)

# Display the image
cv2.imshow(np.array(composite))

# %%
import cv2 as cv
img1 = cv.imread('night_sky.jpg')
img2 = cv.imread('moon.jpg')
img_2_shape = img2.shape
roi = img1[0:img_2_shape[0],0:img_2_shape[1]]
img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
# Now black-out the area of moon in ROI
img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
print(img1.shape, mask.shape)
# Take only region of moon from moon image.
img2_fg = cv.bitwise_and(img2,img2,mask = mask)


# %%
# imgIds = coco.getImgIds(imgIds = img_ids[0])
# img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
# anns_img = np.zeros((img['height'], img['width']))
# for ann in annotations:
#    anns_img = np.maximum(anns_img, coco.annToMask(ann) * ann['category_id'])


# for ann in annotation_ids:
#     # Get individual masks
#     mask = coco.annToMask(coco.loadAnns(ann)[0])
#     # Save masks to BW images
#     file_path = os.path.join(img_root,coco.loadCats(coco.loadAnns(ann)[0]['category_id'])[0]['name'],coco.loadImgs(coco.loadAnns(ann)[0]['image_id'])[0]['file_name'])
#     image.imsave(file_path, mask, cmap="gray")


# mask = coco.annToMask(annotations)
# plt.show(mask)

# plt.axis("off")
# plt.imshow(np.asarray(I))
# plt.savefig(f"{img_ids}.jpg", bbox_inches="tight", pad_inches=0)
# # Plot segmentation and bounding box.
# coco.showAnns(annotations, draw_bbox=True)
# plt.savefig(f"{img_ids}_annotated.jpg", bbox_inches="tight", pad_inches=0)

# %%

# from PIL import Image, ImageDraw
# root = 'C:/Users/Students/Box/IoT-4Ag -Data/Multispectral_UAV/Pistachio/Pistachio_8-12-22_4cm/map'
# root_color = root + '/index_map_color/NDVI_local.tif'
# original = Image.open(root_color)
# xy = [(100,100),(1000,100),(1500,1500),(1000,3000),(100,3000)]
# mask = Image.new("L", original.size, 0)
# draw = ImageDraw.Draw(mask)
# draw.polygon(xy, fill=255, outline=None)
# black =  Image.new("RGB", original.size, 0)
# result = Image.composite(original, black, mask)
# result.save("result.jpg")

# %%

# Project: BenJamesbabala/NNProject_DeepMask
# class CocoUtils(object):
#     def __init__(self, data_dir, data_type):
#         ann_file = '%s/annotations/instances_%s.json' % (data_dir, data_type)
#         # initialize COCO api for instance annotations
#         self.coco = COCO(ann_file)

#     def get_img_annotations(self, pic_id):
#         ann_ids = self.coco.getAnnIds(imgIds=pic_id, iscrowd=None)
#         return self.coco.loadAnns(ann_ids)

#     def get_mask_array_and_image(self, annotation, img_width, img_height, fill_color):
#         seg = annotation['segmentation']
#         raster_img = Image.new('L', (img_width, img_height), 0)
#         for polyg in seg:
#             ImageDraw.Draw(raster_img).polygon(polyg, outline=fill_color, fill=fill_color)
#         return np.array(raster_img), raster_img

#     def get_annotation_mask(self, annotation, img_width, img_height):
#         seg_mask, seg_img = self.get_mask_array_and_image(annotation, img_width, img_height, 1)
#         return seg_mask

#     # mask true's are 1 but image true's are 128- otherwise it's pretty much invisible
#     def get_annotation_image(self, annotation, img_width, img_height):
#         seg_mask, seg_img = self.get_mask_array_and_image(annotation, img_width, img_height, mask_pic_true_color)
#         return seg_img

#     def are_legal_anotations(self, annotations):
#         # unfortunately, only polygon segmentations work for now (RLE mask type decoding causes a python crash)
#         polygon_segmentations = ['segmentation' in ann and type(ann['segmentation']) == list for ann in annotations]
#         return all(polygon_segmentations)

#     def show_annotations(self, pic_path, annotations):
#         if self.are_legal_anotations(annotations):
#             pylab.rcParams['figure.figsize'] = (10.0, 8.0)
#             read_img = io.imread(pic_path)
#             plt.figure()
#             plt.imshow(read_img)
#             self.coco.showAnns(annotations)
#         else:
#             print 'cannot show invalid annotation'

#     def get_images_data(self):
#         # each item is image_id, image_file_name
#         return [pic_data[1] for pic_data in self.coco.imgs.items()]

