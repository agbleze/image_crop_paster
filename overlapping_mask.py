#%%
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
import cv2
import os

# Load COCO annotations
# with open('path_to_coco_annotations.json') as f:
#     coco_data = json.load(f)

# coco = COCO('path_to_coco_annotations.json')

# # Get the image and annotation IDs
# image_id = coco.getImgIds()[0]
# annotation_ids = coco.getAnnIds(imgIds=image_id)

# # Load the annotations
# annotations = coco.loadAnns(annotation_ids)
# image_info = coco.loadImgs(image_id)[0]

#%%
bkgwith_obj_cpaug = "/home/lin/codebase/image_crop_paster/bkgwith_obj_cpaug.json"
pasted_bkg_withobj_dir = "/home/lin/codebase/image_crop_paster/pasted_bkg_withobj"
coco = COCO(annotation_file=bkgwith_obj_cpaug)


with open(bkgwith_obj_cpaug) as f:
    coco_data = json.load(f)

coco = COCO(bkgwith_obj_cpaug)

# Get the image and annotation IDs
image_id = coco.getImgIds()[0]
annotation_ids = coco.getAnnIds(imgIds=image_id)

# Load the annotations
annotations = coco.loadAnns(annotation_ids)
image_info = coco.loadImgs(image_id)[0]

#os.path.join(pasted_bkg_withobj_dir, image_info.get("file_name"))
# Load the image
image_path = os.path.join(pasted_bkg_withobj_dir, image_info.get("file_name")) 
# Load the image
#image_path = 'path_to_image.jpg'  # Replace with the actual path to your image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create an empty mask for the entire image
mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

# Create a list to store individual object masks
object_masks = []

# Extract segmentation masks and fill the mask
for ann in annotations:
    if 'segmentation' in ann:
        segmentation = ann['segmentation']
        object_mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        if isinstance(segmentation, list):  # Polygon format
            for polygon in segmentation:
                poly = np.array(polygon).reshape((-1, 2))
                rr, cc = poly[:, 1].astype(int), poly[:, 0].astype(int)
                object_mask[rr, cc] = 1
        elif isinstance(segmentation, dict):  # RLE format
            rle = maskUtils.frPyObjects(segmentation, image_info['height'], image_info['width'])
            object_mask = maskUtils.decode(rle)
        object_masks.append(object_mask)
        mask = np.maximum(mask, object_mask)

# Find overlapping pixels
overlap_mask = np.zeros_like(mask)
for i in range(len(object_masks)):
    for j in range(i + 1, len(object_masks)):
        overlap_mask = np.maximum(overlap_mask, object_masks[i] & object_masks[j])

# Find the pixel locations of overlapping areas
overlap_pixel_coords = np.argwhere(overlap_mask == 1)
print("Overlapping pixel coordinates:", overlap_pixel_coords)

# Visualize the overlapping pixels on the image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.scatter(overlap_pixel_coords[:, 1], overlap_pixel_coords[:, 0], color='blue', s=1)
plt.title('Overlapping Segmentation Mask Overlay')
plt.axis('off')
plt.show()



#%%  ###########!/usr/bin/env python3

import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
import cv2

# # Load COCO annotations
# with open('path_to_coco_annotations.json') as f:
#     coco_data = json.load(f)

# coco = COCO('path_to_coco_annotations.json')

# # Get the image and annotation IDs
# image_id = coco.getImgIds()[0]
# annotation_ids = coco.getAnnIds(imgIds=image_id)

# # Load the annotations
# annotations = coco.loadAnns(annotation_ids)
# image_info = coco.loadImgs(image_id)[0]

# # Load the image
# image_path = 'path_to_image.jpg'  # Replace with the actual path to your image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a list to store individual object masks
object_masks = []

# Extract segmentation masks and fill the mask
for ann in annotations:
    if 'segmentation' in ann:
        segmentation = ann['segmentation']
        object_mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        if isinstance(segmentation, list):  # Polygon format
            for polygon in segmentation:
                poly = np.array(polygon).reshape((-1, 2))
                rr, cc = poly[:, 1].astype(int), poly[:, 0].astype(int)
                object_mask[rr, cc] = 1
        elif isinstance(segmentation, dict):  # RLE format
            rle = maskUtils.frPyObjects(segmentation, image_info['height'], image_info['width'])
            object_mask = maskUtils.decode(rle)
        object_masks.append(object_mask)

# Find overlapping pixels
overlap_mask = np.zeros_like(object_masks[0])
for i in range(len(object_masks)):
    for j in range(i + 1, len(object_masks)):
        overlap_mask = np.maximum(overlap_mask, np.logical_and(object_masks[i], object_masks[j]))

# Find the pixel locations of overlapping areas
overlap_pixel_coords = np.argwhere(overlap_mask == 1)
print("Overlapping pixel coordinates:", overlap_pixel_coords)

# Visualize the overlapping pixels on the image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.scatter(overlap_pixel_coords[:, 1], overlap_pixel_coords[:, 0], color='blue', s=1)
plt.title('Overlapping Segmentation Mask Overlay')
plt.axis('off')
plt.show()


# %%
import cv2
from PIL import Image
bboxes = [ann["bbox"] for ann in coco_data["annotations"]]

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

for bbox in bboxes:
    image_with_bbox = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
    
Image.fromarray(image_with_bbox)
    

# %%
