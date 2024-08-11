

#%%
import numpy as np
import cv2
from pycocotools import mask

# Define the arrays
array1 = np.array([[263,  48],
                   [263,  53],
                   [264,  45],
                   [265,  55],
                   [267,  44],
                   [268,  66],
                   [269,  56],
                   [269,  63],
                   [269,  68],
                   [271,  44],
                   [271,  56],
                   [274,  55],
                   [279,  55]])

array2 = np.array([[263,  48],
                   [263,  53],
                   [264,  45],
                   [265,  55],
                   [267,  44],
                   [268,  66],
                   [269,  56],
                   [269,  63],
                   [  0,   0]])

# Find the matching rows
matching_values = np.array([row for row in array1 if any((row == x).all() for x in array2)])

# Remove the matching rows from array1
unique_array1 = np.array([row for row in array1 if not any((row == x).all() for x in matching_values)])

# Create an empty binary mask
height, width = 300, 300  # Assuming the image size is 300x300
binary_mask = np.zeros((height, width), dtype=np.uint8)

# Set the pixel locations to 1
for x, y in unique_array1:
    binary_mask[y, x] = 1

# Find contours (polygons) from the binary mask
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert contours to COCO format
coco_segmentation = []
for contour in contours:
    if contour.size >= 6:  # Ensure the contour has at least 3 points (6 coordinates)
        segmentation = contour.flatten().tolist()
        coco_segmentation.append(segmentation)

# Create the COCO annotation format
coco_annotation = {
    "segmentation": coco_segmentation,
    "area": int(np.sum(binary_mask)),  # Area of the mask
    "iscrowd": 0,
    "image_id": 1,  # Example image ID
    "bbox": [int(np.min(unique_array1[:, 0])), int(np.min(unique_array1[:, 1])),
            int(np.max(unique_array1[:, 0]) - np.min(unique_array1[:, 0])),
            int(np.max(unique_array1[:, 1]) - np.min(unique_array1[:, 1]))],
    "category_id": 1,  # Example category ID
    "id": 1  # Example annotation ID
}

print(coco_annotation)

# %%
