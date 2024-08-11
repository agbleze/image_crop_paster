
#%%
from pycocotools.coco import COCO
import cv2
import numpy as np
from PIL import Image
from clusteval import clusteval
import pandas as pd
import json
from typing import Union, List, Dict
import os
import random
#%% Load COCO annotations
coco = COCO('/home/lin/codebase/cv_with_roboflow_data/coco_annotation_coco.json')


# %%
tomato_coco_path = "/home/lin/codebase/cv_with_roboflow_data/tomato_coco_annotation/annotations/instances_default.json"
img_dir = "/home/lin/codebase/cv_with_roboflow_data/images"
coco = COCO(annotation_file=tomato_coco_path)

#%%
def get_objects(imgname, coco, img_dir):
    try:
        val = next(obj for obj in coco.imgs.values() if obj["file_name"] == imgname)
    except StopIteration:
        raise ValueError(f"Image {imgname} not found in COCO dataset.")
    
    img_id = val['id']
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, imgname)
    image = cv2.imread(img_path)

    # Get annotation IDs for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    img_obj = []

    for ann in anns:
        mask = coco.annToMask(ann)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cropped_object = image[y:y+h, x:x+w]

            # Apply the mask to the cropped object
            mask_cropped = mask[y:y+h, x:x+w]
            print(f"mask_cropped: {mask_cropped.shape}")
            cropped_object = cv2.bitwise_and(cropped_object, cropped_object, mask=mask_cropped)
            
            # Remove the background (set to transparent)
            cropped_object = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGBA)
            cropped_object[:, :, 3] = mask_cropped * 255

            img_obj.append(cropped_object)
    
    os.makedirs(name="crop_objs", exist_ok=True)
    for img_count, each_img_obj in enumerate(img_obj):
        cv2.imwrite(filename=f"crop_objs/img_obj_{img_count}.png", img=each_img_obj)
    
    return img_obj




#%%
objects = get_objects(imgname="0.jpg", coco=coco, img_dir=img_dir)

#%%

Image.fromarray(cv2.cvtColor(objects[0], cv2.COLOR_BGR2BGRA))
#%%

objects[2].shape


#%%    ########## with resize   #########

import os
import cv2
import numpy as np
from typing import Tuple

def paste_object(dest_img_path, cropped_objects: Dict[str, List[np.ndarray]], min_x=None, min_y=None, 
                 max_x=None, max_y=None, 
                 resize_w=None, resize_h=None, 
                 sample_location_randomly: bool = True,
                 )->Tuple[np.ndarray, List, List, List]:
    # Load the destination image
    dest_image = cv2.imread(dest_img_path, cv2.IMREAD_UNCHANGED)
    dest_image = cv2.cvtColor(dest_image, cv2.COLOR_BGR2RGB)
    dest_h, dest_w = dest_image.shape[:2]
        
    if not isinstance(cropped_objects, dict):
        raise ValueError(f"""cropped_objects is expected to be a dictionary of 
                         key being the category_id and value being a list of
                         cropped object image (np.ndarray)
                         """)
    bboxes, segmentations, category_ids = [], [], []
    # Resize the cropped object if resize dimensions are provided
    for cat_id in cropped_objects:
        cat_cropped_objects = cropped_objects[cat_id]
        if not isinstance(cat_cropped_objects, list):
            cat_cropped_objects = [cat_cropped_objects]
        for cropped_object in cat_cropped_objects:
            print(f"cropped_object: {cropped_object.shape}")
            if sample_location_randomly:
                min_x = random.random()
                max_x = random.uniform(min_x, 1)
                min_y = random.random()
                max_y = random.uniform(min_y, 1)
                
                x = int(min_x * dest_w)
                y = int(min_y * dest_h)
                max_x = int(max_x * dest_w)
                max_y = int(max_y * dest_h)
            else:
                x = int(min_x * dest_w)
                y = int(min_y * dest_h)
                max_x = int(max_x * dest_w)
                max_y = int(max_y * dest_h)
                
            if resize_w and resize_h:
                print(f"cropped_object: {cropped_object} \n")
                obj_h, obj_w = cropped_object.shape[:2]
                aspect_ratio = obj_w / obj_h
                if resize_w / resize_h > aspect_ratio:
                    resize_w = int(resize_h * aspect_ratio)
                else:
                    resize_h = int(resize_w / aspect_ratio)
                resized_object = cv2.resize(cropped_object, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
            else:
                resized_object = cropped_object

            # Ensure the resized object fits within the specified area
            obj_h, obj_w = resized_object.shape[:2]
            if obj_w > (max_x - x) or obj_h > (max_y - y):
                scale_x = (max_x - x) / obj_w
                if scale_x <= 0:
                    scale_x = 0.1
                scale_y = (max_y - y) / obj_h
                if scale_y <= 0:
                    scale_y = 0.1
                    
                scale = min(scale_x, scale_y)
                new_w = int(obj_w * scale)
                new_h = int(obj_h * scale)
                resized_object = cv2.resize(resized_object, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Create a mask for the resized object
            print(f"resized_object: {resized_object.shape} \n")
            if resized_object.shape[2] == 3:
                resized_object = cv2.cvtColor(resized_object, cv2.COLOR_RGB2RGBA)
                print(f"after resized object to RGBA: {resized_object.shape}")
            mask = resized_object[:, :, 3]
            mask_inv = cv2.bitwise_not(mask)
            resized_object = resized_object[:, :, :3]

            # Define the region of interest (ROI) in the destination image
            roi = dest_image[y:y+resized_object.shape[0], x:x+resized_object.shape[1]]

            # Black-out the area of the object in the ROI
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of the object from the object image
            obj_fg = cv2.bitwise_and(resized_object, resized_object, mask=mask)

            # Put the object in the ROI and modify the destination image
            dst = cv2.add(img_bg, obj_fg)
            dest_image[y:y+resized_object.shape[0], x:x+resized_object.shape[1]] = dst

            # Calculate the bounding box
            bbox = [x, y, resized_object.shape[1], resized_object.shape[0]]
            # Calculate the segmentation
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = []
            for contour in contours:
                contour = contour.flatten().tolist()
                segmentation.append(contour)
            bboxes.append(bbox)
            segmentations.append(segmentation)
            category_ids.append(int(cat_id))
            #TODO:
            # Include checking if crop obj segmentation mask overlaps with 
            # any segmask in the background and
    return dest_image, bboxes, segmentations, category_ids

def create_coco_annotation(image_id, bbox, segmentation):
    annotation = {
        "image_id": image_id,
        "bbox": bbox,
        "segmentation": segmentation,
        "category_id": 1,  # Assuming a single category for simplicity
        "id": 1  # Annotation ID
    }
    return annotation

def export_coco_annotation(annotation, output_path):
    with open(output_path, 'w') as f:
        json.dump(annotation, f, indent=4)
#%% Example usage
#cropped_object = cv2.imread('path_to_cropped_object.png', cv2.IMREAD_UNCHANGED)
dest_img_path = '/home/lin/codebase/cv_with_roboflow_data/images/166.jpg'
min_x, min_y = 0.7, 0.7  # Define the minimum coordinates (0 to 1)
max_x, max_y = 0.999, 0.999  # Define the maximum coordinates (0 to 1)
resize_w, resize_h = 150, 150  # Define the resize dimensions for the cropped object

#%%
result_image, bbox, segmentation, cat_id = paste_object(dest_img_path, {"1": objects[0]}, min_x, min_y, 
                                                        max_x, max_y, resize_w, resize_h
                                                        )

result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
cv2.imwrite('path_to_result_image.png', result_image)

annotation = create_coco_annotation(image_id=1, bbox=bbox, segmentation=segmentation)
export_coco_annotation(annotation, 'path_to_annotation.json')

#%%
#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
import cv2
bbox = bbox[0]
cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)


#%%
cv2.imwrite("/home/lin/codebase/image_crop_paster/viz_image.png", result_image)


#%%
import numpy as np
from pycocotools import mask

def annToMask(ann, height, width):
    """
    Convert annotation to binary mask.
    
    Parameters:
    ann (dict): COCO annotation dictionary.
    height (int): Height of the image.
    width (int): Width of the image.
    
    Returns:
    np.ndarray: Binary mask.
    """
    segmentation = ann['segmentation']
    rle = mask.frPyObjects(segmentation, height, width)
    binary_mask = mask.decode(rle)
    
    return binary_mask

#%%
from typing import Union
def crop_obj_per_image(obj_names: list, imgname: Union[str, List], img_dir,
                       coco_ann_file: str
                       ) -> Union[Dict[str,List], None]:
    cropped_objs_collection = {}
    # get objs in image
    with open(coco_ann_file, "r") as filepath:
        coco_data = json.load(filepath)
        
    categories = coco_data["categories"]
    category_id_to_name_map = {cat["id"]: cat["name"] for cat in categories}
    category_name_to_id_map = {cat["name"]: cat["id"] for cat in categories}
    
    coco = COCO(coco_ann_file)
    # if isinstance(imgnames, str):
    #     imgnames = [imgnames]
    images = coco_data["images"]
    # for imgname in imgnames:
    image_info = [img_info for img_info in images if img_info["file_name"]==imgname][0]
    image_id = image_info["id"]
    image_height = image_info["height"]
    image_width = image_info["width"]
    annotations = coco_data["annotations"]
    img_ann = [ann_info for ann_info in annotations if ann_info["image_id"]==image_id]
    img_catids = set(ann_info["category_id"] for ann_info in img_ann)
    img_objnames = [category_id_to_name_map[catid] for catid in img_catids]
    img_path = os.path.join(img_dir, imgname)
    image = cv2.imread(img_path)
    objs_to_crop = set(img_objnames).intersection(set(obj_names))
    if objs_to_crop:
        for objname in obj_names:
            print(f"objname: {objname} \n")
            object_masks = []
            if objname in img_objnames:
                obj_id = category_name_to_id_map[objname]
                for ann in img_ann:
                    if ann["category_id"] == obj_id:
                        mask = coco.annToMask(ann)
                        #mask = annToMask(ann=ann, height=image_height, width=image_width)
                        
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            cropped_object = image[y:y+h, x:x+w]
                            mask_cropped = mask[y:y+h, x:x+w]
                            cropped_object = cv2.bitwise_and(cropped_object, cropped_object, 
                                                             mask=mask_cropped)
                            # Remove the background (set to transparent)
                            cropped_object = cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGBA)
                            print(f"mask_cropped: {mask_cropped.shape} \n")
                            print(f"new mask_cropped[:,:] {mask_cropped[:,:].shape} \n")
                            cropped_object[:, :, 3] = mask_cropped * 255
                            object_masks.append(cropped_object)
                            #print(f"in contours loop cropped_objs_collection[objname]: {cropped_objs_collection[objname]} \n")
                            #cropped_objs_collection[objname] = cropped_objs_collection[objname].append([cropped_object])
            #print(f"imgname: {imgname},  objname: {objname}")
                if objname not in cropped_objs_collection.keys():
                    cropped_objs_collection[objname] = object_masks
                    #print(f"cropped_objs_collection: {cropped_objs_collection.keys()} \n")
                else:
                    for each_mask in object_masks:
                        #print(f"each mask cropped_objs_collection: {cropped_objs_collection.keys()} \n")
                        #print(f"{objname}: {cropped_objs_collection[objname]} \n")
                        #cropped_objs_collection[objname] = 
                        cropped_objs_collection[objname].append(each_mask)
            
            
    return cropped_objs_collection
        #else:
        #    return None

#%%
coco_ann_path = "/home/lin/codebase/cv_with_roboflow_data/tomato_coco_annotation/annotations/instances_default.json"
img_dir= "/home/lin/codebase/cv_with_roboflow_data/images"
imgname = "494.jpg"
objnames = ["ripe", "unripe", "flowers"]

cropped_obj_collect = crop_obj_per_image(obj_names=objnames, imgname=imgname, img_dir=img_dir,
                                        coco_ann_file=coco_ann_path
                                        )

#%%

cropped_obj_collect.keys()
#%%

len(cropped_obj_collect["ripe"])

#%%

unripe = cropped_obj_collect["unripe"][0]

unripe.shape

#%%

Image.fromarray(unripe) #.shape

#%%

unripe[:,:,1]

#%%

Image.fromarray(unripe)
#%%
Image.fromarray(cropped_obj_collect["ripe"][2])
#%%
#cropped_obj_collect["unripe"][2]


#%%
imgnames_for_cropping = ["0.jpg", "1235.jpg", "494.jpg", "446.jpg", "10.jpg"]
# all_crop_objects = crop_obj_per_image(obj_names=objnames, imgnames=imgnames_for_cropping, img_dir=img_dir,
#                    coco_ann_file=coco_ann_path)



#%% # pseudo code
def collate_all_crops(object_to_cropped, imgnames_for_crop, img_dir,
                      coco_ann_file
                      ):
    #all_crops = {obj: [] for obj in object_to_cropped}
    #allimg_crops = []
    all_crops = {}
    for img in imgnames_for_crop:
        #print(f"starting all_crops: {all_crops.keys()} \n")
        #print(f"img: {img}")
        crop_obj = crop_obj_per_image(obj_names=object_to_cropped, 
                                      imgname=img, 
                                    img_dir=img_dir,
                                    coco_ann_file=coco_ann_file
                                    )
        #print(f"crop_obj: {crop_obj} \n")
        for each_object in crop_obj.keys():
            if each_object not in all_crops.keys():
                all_crops[each_object] = crop_obj[each_object]
            else:
                #print(f"each_object: {each_object}\n all_crops: {all_crops.keys()}")
                cpobjs = crop_obj[each_object]
                if all_crops[each_object] is None:
                    all_crops[each_object] = cpobjs
                else:
                    #print(f"in else: {all_crops[each_object]}\n")
                    for idx, cpobj in enumerate(cpobjs): 
                        #print(f"idx: {idx}")
                        #append_obj = all_crops[each_object]
                        #print(f"img: {img} len(append_obj): {len(append_obj)} \n type(append_obj): {type(append_obj)}")
                        #all_crops[each_object] = append_obj.append(cpobj)
                        all_crops[each_object].append(cpobj)
                        #print(f"idx: {idx} img: {img} successful appending")
                        #print(f"all_crops[each_object]: {all_crops[each_object]}")
        #print(f"finished all_crops: {all_crops.keys()} \n")            
                    
    return all_crops


#%%

imgnames_for_cropping = ["0.jpg", "1235.jpg", "494.jpg", "446.jpg", "10.jpg"]
["10.jpg"]
all_crop_objects = collate_all_crops(object_to_cropped=objnames, imgnames_for_crop=imgnames_for_cropping,
                                    img_dir=img_dir, coco_ann_file=coco_ann_path
                                    )


#%%
from collections import Counter


[print(f"{i}: {len(all_crop_objects[i])}") for i in all_crop_objects]

all_crop_objects.keys()


#%%

Image.fromarray(all_crop_objects["ripe"][1])

#%%

len(all_crop_objects["unripe"])

#%%         
def paste_crops_on_bkgs(bkgs, all_crops, objs_paste_num: Dict, output_img_dir, save_coco_ann_as,
                        min_x=None, min_y=None, 
                        max_x=None, max_y=None, 
                        resize_width=None, resize_height=None,
                        sample_location_randomly=True
                        ):
    os.makedirs(output_img_dir, exist_ok=True)
    coco_ann = {"categories": [{"id": obj_idx+1, "name": obj} for obj_idx, obj in enumerate(sorted(objs_paste_num))], 
                "images": [], 
                "annotations": []
                }
    ann_ids = []
    for bkg_idx, bkg in enumerate(bkgs):
        # for obj_idx, obj in enumerate(objs_paste_num):
        #     num_obj = objs_paste_num[obj]
        #     objs_to_paste = all_crops[obj]
        #     sampled_obj = random.sample(objs_to_paste, int(num_obj))
        sampled_obj = {obj_idx+1: random.sample(all_crops[obj], int(objs_paste_num[obj])) 
                       for obj_idx, obj in enumerate(sorted(objs_paste_num))
                       }    
            # for multiple objects, last pasted object is overriding the first pasted
            # TODO: sample all objects to be pasted at once and send for pasting
            
        dest_img, bboxes, segmasks, category_ids = paste_object(dest_img_path=bkg,  ## showed also return the obj_idx as category_id
                                                                cropped_objects=sampled_obj,
                                                                min_x=min_x, min_y=min_y, max_x=max_x,
                                                                max_y=max_y, resize_h=resize_height,
                                                                resize_w=resize_width,
                                                                sample_location_randomly=sample_location_randomly
                                                                )
        file_name = os.path.basename(bkg)
        img_path = os.path.join(output_img_dir, file_name)
        dest_img = cv2.cvtColor(dest_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path, dest_img)
        assert(len(bboxes) == len(segmasks) == len(category_ids)), f"""bboxes: {len(bboxes)}, segmasks: {len(segmasks)} and category_ids: {len(category_ids)} are not equal length"""
                    
        #image = cv2.imread(bkg)
        img_height, img_width = dest_img.shape[0], dest_img.shape[1]
        img_id = bkg_idx+1
        
        
        image_info = {"file_name": file_name, "height": img_height, 
                        "width": img_width, "id": img_id
                        }
        coco_ann["images"].append(image_info)
        
        for ann_ins in range(0, len(bboxes)):
            bbox = bboxes[ann_ins]
            segmask = segmasks[ann_ins]
            ann_id = len(ann_ids) + 1
            ann_ids.append(ann_id)
            category_id = category_ids[ann_ins]
            annotation = {"id": ann_id, 
                          "image_id": img_id, 
                        "category_id": category_id,
                        "bbox": bbox,
                        "segmentation": segmask
                        } 
            #coco_ann["annotations"] = 
            coco_ann["annotations"].append(annotation)
    with open(save_coco_ann_as, "w") as filepath:
        json.dump(coco_ann, filepath)            
                

#%%

all_crop_objects["unripe"][0].shape 

#%%

Image.fromarray(all_crop_objects["ripe"][2])       
#%%
bkgs = ["/home/lin/codebase/cv_with_roboflow_data/images/1859.jpg",
        "/home/lin/codebase/cv_with_roboflow_data/images/1668.jpg",
        "/home/lin/codebase/cv_with_roboflow_data/images/1613.jpg",
        "/home/lin/codebase/cv_with_roboflow_data/images/1541.jpg",
        "/home/lin/codebase/cv_with_roboflow_data/images/1892.jpg"
        ]

#%%
obj_paste_num = {"ripe": 2, "unripe": 2}    
paste_crops_on_bkgs(bkgs=bkgs, all_crops=all_crop_objects, 
                    objs_paste_num=obj_paste_num,
                    output_img_dir="pasted_output_dir",
                    save_coco_ann_as="cpaug.json",
                    sample_location_randomly=True,
                    #min_x=0, min_y=0, max_x=1, max_y=1, 
                    resize_height=50, 
                    resize_width=50
                    )

#%%
bkg_with_obj = ["/home/lin/codebase/image_crop_paster/images/10.jpg"]

paste_crops_on_bkgs(bkgs=bkg_with_obj, all_crops=all_crop_objects, 
                    objs_paste_num=obj_paste_num,
                    output_img_dir="pasted_bkg_withobj",
                    save_coco_ann_as="bkgwith_obj_cpaug.json",
                    sample_location_randomly=False,
                    min_x=0.4, min_y=0.5, max_x=0.6, max_y=0.6, 
                    resize_height=None, 
                    resize_width=None
                    )

#%% TODO.
"""
For each obj about to be cropped, check if the background image has objs
if bkg has objs then check if segmask of crop and and any of segmaks
of bkg objs overlap
if they overlap, adjust the segmask of the crop obj
"""
#%%
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
import cv2

# Load COCO annotations

with open(tomato_coco_path) as f:
    coco_data = json.load(f)

coco = COCO(tomato_coco_path)

# Get the image and annotation IDs
image_id = coco.getImgIds()[0]
annotation_ids = coco.getAnnIds(imgIds=image_id)

# Load the annotations
annotations = coco.loadAnns(annotation_ids)
image_info = coco.loadImgs(image_id)[0]

os.path.join(img_dir, image_info.get("file_name"))
# Load the image
image_path = os.path.join(img_dir, image_info.get("file_name"))  # Replace with the actual path to your image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create an empty mask
mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

# Extract segmentation masks and fill the mask
for ann in annotations:
    if 'segmentation' in ann:
        segmentation = ann['segmentation']
        if isinstance(segmentation, list):  # Polygon format
            for polygon in segmentation:
                poly = np.array(polygon).reshape((-1, 2))
                rr, cc = poly[:, 1].astype(int), poly[:, 0].astype(int)
                mask[rr, cc] = 1
        elif isinstance(segmentation, dict):  # RLE format
            rle = maskUtils.frPyObjects(segmentation, image_info['height'], image_info['width'])
            mask = np.maximum(mask, maskUtils.decode(rle))

# Find the pixel locations
pixel_coords = np.argwhere(mask == 1)
print("Pixel coordinates:", pixel_coords)

# Visualize the mask on the image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.scatter(pixel_coords[:, 1], pixel_coords[:, 0], color='red', s=1)
plt.title('Segmentation Mask Overlay')
plt.axis('off')
plt.show()

#%%
Image.fromarray(image)
#%%

duparray = np.array([[263,  48],
                    [263,  53],
                    [264,  45],
                    [265,  55],
                    [267,  44],
                    [268,  66],
                    [269,  56],
                    [269,  63],
                    [0, 0], [2, 0], [0, 10]])
len(pixel_coords)


#%%

len(duparray)

res_array = np.array([i for i in pixel_coords if i in duparray])

matching_values = np.array([row for row in pixel_coords if any((row == x).all() for x in duparray)])
print(matching_values)
#[print(i) for i in res_array if i == a for a in duparray]   





#%%
pixel_present_in_both = []
for i in res_array:
    #print(type(i))
    for a in duparray:
        if np.array(i).all() == np.array(a).all():
            pixel_present_in_both.append(i)
 
print(pixel_present_in_both)
#%%

duparray in res_array     
#%%

np.logical_and(pixel_coords, res_array)      
        
#%%

exmdict = {"first": [], "second": []}

if not exmdict["first"]:
    print("empty")
else:
    print("occupied")
# %%
import random

random.sample([1,2,3,4,9,2,3,3], 4)
# %%


""" TODO:
given the annotation of images in coco format, read the bbox and segmentation 
and use that to determine and identify images with occlusion. After that, for 
occluded images, adjust the segmentation and bbox such that one of the image 
takes the full bbox  and segmentation and the others take the non-occluded 
part of the segmentation

"""