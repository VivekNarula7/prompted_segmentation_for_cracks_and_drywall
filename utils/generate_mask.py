import json
import cv2
import numpy as np
import os
import random
import argparse

def convert_coco_to_binary_masks(json_path, output_dir, prompts):
    """
    Reads a COCO JSON annotation file and generates binary PNG masks.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # a visual check to see how the json file looks like
    print(coco_data) 
    print(f"Images : {coco_data['images']}")
    print(f"Annotations : {coco_data['annotations']}")

    images_info = {img['id']: img for img in coco_data['images']}
    print(f"Items in images : {images_info.items()}")
    
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = [] # adds an empty list
        annotations_by_image[image_id].append(ann)

    # 4. Process each image and draw the masks
    for image_id, img_info in images_info.items():
        width = img_info['width']
        height = img_info['height']
        file_name = img_info['file_name']
        
        # Create a blank black image (single channel, 0 values)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # If the image has annotations, draw them as white polygons (255 values)
        if image_id in annotations_by_image:
            for ann in annotations_by_image[image_id]:
                for seg in ann['segmentation']:
                    # COCO polygons are flat lists: [x1, y1, x2, y2, ...]
                    # OpenCV needs them as numpy arrays of shape (N, 2)
                    poly = np.array(seg, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(mask, [poly], 255)
        
        # 5. Format the filename according to the requirements
        # Get the base image ID (filename without the .jpg/.png extension)
        base_id = os.path.splitext(file_name)[0]
        
        # Randomly select a prompt from the list provided
        raw_prompt = random.choice(prompts)
        
        # Replace spaces and slashes with underscores for a clean filename
        # e.g., "segment crack" becomes "segment_crack"
        safe_prompt = raw_prompt.replace(" ", "_").replace("/", "_")
        
        # Construct final filename: 123__segment_crack.png
        mask_filename = f"{base_id}__{safe_prompt}.png"
        mask_path = os.path.join(output_dir, mask_filename)
        
        # Save the binary mask as a PNG
        cv2.imwrite(mask_path, mask)
        
    print(f"Success! Generated {len(images_info)} masks in '{output_dir}'.")

# Parsing input
parser = argparse.ArgumentParser()
parser.add_argument("json_file_path", type=str, help="Enter the path to the json file")
parser.add_argument("output_folder", type=str, help="Enter the path to the output directory")
parser.add_argument("--dataset", type=str, required=True, choices=['cracks', 'drywall'], 
                    help="Specify which dataset this is to use the correct prompts.")
parser.add_argument("--split", type=str, required=True, choices=['train','val','test'], 
                    help="Specify the split for the data mask being generated.")
args = parser.parse_args()

# json_file_path = "cracks/test/_annotations.coco.json"  # Point this to your actual JSON file
# output_folder = "cracks/test_masks"       # Where you want the masks saved
crack_prompts = [
   "segment crack", 
    "segment wall crack"
]
drywall_prompts = [
    "segment taping area", 
    "segment joint/tape", 
    "segment drywall seam" ]

if args.dataset == 'cracks':
    selected_prompts = crack_prompts
elif args.dataset == 'drywall':
    selected_prompts = drywall_prompts

# drywall_prompts = ["segment taping area", "segment joint/tape", "segment drywall seam"]

output_path = os.path.join(args.output_folder, args.split)
convert_coco_to_binary_masks(
    json_path=args.json_file_path, 
    output_dir=output_path, 
    prompts=selected_prompts
)