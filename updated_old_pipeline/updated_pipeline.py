#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NAME: updated_pipeline.py

PURPOSE: 
    - run entire image processing pipeline

HOW TO USE:
    - run script with 'python3 pipeline.py'
NOTES:
    - this script assumes that 'label.py' has already been ran to get input
      points for all images
"""

##############################################################################
#                                  IMPORTS                                   #
##############################################################################
# basics
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
# file work
import os
import json
# computer vision
import cv2
import laser_detection as ld
# models
from segment_anything import sam_model_registry, SamPredictor

import sys
sys.path.append(os.path.abspath("easy_ViTPose"))
from easy_ViTPose import VitInference
from huggingface_hub import hf_hub_download

##############################################################################
#                          PART 1: LOAD IMAGE DATA                           #
##############################################################################
# ALL DATA
# json_file = '../data/red_laser_data.json'
# image_folder = '../data/red_laser_data'
# mask_folder = '../data/red_laser_data_masks'

# SAMPLE DATA
json_file = 'image_data_file.json'
image_folder = 'images'
mask_folder = 'masks'

# load data
with open(json_file, 'r') as file:
    image_data = json.load(file)

# self generated pixel distances
truth_json = 'sample_results.json' # all data but will contain distances for samples as well

# load truth json
with open(truth_json, 'r') as file:
    truth_data = json.load(file)

print(f"Running {len(image_data)} images through pipeline")
##############################################################################
#                                 PART 2: SAM                                #
##############################################################################
# # select checkpoint and model type
# sam_checkpoint = "../sam_vit_h_4b8939.pth"
# model_type = "vit_h"
# device = "cuda"
# # define predictor
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
# predictor = SamPredictor(sam)

# select checkpoint and model type
model_type = "vit_h"

# Automatically detect GPU
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# checkpoint filename and local folder
ckpt_filename = "sam_vit_h_4b8939.pth"
ckpt_local_dir = os.path.join(os.getcwd(), "checkpoints")
ckpt_local_path = os.path.join(ckpt_local_dir, ckpt_filename)

# download if checkpoint does not exist locally
if not os.path.exists(ckpt_local_path):
    print(f"Checkpoint not found at {ckpt_local_path}. Downloading...")
    os.makedirs(ckpt_local_dir, exist_ok=True)
    # download to local folder
    downloaded_path = hf_hub_download(
        repo_id="ybelkada/segment-anything",
        filename=f"checkpoints/{ckpt_filename}"
    )
    # copy from Hugging Face cache to local folder
    import shutil
    shutil.copy(downloaded_path, ckpt_local_path)
    print(f"Downloaded checkpoint to {ckpt_local_path}")
else:
    print(f"Using existing checkpoint at {ckpt_local_path}")

sam_checkpoint = ckpt_local_path

# load SAM
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

#function to generate mask
def generate_mask(im, input_point):
    input_label = np.array([1])
    predictor.set_image(im)
    masks, scores, _ = predictor.predict(point_coords=input_point, 
                                         point_labels=input_label, 
                                         multimask_output=True)
    # return best mask
    return masks[np.argmax(scores)]

# iterate through each entry in the JSON data
for image_name, im_data in tqdm(image_data.items(), desc="SAM"):
    image_path = os.path.join(image_folder, image_name)
    
    # Load the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if img is not None:
        # extract coordinates
        input_point = np.array([im_data['input_point']])
        # Generate the mask
        mask = generate_mask(img, input_point)
        mask = mask.astype(np.uint8)
        
        # Define the mask filename
        mask_filename = os.path.splitext(image_name)[0] + "_mask.png"
        mask_path = os.path.join(mask_folder, mask_filename)
        
        # Save the mask
        cv2.imwrite(mask_path, mask)
        
        # Update the JSON data with the mask filename
        image_data[image_name]['mask'] = mask_filename

##############################################################################
#                        PART 3: LASER POINT DETECTION                       #
##############################################################################
# iterate through each entry in json
for image_name, info in tqdm(image_data.items(), desc="Laser Point Detection"):
    image_path = os.path.join(image_folder, image_name)
    # load image
    source = cv2.imread(image_path)

    # Load the mask
    mask_filename = info['mask']
    mask_path = os.path.join(mask_folder, mask_filename)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask in grayscale

    mask = mask > 0  # need to convert to boolean values

    if source is not None:
        # run laser detection image
        points = ld.detect_red_laser_points(source, mask)
        if len(points) < 2:
            print("Less than two lasers detected. Retrying without mask...")
            points = ld.detect_red_laser_points(source, 
                                            np.ones(source.shape, dtype=bool))
            print(f"Detected {len(points)} points without mask")
        
        # Update the JSON data with the mask filename
        image_data[image_name]['laser_points'] = points

##############################################################################
#                               PART 4: VITPOSE                              #
##############################################################################
# define model parameters
MODEL_SIZE = 'b'
YOLO_SIZE = 'n'
DATASET = 'apt36k'
ext = '.pth'
ext_yolo = '.pt'

# download model_path and yolo_path
# download model_path and yolo_path
MODEL_TYPE = "torch"
YOLO_TYPE = "torch"
REPO_ID = 'JunkyByte/easy_ViTPose'

# --- replace this ---
# FILENAME = os.path.join(MODEL_TYPE, f'{DATASET}/vitpose-' + MODEL_SIZE + f'-{DATASET}') + ext

# --- with this ---
FILENAME = f"{MODEL_TYPE}/{DATASET}/vitpose-{MODEL_SIZE}-{DATASET}.pth"

FILENAME_YOLO = f"yolov8/yolov8{YOLO_SIZE}.pt"

print(f'Downloading model {REPO_ID}/{FILENAME}')
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
yolo_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_YOLO)


# initialize model
model = VitInference(model_path, yolo_path, MODEL_SIZE, dataset=DATASET)

# iterate through each entry in json
for image_name, info in tqdm(image_data.items(), desc="ViTPose"):
    image_path = os.path.join(image_folder, image_name)
    # load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for plotting

    # get keypoints
    img_arr = np.array(img, dtype=np.uint8)
    keypoints = model.inference(img_arr)

    if keypoints:
      # store results
      shoulder = keypoints[0][3]
      shoulder = [int(shoulder[1]), int(shoulder[0])]
      rump = keypoints[0][4]
      rump = [int(rump[1]), int(rump[0])]
      # Update the JSON data with shoulder and rump
      image_data[image_name]['shoulder_rump'] = [shoulder, rump]
    else:
      print(f"vitpose failed for {image_path}")
      # TODO how do we want to handle this case in the json?
      image_data[image_name]['shoulder_rump'] = None

##############################################################################
#                      PART 5: GET FINAL DISTANCES                           #
##############################################################################
# measured_df = pd.read_csv('../measured.csv')
# measured_df = measured_df.dropna(subset=['PhotoID'])
# conversion_dict = dict(zip(measured_df['PhotoID'], measured_df['Laser Width']))
# true_dist_dcit = dict(zip(measured_df['PhotoID'], measured_df['BodyLength1']))

# # open truth data

# # iterate through each entry in json
# for image_name, info in tqdm(image_data.items(), desc="Calculating Final Distances"):
#     # distances in pixels
#     laser_points = info['laser_points']
#     shoulder_rump = info['shoulder_rump']

#     # assert we have values
#     if laser_points is None or shoulder_rump is None:
#         continue
    
#     # calculate ratio
#     laser_dist = math.dist(points[0], points[1])
#     sr_dist = math.dist(shoulder_rump[0], shoulder_rump[1])
#     ratio = laser_dist / sr_dist
    
#     # lookup laser width
#     # TODO
#     true_laser_points = truth_data[image_name]['laser_points']
#     true_laser_dist = math.dist(true_laser_points[0], true_laser_points[1])
    
#     photo_id = image_name.split('.')[0]
#     laser_width = conversion_dict[photo_id]
#     body_length = laser_width / ratio
#     print(f"Calculated length: {round(body_length, 3)}\tactual length: {round(true_dist_dcit[id], 3)}")
    
    
    
    
# # Write the updated JSON data to a file
# with open(json_file, 'w') as file:
#     json.dump(image_data, file, indent=4)

# print(f"Updated data saved to {json_file}")


##############################################################################
#                      PART 5.5: GET FINAL DISTANCES NO COMPARSION TO TRUTH  #
##############################################################################

# Dictionary to store body length calculations and error reasons
body_length_results = {}

# iterate through each entry in json
for image_name, info in tqdm(image_data.items(), desc="Calculating Final Distances"):
    # distances in pixels
    laser_points = info.get('laser_points')
    shoulder_rump = info.get('shoulder_rump')

    # Validate detected laser points
    if not laser_points or not isinstance(laser_points, (list, tuple)):
        body_length_results[image_name] = {
            'error': 'no_detected_laser_points',
            'detected_count': 0
        }
        continue
    if len(laser_points) < 2:
        body_length_results[image_name] = {
            'error': 'insufficient_detected_laser_points',
            'detected_count': len(laser_points)
        }
        continue

    # Validate shoulder/rump keypoints
    if not shoulder_rump or not isinstance(shoulder_rump, (list, tuple)) or len(shoulder_rump) < 2:
        body_length_results[image_name] = {
            'error': 'missing_or_invalid_shoulder_rump',
            'shoulder_rump': shoulder_rump
        }
        continue

    # Safely compute distances
    try:
        detected_laser_dist = math.dist(laser_points[0], laser_points[1])
        sr_dist = math.dist(shoulder_rump[0], shoulder_rump[1])
    except Exception as e:
        body_length_results[image_name] = {
            'error': 'distance_calculation_failed',
            'message': str(e)
        }
        continue

    # Avoid division by zero
    if detected_laser_dist == 0:
        body_length_results[image_name] = {
            'error': 'zero_detected_laser_distance'
        }
        continue

    # ratio = SR distance / detected laser distance
    ratio = sr_dist / detected_laser_dist

    # get actual laser distance from truth_data (sample_results.json)
    if image_name in truth_data and 'laser_points' in truth_data[image_name] and isinstance(truth_data[image_name]['laser_points'], (list, tuple)) and len(truth_data[image_name]['laser_points']) >= 2:
        true_laser_points = truth_data[image_name]['laser_points']
        actual_laser_dist = math.dist(true_laser_points[0], true_laser_points[1])

        # Calculate body length = actual laser distance * ratio
        calculated_body_length = actual_laser_dist * ratio

        body_length_results[image_name] = {
            'detected_laser_dist': detected_laser_dist,
            'sr_dist': sr_dist,
            'ratio': ratio,
            'actual_laser_dist': actual_laser_dist,
            'calculated_body_length': calculated_body_length
        }
    else:
        body_length_results[image_name] = {
            'error': 'missing_actual_laser_in_truth_data'
        }

# Write the body length results to a new JSON file
output_file = 'body_length_results.json'
with open(output_file, 'w') as file:
    json.dump(body_length_results, file, indent=4)

print(f"Body length calculations saved to {output_file}")
