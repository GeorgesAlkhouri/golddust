#!/usr/bin/python

import urllib.request
import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
import glob
import pandas as pd

from mrcnn.model import MaskRCNN
from pathlib import Path
from datetime import datetime


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = 'coco_pretrained_model_config'
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.3

# Filter a list of Mask R-CNN detection results to get only the detected cars busses and trucks


def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


# Root directory of the project
ROOT_DIR = Path('./data')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Path to images, replace with correct path
IMAGE_PATH = '/home/georges/images'

images = {datetime.strptime(os.path.basename(file)[:-4], '%Y-%m-%d_%H:%M'): cv2.cvtColor(
    cv2.imread(file), cv2.COLOR_BGR2RGB) for file in glob.glob(IMAGE_PATH + '/*.jpg')}

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode='inference', model_dir=MODEL_DIR,
                 config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

print(model.keras_model.summary())

cars = {}


# BUG works only with real GPU
for date, image in images.items():
    results = model.detect([image], verbose=0)
    r = results[0]
#     #car_boxes = get_car_boxes(r['rois'], r['class_ids'])
#     #cars[date] = len(car_boxes)

df = pd.DataFrame.from_dict(cars)
df.to_csv(os.path.join(ROOT_DIR, 'cars.csv'))
