# from mrcnn.config import Config
# from mrcnn import model as modellib
# from mrcnn import visualize
# import mrcnn
# from mrcnn.utils import Dataset
# from mrcnn.model import MaskRCNN
# import numpy as np
# from numpy import zeros
# from numpy import asarray
# import colorsys
# import argparse
# import imutils
# import random
# import cv2
# import os
# import time
# from matplotlib import pyplot
# from matplotlib.patches import Rectangle
#
# from os import listdir
# from xml.etree import ElementTree

import config
import custom_dataset
import model as modellib
from skimage import io
import time
import torch
import visualize

# CONFIGURATION
config = config.Config()
config.display()

# DATASET
print("Loading datasets...")
train_set = custom_dataset.LampPostDataset()
train_set.load_dataset("../Master_Thesis_GvA_project/data/4_external", is_train=True)
train_set.prepare()

test_set = custom_dataset.LampPostDataset()
test_set.load_dataset("../Master_Thesis_GvA_project/data/4_external", is_train=False)
test_set.prepare()
print("Train: %d, Test: %d images" % (len(train_set.image_ids), len(test_set.image_ids)))

# LOAD MODEL
print("Loading the model...")
model = modellib.MaskRCNN(config=config, model_dir='./models/')

# LOAD WEIGHTS
model.load_weights('./models/mask_rcnn_coco.pth', callback=True)  # exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
# "mrcnn_bbox", "mrcnn_mask"]

# TRAIN MODEL
# train heads with higher lr to speedup the learning
model.train_model(train_set, test_set, learning_rate=2 * config.LEARNING_RATE, epochs=5, layers='5+')
# history: model.loss_history, model.val_loss_history

# TEST MODEL
model = modellib.MaskRCNN(config=config, model_dir='./models')
# loading the trained weights of the custom dataset
model.load_weights(model.find_last()[1])
# img = io.imread("../Master_Thesis_GvA_project/data/4_external/TMX7316010203-000363_pano_0000_000600")
# # detecting objects in the image
# result = model.detect([img])

image_id = 1
image, image_meta, gt_class_id, gt_bbox = modellib.load_image_gt(test_set, config, image_id,
                                                                          use_mini_mask=False)
info = test_set.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       test_set.image_reference(image_id)))
# Run object detection
results = model.detect([image])
# Display results

r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            test_set.class_names, r['scores'],
                            title="Predictions")
