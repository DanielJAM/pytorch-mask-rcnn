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

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # for debugging

import config
import custom_dataset
import model as modellib
import os
from skimage import io
import time
import torch
import visualize

torch.backends.cudnn.benchmark = False  # Fails with True, probably due to CUDA 9.0 instead of 10.0 icw RTX 2080Ti

start_time = time.process_time()
print("start time time(s): ", round(start_time, 2))

# CONFIGURATION
config = config.Config()
config.display()

ONLY_TEST = 0

if not ONLY_TEST:
    # DATASET
    train_set = custom_dataset.LampPostDataset()
    train_set.load_dataset("../Master_Thesis_GvA_project/data/4_external", is_train=True)
    train_set.prepare()

test_set = custom_dataset.LampPostDataset()
test_set.load_dataset("../Master_Thesis_GvA_project/data/4_external", is_train=False)
test_set.prepare()
if not ONLY_TEST: print("Train: %d, Test: %d images" % (len(train_set.image_ids), len(test_set.image_ids)))

if not ONLY_TEST:
    config.STEPS_PER_EPOCH = len(train_set.image_info)

    data_time = time.process_time()
    print("load data time(s): ", round(data_time - start_time, 2), "total elapsed: ", round(data_time, 2))

    # LOAD MODEL
    model = modellib.MaskRCNN(config=config, model_dir='./models/')

    load_model_time = time.process_time()
    print("loading model time(s): ", round(load_model_time - data_time, 2), "total elapsed: ",
          round(load_model_time, 2))

    # LOAD WEIGHTS
    model.load_weights('./models/mask_rcnn_coco.pth', callback=True)  # exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
    # "mrcnn_bbox", "mrcnn_mask"]

    load_weights_time = time.process_time()
    print("loading weights time(s): ", round(load_weights_time - load_model_time, 2), "total elapsed: ",
          round(load_weights_time, 2))

    # Save final config before start training
    config.to_txt(model.log_dir)

    # TRAIN MODEL
    # train heads with higher lr to speedup the learning
    model.train_model(train_set, test_set, learning_rate=2 * config.LEARNING_RATE, epochs=50, layers='5+')
    #       heads: The RPN, classifier and mask heads of the network
    #       all: All the layers
    #       3+: Train Resnet stage 3 and up
    #       4+: Train Resnet stage 4 and up
    #       5+: Train Resnet stage 5 and up

    train_time = time.process_time()
    print("training time(s): ", round(train_time - load_weights_time, 2), "total elapsed: ", round(train_time, 2))


# TEST MODEL
model = modellib.MaskRCNN(config=config, model_dir='./models')
# loading the trained weights of the custom dataset
model.load_weights(model.find_last()[1])
# img = io.imread("../Master_Thesis_GvA_project/data/4_external/TMX7316010203-000363_pano_0000_000600")
# # detecting objects in the image
# result = model.detect([img])

# Delete test model log directory
os.rmdir(model.log_dir)

image_id = 3
# 1 = TMX7316010203-001499_pano_0000_001233 - only a hanging lamp post
# 2 = TMX7316010203-001209_pano_0000_002760 - on the right, behind/above blue car
# 3 = TMX7316010203-001187_pano_0000_002097 - clearly in the middle (old one)
# now seems to only "detect" lamp posts in the top of the picture...
image, image_meta, gt_class_id, gt_bbox = modellib.load_image_gt(test_set, config, image_id, use_mini_mask=False)
info = test_set.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       test_set.image_reference(image_id)))
# Run object detection
results = model.detect([image])

# Display results
r = results[0]
visualize.display_instances(image, r['rois'], r['class_ids'],  # r['masks'],
                            test_set.class_names, r['scores'],
                            title="Predictions")
