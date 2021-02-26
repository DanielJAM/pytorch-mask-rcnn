"""
Mask R-CNN
Base Configurations class.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Edited by Daniel Maaskant
"""

import datetime
import math
import numpy as np
import os


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Empty string to save starting command in for reference
    RUN_CONFIG = ""

    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = "PanorAMS"  # Override in sub-classes

    # NUMBER OF GPUs to use. For CPU use 0
    # Currently only supports 1 or 0.
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1

    # Specify the number of subprocesses to use for each DataLoader.
    # Currently only supports 1.
    NUMBER_OF_WORKERS = 1

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000  # possibly set to 1/10th of current train set: 797

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone. - must be equal length as RPN_ANCHOR_SCALES
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Number of classification classes (including background)
    NUM_CLASSES = 2  # Override in sub-classes

    # Length of square anchor side in pixels - must be equal length as BACKBONE_STRIDES
    RPN_ANCHOR_SCALES = (128, 256, 512, 1024, 2048)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a tall anchor
    RPN_ANCHOR_RATIOS = [0.5, 0.25, 0.125]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more proposals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # Input image resizing
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    # Must be divisible by 64 (2^6).
    IMAGE_MIN_DIM = 704
    IMAGE_MAX_DIM = 1408
    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported

    # Mean pixel values of image dataset (RGB)
    MEAN_PIXEL = np.array([116.86, 119.87, 119.94])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 20  # Max amount of GT boxes / image is 11

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100  # 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7  # 0.9

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3  # 0.3/0.5 by Mask-RCNN

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02 and lm=0.9, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimiser
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization (0.0001 Mask-RCNN paper)
    WEIGHT_DECAY = 0.0001

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        if self.GPU_COUNT > 0:
            self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        else:
            self.BATCH_SIZE = self.IMAGES_PER_GPU

        # Adjust step size based on batch size
        self.STEPS_PER_EPOCH = self.BATCH_SIZE * self.STEPS_PER_EPOCH

        # Input image size
        self.IMAGE_SHAPE = np.array(
            [self.IMAGE_MIN_DIM, self.IMAGE_MAX_DIM, 3])
        # MAX, MAX

        # Compute backbone size from input image size
        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
             for stride in self.BACKBONE_STRIDES])

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def to_txt(self, log_dir):
        """Save Configuration values in text file."""
        if 'self.file' not in locals():
            self.file = os.path.join(log_dir + "/config.txt")
        if not os.path.exists(self.file):
            with open(self.file, "x") as config_txt:
                for a in dir(self):
                    if not a.startswith("__") and not callable(getattr(self, a)):
                        config_txt.write("{:30} {}\n".format(a, getattr(self, a)))

    def save_time(self):
        """Save intermediate training time to config text file"""
        with open(self.file, "a") as config_txt:
            config_txt.write("{:30} {} @ {}\n".format(
                "intermediate_time", self.intermediate_time, datetime.datetime.now()))
