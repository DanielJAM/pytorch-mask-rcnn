"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Edited by Daniel Maaskant

------------------------------------------------------------

Usage: import the module, or run from the command line as such:

    # Train a new model from scratch
    train --dataset=../Master_Thesis_GvA_project/data/4_external --model=none

    # Train a new model starting from pre-trained COCO weights
    TODO: fix COCO .pth to match dimensions of this network
    CURRENTLY NOT WORKING: train --dataset=../Master_Thesis_GvA_project/data/4_external --model=coco

    # Train a new model starting from ImageNet weights
    train --dataset=../Master_Thesis_GvA_project/data/4_external --model=imagenet

    # Continue training a model that you had trained earlier
    train --dataset=../Master_Thesis_GvA_project/data/4_external --model=/path/to/weights.h5

    # Continue training the last model you trained
    train --dataset=../Master_Thesis_GvA_project/data/4_external --model=last


    # Run COCO evaluation with validation set on last trained model
    evaluate --dataset=../Master_Thesis_GvA_project/data/4_external --model=last --val_test=validation

    # Run COCO evaluation with test set
    evaluate --dataset=../Master_Thesis_GvA_project/data/4_external --model=last --val_test=test

    # Close to deterministic behaviour by setting seed for both train and evaluate
    --random=1
"""

import datetime
from distutils.util import strtobool
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import time
import torch

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from config import Config
import utils
import model as modellib
import visualize

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.pth")
IMAGENET_MODEL_PATH = os.path.join(ROOT_DIR, "models/resnet50_imagenet.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"


############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, class_ids=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: XX Not implemented yet. Supports mapping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        coco_file = COCO("{}/GT_{}_set(pano_id-int).json".format(dataset_dir, subset))

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco_file.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for i in class_ids:
                image_ids.extend(list(coco_file.getImgIds(catIds=[i])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco_file.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("PanorAMS", i, coco_file.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "PanorAMS", image_id=i,
                path=coco_file.imgs[i]['file_name'],
                width=coco_file.imgs[i]["width"],
                height=coco_file.imgs[i]["height"],
                annotations=coco_file.loadAnns(coco_file.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco_file

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores):
    """Arrange results to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "PanorAMS"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, display, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with validation data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    print("Running COCO evaluation on {} images.".format(len(image_ids)))

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[i]["id"] for i in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image])[0]
        t_prediction += (time.time() - t)

        # Visualize result
        if display == "True":
            visualize.display_instances(image, r['rois'], r['class_ids'], dataset.class_names, r['scores'],
                                        title="Predictions", figsize=image.shape[:2])

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.params.iouThrs = np.linspace(.25, 0.5, int(np.round((0.5 - .25) / .05)) + 1, endpoint=True)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}s. Average {}s/image".format(
        round(t_prediction, 2), round(t_prediction / len(image_ids), 2)))
    print("Total time: ", round(time.time() - t_start, 2), "s\n")


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on a MS-COCO format dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO format dataset')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file, 'none', 'last', or 'imagenet'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--val_test', required=False,
                        default='validation',
                        metavar='"validation" or "test"',
                        help="Evaluate with test or validation set (default=validation)")
    parser.add_argument('--random', required=False,
                        default=None,
                        metavar='Any integer',
                        help='Set random seed for consistent results (default=None)')
    parser.add_argument('--schedule', required=False,
                        default='example',
                        metavar='"example", "all", "3+", "4+", "heads"',
                        help='specify training schedule (default=example)')
    parser.add_argument('--display', required=False,
                        default=False,
                        metavar='boolean',
                        help='Whether to display detection results per image.')

    args = parser.parse_args()

    # Set random seed
    if type(args.random) is int:
        seed = int(args.random)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        print("Random seed PyTorch, NumPy, and random set to {}".format(args.random))

    # Configurations
    if args.command == "train":
        config = Config()

        # Add starting model name to log folder
        if isinstance(args.model, str):
            start_model_name = args.model
            if args.model[-3:] == 'pth':
                split_path = start_model_name.split('/')
                start_model_name = os.path.join(split_path[-2], split_path[-1])  # add last folder and model name
        else:
            start_model_name = ""
        config.NAME = config.NAME + "_" + start_model_name + "-"
    else:
        class TempConfig(Config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0

        config = TempConfig()
    # Save run config commands for reference
    config.RUN_CONFIG = args.__dict__

    # Create model
    modellib.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # modellib.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # modellib.device = torch.device("cpu")
    # For some reason parallel running anything doesn't work with these settings when training layers "3+"
    model = modellib.MaskRCNN(config=config, models_dir=args.logs)

    # if config.GPU_COUNT > 1:
    #     model = torch.nn.DataParallel(model)  For being able to use multiple GPUs, but this requires rewriting a lot
    model.to(modellib.device)

    # Select weights file to load
    if isinstance(args.model, str):
        model_command = args.model.lower()
        if model_command == "last":
            # Find last trained weights
            model.model_dir, model_path = model.find_last()
        elif model_command[-3:] == 'pth':
            model_path = args.model
            model.model_dir = model_path.split(os.path.basename(model_path))[0]
        elif model_command == "coco":
            # Start from COCO trained weights - not working yet
            model_path = COCO_MODEL_PATH
            model.model_dir = os.path.join(model_path.split(os.path.basename(model_path))[0], model_command)
        elif model_command == "imagenet":
            # Start from ImageNet trained weights
            model_path = IMAGENET_MODEL_PATH
            model.model_dir = os.path.join(model_path.split(os.path.basename(model_path))[0], model_command)
        else:
            model_path = args.model
    else:
        model_path = ""

    # Train or evaluate
    if args.command == "train":
        start_time = time.process_time()

        print("Command: ", args.command)
        print("Model: ", args.model)
        print("Dataset: ", args.dataset)
        print("Logs: ", args.logs)
        print("Evaluate with:  {} set".format(args.val_test))

        config.display()

        # Load weights
        print("Loading weights ", model_path)
        model.load_weights(model_path)

        # Save final config before start training
        config.to_txt(model.log_dir)

        # Training dataset
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "training")  # otherwise loads all Coco class ids
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        dataset_val.load_coco(args.dataset, "validation")
        dataset_val.prepare()

        # TRAINING SCHEDULES
        # at 75 % reduce lr 10 fold
        if args.schedule == 'example':
            # Training - Stage 1
            print("Training network heads")
            model.train_model(dataset_train, dataset_val,
                              learning_rate=config.LEARNING_RATE,
                              epochs=40,
                              layers='heads', seed=args.random)

            # Training - Stage 2
            # Fine tune layers from ResNet stage 4 and up
            print("Fine tune Resnet stage 4 and up")
            model.train_model(dataset_train, dataset_val,
                              learning_rate=config.LEARNING_RATE,
                              epochs=120,
                              layers='4+', seed=args.random)

            # Training - Stage 3
            # Fine tune all layers
            print("Fine tune all layers - lr / 10")
            model.train_model(dataset_train, dataset_val,
                              learning_rate=config.LEARNING_RATE / 10,
                              epochs=160,
                              layers='all', seed=args.random)

        elif args.schedule == 'all':
            # Fine tune all layers, use with pre-trained imagenet or no pre-trained network
            print("Fine tune all layers")
            model.train_model(dataset_train, dataset_val,
                              learning_rate=config.LEARNING_RATE,
                              epochs=120,
                              layers='all', seed=args.random)

            print("Fine tune all layers - lr / 10")
            model.train_model(dataset_train, dataset_val,
                              learning_rate=config.LEARNING_RATE / 10,
                              epochs=160,
                              layers='all', seed=args.random)

        elif args.schedule == '3+':
            # Pre-trained imagenet schedule mask-rcnn
            print("Fine tune Resnet stage 3 and up")
            model.train_model(dataset_train, dataset_val,
                              learning_rate=config.LEARNING_RATE,
                              epochs=120,
                              layers='3+', seed=args.random)
            print("Fine tune Resnet stage 3 and up, lr / 10")
            model.train_model(dataset_train, dataset_val,
                              learning_rate=config.LEARNING_RATE / 10,
                              epochs=160,
                              layers='3+', seed=args.random)

        end_time = time.process_time()
        with open(config.file, "a") as f:
            f.write("\nTotal time elapsed: {} hours\n".format(round((end_time - start_time) / 3600.0, 2)))

    elif args.command == "evaluate":
        # TODO: get config settings from config file
        # Read config file and convert back to original variable format
        # Anchor ratios, stride, scales etc. have to be the same for evaluation as for training.
        config_file = os.path.join(model.model_dir, "config.txt")
        with open(config_file) as f:
            config_list = f.readlines()
        config_list = config_list[5:]  # Remove backbone shapes
        for line in config_list:
            line_split = line.split(None, 1)
            var_name = line_split[0].strip()
            var_value = line_split[1].strip()
            if ',' in var_value:  # if it's a list
                var_value = var_value[1:-1]
                if '.' in var_value:
                    var_value = [float(x.replace(',', '')) for x in var_value.split()]
                else:
                    var_value = [int(x.replace(',', '')) for x in var_value.split()]
            else:
                temp = var_value.split()
                start_char = ord(var_value[0])
                if len(temp) > 1:  # list with ' ' separator
                    var_value = var_value[1:-1]  # get rid of brackets
                    if '.' in var_value:
                        var_value = np.fromstring(var_value, dtype=float, sep=' ')
                    else:
                        var_value = np.fromstring(var_value, dtype=int, sep=' ')
                elif start_char > ord('9') or start_char < ord('0'):  # text values
                    if var_value[0] == "T" or var_value == "F":
                        var_value = bool(strtobool(var_value))
                else:  # single digit
                    if '.' in var_value:
                        var_value = float(var_value)
                    else:
                        var_value = int(var_value)
            exec("config." + var_name + " = var_value")

        config.__init__()

        model.load_weights(model_path)

        # Change output to text file
        with open("{}/evaluate_{}-{:%Y%m%dT%H%M}.txt".format(model.model_dir, args.val_test,
                                                             datetime.datetime.now()), 'w') as file:
            sys.stdout = file

            print("Command: ", args.command)
            print("Model: ", args.model)
            print("Dataset: ", args.dataset)
            print("Logs: ", args.logs)
            print("Evaluate with:  {} set".format(args.val_test))

            config.display()
            print("Random seed PyTorch, NumPy, and random set to: {}".format(args.random))

            # Load weights
            print("Loading weights ", model_path)

            # Dataset used for evaluation
            dataset_val = CocoDataset()
            coco = dataset_val.load_coco(args.dataset, args.val_test, return_coco=True)
            dataset_val.prepare()
            evaluate_coco(model, dataset_val, coco, args.display, "bbox", limit=int(args.limit))

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
