"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    train --dataset=../Master_Thesis_GvA_project/data/4_external --model=coco

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
"""

import os
import sys
import time
import numpy as np
import datetime

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

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"


############################################################
#  Configurations
############################################################

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "PanorAMS"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # COCO has 80 classes


############################################################
#  Dataset
############################################################

# noinspection SpellCheckingInspection
class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, class_ids=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports mapping classes from
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


# noinspection SpellCheckingInspection
def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
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
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file, 'last', or 'imagenet'")
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
                        help="Evaluate with test or validation set")

    args = parser.parse_args()

    # Configurations
    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0


        config = InferenceConfig()

    # Create model
    model = modellib.MaskRCNN(config=config, model_dir=args.logs)

    if config.GPU_COUNT:
        model = model.cuda()

    # Select weights file to load
    if args.model:
        if args.model.lower() == "last":
            # Find last trained weights
            model_dir, model_path = model.find_last()
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = config.IMAGENET_MODEL_PATH
        else:
            model_path = args.model
    else:
        model_path = ""

    # Train or evaluate
    if args.command == "train":
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

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE,
                          epochs=40,
                          layers='heads')

        # Training - Stage 2
        # Fine tune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train_model(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE,
                          epochs=120,
                          layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_val,
                          learning_rate=config.LEARNING_RATE / 10,
                          epochs=160,
                          layers='all')

    if args.command == "evaluate":
        # Change output to text file
        with open("{}/evaluate_{}-{:%Y%m%dT%H%M}.txt".format(model_dir, args.val_test,
                                                             datetime.datetime.now()), 'w') as f:
            sys.stdout = f

            print("Command: ", args.command)
            print("Model: ", args.model)
            print("Dataset: ", args.dataset)
            print("Logs: ", args.logs)
            print("Evaluate with:  {} set".format(args.val_test))

            config.display()

            # Load weights
            print("Loading weights ", model_path)
            model.load_weights(model_path)

            # Dataset used for evaluation
            dataset_val = CocoDataset()
            coco = dataset_val.load_coco(args.dataset, args.val_test, return_coco=True)
            dataset_val.prepare()
            print("Running COCO evaluation on {} images.".format(args.limit))
            evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
