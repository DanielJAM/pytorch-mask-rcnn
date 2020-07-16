import os
import numpy as np

from tqdm import tqdm
from utils import Dataset
from xml.etree import ElementTree


class LampPostDataset(Dataset):

    def load_dataset(self, data_dir, is_train=True):
        self.add_class("dataset", 1, "lamp post")

        images_dir = "../Master_Thesis_GvA_project/data/examples/images/"
        # images_dir = data_dir + '/PanorAMS_panoramas_GT/'
        annotations_dir = "../Master_Thesis_GvA_project/data/examples/examples_voc/lamp_post/"
        # annotations_dir = data_dir + '/PanorAMS_GT_pascal-VOC-absolute/'

        images = os.listdir(images_dir)
        for i, filename in tqdm(enumerate(images)):
            image_id = filename[:-4]

            # skip images after split if we build training set
            split = int(0.74 * len(images))
            if is_train and i > split:
                continue
            # skip all images before split if we are building the test/val set
            if not is_train and i <= split:
                continue

            # setting image file
            img_path = images_dir + filename

            # setting annotations file
            ann_path = annotations_dir + image_id + '.xml'

            # adding images and annotations to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file
    def extract_bboxes(self, image_id):
        filename = self.source_annotation_link(image_id)
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        box_annos = root.findall('.//bndbox')
        boxes = np.zeros([len(box_annos), 4], dtype=np.int32)
        for i, box in enumerate(box_annos):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = np.array([xmin, ymin, xmax, ymax])
            boxes[i] = coors

        # extract image dimensions
        # width = int(root.find('.//size/width').text)
        # height = int(root.find('.//size/height').text)

        return boxes.astype(np.int32)


    # Return the path of the image
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)

        return info['path']
