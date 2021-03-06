import os
import numpy as np

# from tqdm import tqdm
from utils import Dataset
from xml.etree import ElementTree


class LampPostDataset(Dataset):

    def load_dataset(self, img_dir, annos_dir, is_train=True):
        self.add_class("dataset", 1, "lamp post")

        images = os.listdir(img_dir)
        annots = os.listdir(annos_dir)
        for i, filename in enumerate(images):
            image_id = filename[:-4]

            if image_id + ".xml" in annots:
                # skip images after split if we build training set
                # split = int(0.4 * len(images))  # Test example dataset
                split = int(0.8 * len(images))  # For actual dataset
                if is_train and i > split:
                    continue
                # skip all images before split if we are building the test/val set
                if not is_train and i <= split:
                    continue

                # setting image file
                img_path = img_dir + filename

                # setting annotations file
                ann_path = annos_dir + image_id + '.xml'

                # adding images and annotations to dataset
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def extract_bboxes_VOC(self, image_id):
        """ Extract bounding boxes from a 'Pascal VOC'-format annotation file
        """
        filename = self.source_annotation_link(image_id)
        # load and parse the file
        tree = filename
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

        return boxes.astype(np.int32)

    # Return the path of the image
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)

        return info['path']
