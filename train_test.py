"""
@Daniel Maaskant
"""


import custom_dataset
import model as modellib
import os
import time
import torch
import visualize


def run(img_dir, annos_dir, ONLY_TEST=1, STEPS_IS_LEN_TRAIN_SET=0, n_epochs=5, layer_string="5+", name="Faster_RCNN-"):
    """ heads: The RPN, classifier and mask heads of the network
        all: All the layers
        3+: Train Resnet stage 3 and up
        4+: Train Resnet stage 4 and up
        5+: Train Resnet stage 5 and up

        img_dir: path to directory containing images
        annos_dir: path to directory containing annotations """

    # torch.backends.cudnn.benchmark = True

    start_time = time.process_time()
    print("start time time(s): ", round(start_time, 2))

    # CONFIGURATION
    import config
    config = config.Config()
    config.NAME = name
    config.display()

    # TEST SET
    test_set = custom_dataset.LampPostDataset()
    test_set.load_dataset(img_dir, annos_dir, is_train=False)
    test_set.prepare()

    if not ONLY_TEST:
        # TRAINING SET
        train_set = custom_dataset.LampPostDataset()
        train_set.load_dataset(img_dir, annos_dir, is_train=True)
        train_set.prepare()

        print("Train: %d, Test: %d images" % (len(train_set.image_ids), len(test_set.image_ids)))

        if STEPS_IS_LEN_TRAIN_SET:
            config.STEPS_PER_EPOCH = len(train_set.image_info)

        data_time = time.process_time()
        print("load data time(s): ", round(data_time - start_time, 2), "total elapsed: ", round(data_time, 2))

        # LOAD MODEL
        model = modellib.MaskRCNN(config=config, model_dir='./models/')

        load_model_time = time.process_time()
        print("loading model time(s): ", round(load_model_time - data_time, 2), "total elapsed: ",
              round(load_model_time, 2))

        # LOAD WEIGHTS
        model.load_weights('./models/mask_rcnn_coco.pth',
                           callback=True)  # exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
        # "mrcnn_bbox", "mrcnn_mask"]

        load_weights_time = time.process_time()
        print("loading weights time(s): ", round(load_weights_time - load_model_time, 2), "total elapsed: ",
              round(load_weights_time, 2))

        # Save final config before start training
        config.to_txt(model.log_dir)

        # TRAIN MODEL
        # train heads with higher lr to speedup the learning
        model.train_model(train_set, test_set, learning_rate=2 * config.LEARNING_RATE, epochs=n_epochs,
                          layers=layer_string)

        train_time = time.process_time()
        print("training time(s): ", round((train_time - load_weights_time) / 60, 2), "total minutes elapsed: ",
              round(train_time, 2))

    # TEST MODEL
    modellib.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = modellib.MaskRCNN(config=config, models_dir='./models')

    # loading the trained weights of the custom dataset
    # last_model = model.find_last()[1]
    last_model = "./models/resnet50_imagenet.pth"
    print("loading model: ", last_model)
    model.load_weights(last_model)

    # Delete test model log directory
    os.rmdir(model.log_dir)

    image_id = 3
    # 1 = TMX7316010203-001499_pano_0000_001233 - only a hanging lamp post
    # 2 = TMX7316010203-001209_pano_0000_002760 - on the right, behind/above blue car
    # 3 = TMX7316010203-001187_pano_0000_002097 - clearly in the middle (old one) and further down the road on the right
    image, image_meta, gt_class_id, gt_bbox = modellib.load_image_gt(test_set, config, image_id)
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
