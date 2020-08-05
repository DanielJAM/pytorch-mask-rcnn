import train_test


# annos_dir, img_dir, ONLY_TEST=1, STEPS_IS_LEN_TRAIN_SET=0, n_epochs=5, layer_string="5+", name="Faster_RCNN-"
train_test.run("../Master_Thesis_GvA_project/data/4_external/PanorAMS_panoramas_GT/",
               "../Master_Thesis_GvA_project/data/4_external/PanorAMS_GT_pascal-VOC_selection-50/",
               0, 1, 100, "5+", "100e_selection50-")
