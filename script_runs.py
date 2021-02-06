"""
@Daniel Maaskant
"""


import train_test


# img_dir, annos_dir, ONLY_TEST=1, STEPS_IS_LEN_TRAIN_SET=0, n_epochs=5, layer_string="5+", name="Faster_RCNN-"
train_test.run("../Master_Thesis_GvA_project/data/4_external/PanorAMS_panoramas_GT/",
               "../Master_Thesis_GvA_project/data/4_external/PanorAMS_GT_pascal-VOC_selection-50/",
               1, 1, 100, "all", "100e_selection50-lr00001")  # 5+
