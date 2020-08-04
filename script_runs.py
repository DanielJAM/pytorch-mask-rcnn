import train_test


# ONLY_TEST=1, STEPS_IS_LEN_TRAIN_SET=0, n_epochs=5, layer_string="5+", name="Faster_RCNN-", img_dir, annos_dir
train_test.run("../Master_Thesis_GvA_project/data/4_external/PanorAMS_GT_boxes_lichtmast_selection.csv",
               "../Master_Thesis_GvA_project/data/4_external/PanorAMS_panoramas_GT",
               0, 1, 100, "5+", "100e-")
