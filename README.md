# UNet-Keypointdetector

This repository consists of U-Net implementation to extract keypoints in an image.

**Requirements**
- pytorch 0.4.0
- pytorch-ignite 0.1.0
- imgaug 0.2.0
- opencv-python 3.4

**Data Preparation**
1. Organise the data in two folders-
   - train images folder
   - train annotation folder

2. Annotation files
Annotation files are structured in following json format:
```json

{
"file_path": "/home/ankit/Documents/Ankit-BackUp/Develop/vbtrack/data/multiple_markers_images/",
"file_name": "multiple_markers_1.png",
"image_shape": [480, 640, 3],
"object": [
{
"bbox_dimensions": [[205,313,221,294]],
"points_location": [[265,245],
                    [249,274],
                    [280,267],
                    [304,261]]
}
]
}
```
Bounding boxes are provided to facilitate the region of interest and are given as the input to the unet.

**Training**
1. Edit the config file:
```json
{
    "data": {
        "image_folder":         "",
        "annotation_folder":    "",
        "augment":              true,
        "bbox_scale_factor":     0.3,
        "negative_sample_ratio": 0.3
    },

    "mask": {
        "kernel_size":          5,
        "variance":             0.3
    },

    "train": {
        "learning_rate":        1e-5,
        "decay_type":           "step",
        "learning_rate_decay":  0.1,
        "epoch_decay":          10,
        "epochs":               100,
        "batch_size":           8,
        "val_split_ratio":      0.2,
        "model_save_path":      "",
        "dropout":              0.2,
        "log_dir":              ""
    },

    "unet": {
        "input_shape":          [128, 128, 3],
        "n_classes":            1,
        "start_ch":             32,
        "depth":                5,
        "batch_norm":           false,
        "up_mode":              "transpose",
        "merge_mode":           "add"
    },

}
```

The train section provides the parameters for the training of the unet and where to save the model and logs. In unet section you can define the structure of the unet to train. The `bbox_scale_factor` and `negative_sample_ratio` are to provide how much of the actual region of interest is provided to the input of unet and how many negative samples are to be given in the dataset respectively.

2. Run training
Update the json file and run training:

  `python train.py -c config.json`

3. Evaluate
Specify the U-Net structure in the `config_detector.json` and provide the path of images for the detections, as the cropped bounding box of the instrument marker.

  `python evaluate.py -c config_detector.json -p path_to_images`
