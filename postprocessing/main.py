import os
import json
import cv2 as cv
import numpy as np

from ..mrcnn.Configs import Config
from ..mrcnn.model import MaskRCNN
from ..mrcnn.Dataset import load_images_dataset
from ..mrcnn.Utils.utilfunctions import download_dataset, unzip_dataset
from alites.measurement import calc_longest_diagonal_pca as measure_c3s

def main (data_set_url: str) -> str:

    # TODO: add weights path
    WEIGHTS_PATH = ""
    # TODO: confirm how dataset name should be
    DATASET_NAME = "alita_and_poros"
    DATASET_PATH = unzip_dataset(download_dataset(data_set_url))
    ANNOTATIONS_FILENAME = "_annotations.coco.json"
    ANNOTATIONS_PATH = os.path.join(DATASET_PATH, ANNOTATIONS_FILENAME)

    dataset = load_images_dataset(ANNOTATIONS_PATH, DATASET_PATH, "test")

    cfg_kwargs = {
        "num_classes": dataset.count_classes(),
        "name": DATASET_NAME,
    }
    config = Config(**cfg_kwargs)

    model = MaskRCNN(mode="inference", config=config)
    model.load_weights(filepath=WEIGHTS_PATH)

    # TODO: make sure which information exactly should be passed to the backend
    output_dict = {}

    # TODO: adapt code to get images from dataset and not from directory
    # TODO: adapt code to process masks correctly
    image_filenames = os.listdir(DATASET_PATH)

    for filename in image_filenames:

        img_path = os.path.join(DATASET_PATH, filename)
        img = cv.imread(img_path)

        # predict
        r = model.detect([img], verbose=0)[0]
        masks = r['masks']

        # TODO: add c3s classification

        masks_info = []
        masks = np.array([1])
        masks.reshape
        for mask in masks:
            contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            contour = contours[0]
            
            start, end, length = measure_c3s(contour)

            mask_info.append((start, end), length)

        output_dict[filename] = {
            'masks': [
                {
                    'class': 2,
                    'id': 2
                    'longest_diagonal': [start, end, size]
                }
            ]
            'image_dir': 
        }

    json_str = json.dumps(output_dict)
    return json_str


## mask array shape = (r, c, n_masks)
## convert 0's and 1's to 0's and 255's
#########################################

## Sample code from Mi
# image_ids = np.random.choice(dataset_train.image_ids, 1)

# for image_id in image_ids:
#   image = dataset_train.load_image(image_id)
#   image, image_meta, gt_class_ids, gt_boxes, gt_masks = load_image_gt(
#         dataset_train, config, image_id, augmentation=None,
#         use_mini_mask=False,
#         use_skimage_resize=False
#   )
#   gt_masks = np.where(gt_masks == 1, 255, 0)
#   maskered = cv.bitwise_and(image, image, mask=gt_masks[:, :, 0].astype(np.uint8))
#   maskered = cv.cvtColor(maskered, cv.COLOR_B