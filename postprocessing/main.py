import os
import json
import cv2 as cv
from ..mrcnn.Configs import Config
from ..mrcnn.model import MaskRCNN
from alites.measurement import calc_longest_diagonal_pca as measure_c3s


# TODO: add weights and images to inference path

WEIGHTS_PATH = ""
IMAGES_PATH = ""

kwargs = {
    "interpolation_method": "bicubic",
    # TODO: what config to use on inference?
}
config = Config(**kwargs)


model = MaskRCNN(mode="inference", config=config)
model.load_weights(filepath=WEIGHTS_PATH)

# TODO: make sure which information exactly should be passed to the backend
output_dict = {}

image_filenames = os.listdir(IMAGES_PATH)

for filename in image_filenames:

    img_path = os.path.join(IMAGES_PATH, filename)
    img = cv.imread(img_path)

    # predict
    r = model.detect([img], verbose=0)[0]
    masks = r['masks']

    # TODO: inspect mask format to make sure this is the correct way to handle the masks
    # TODO: add c3s classification

    masks_info = []
    for mask in masks:
        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contour = contours[0]
        
        start, end, length = measure_c3s(contour)

        mask_info.append((start, end), length)

    output_dict[filename] = {
        'masks': masks,
        'info': masks_info
    }

# TODO: make sure saving as json is the correct way to export to the backend
#       if yes, where is the file supposed to be?
with open('data.json', 'r') as output_json:
    json.dump(output_dict, output_json)
