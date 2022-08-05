import os
import cv2 as cv
import random

from Custom_layers import *
from resources import utils
from ShapesConfig import ShapesDataset, ShapesConfig
from model import MaskRCNN
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


config = ShapesConfig()

LOG_PATH = "./logs"
IMAGE_DIR = "../images"
if not os.path.exists(LOG_PATH):
    os.system('mkdir logs')

if not os.path.exists(os.path.join(LOG_PATH, 'mask_rcnn_coco.5')):
    utils.download_trained_weights(os.path.join(LOG_PATH, 'mask_rcnn_coco.5'))

mrcnn = MaskRCNN('training', config, './models')
mrcnn.load_weights(os.path.join(LOG_PATH, 'mask_rcnn_coco.5'), by_name=True)


# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(450, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()


# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    #visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

mrcnn.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')