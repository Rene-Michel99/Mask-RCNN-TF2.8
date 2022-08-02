import os
import cv2 as cv
import random

from Custom_layers import *
from resources import utils
from config import Config
from model import MaskRCNN
from resources import visualize
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


config = ShapesConfig()

LOG_PATH = "./logs"
IMAGE_DIR = "../images"
if not os.path.exists(LOG_PATH):
    os.system('mkdir logs')

if not os.path.exists(os.path.join(LOG_PATH, 'mask_rcnn_coco.5')):
    utils.download_trained_weights(os.path.join(LOG_PATH, 'mask_rcnn_coco.5'))

mrcnn = MaskRCNN('training', config, './models')
mrcnn.load_weights(os.path.join(LOG_PATH, 'mask_rcnn_coco.5'), by_name=True)