import os
import skimage
import random


from config import Config
from Custom_layers import *
from resources import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes
    BATCH_SIZE = 1


config = CocoConfig()

LOG_PATH = "./logs"
IMAGE_DIR = "../images"
if not os.path.exists(LOG_PATH):
    os.system('mkdir logs')

if not os.path.exists(os.path.join(LOG_PATH, 'mask_rcnn_coco.5')):
    utils.download_trained_weights(os.path.join(LOG_PATH, 'mask_rcnn_coco.5'))

from model import MaskRCNN as MRCNN

#tf.config.set_soft_device_placement(True)
#tf.debugging.enable_check_numerics()  # modo code reviewer
tf.debugging.experimental.enable_dump_debug_info(
    "./logs/tfdbg2_logdir",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)


mrcnn = MRCNN('inference', config, './models')
#mrcnn.keras_model.save('models/mrcnn_model.h5')
print(mrcnn.keras_model.summary())
mrcnn.load_weights(os.path.join(LOG_PATH, 'mask_rcnn_coco.5'), by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


file_names = next(os.walk(IMAGE_DIR))[2]
images = [
    skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
]

results = mrcnn.detect(images, verbose=0)
print(results)
