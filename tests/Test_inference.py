import os
import cv2 as cv
import random

from src.Configs import CocoConfig
from model import MaskRCNN
from src.Utils import visualize

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 1
    DETECTION_MIN_CONFIDENCE = 0.75


config = InferenceConfig()

LOG_PATH = "../logs"
IMAGE_DIR = "./images"

if not os.path.exists(IMAGE_DIR):
    raise ValueError("Could not find image directory.")

utils.download_trained_weights()


#tf.debugging.enable_check_numerics()  # modo code reviewer
'''tf.debugging.experimental.enable_dump_debug_info(  # modo debug
    "./logs/tfdbg2_logdir",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)'''


mrcnn = MaskRCNN('inference', config, '../models')
#mrcnn.keras_model.save('models/mrcnn_model.h5')   # descomentar linha para salvar modelo e testar a função
#print(mrcnn.keras_model.summary())
mrcnn.load_weights(by_name=True)

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


images = []
for filename in os.listdir(IMAGE_DIR):
    image = cv.imread(os.path.join(IMAGE_DIR, filename))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    images.append(image)

chosen = random.choice(images)
results = mrcnn.detect([chosen], verbose=0)
r = results[0]

visualize.display_instances(chosen, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
