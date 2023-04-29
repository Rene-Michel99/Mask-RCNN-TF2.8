import os
import random

from CustomLayers import *
from resources import utils, visualize
from Configs.ShapesConfig import ShapesDataset, ShapesConfig
from resources.Data_utils import load_image_gt
from model import MaskRCNN
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # comentar em caso de usar a GPU


config = ShapesConfig()

LOG_PATH = "../logs"
IMAGE_DIR = "../../images"
if not os.path.exists(LOG_PATH):
    os.system('mkdir logs')

if not os.path.exists(os.path.join(LOG_PATH, 'mask_rcnn_coco.5')):
    utils.download_trained_weights(os.path.join(LOG_PATH, 'mask_rcnn_coco.5'))

mrcnn = MaskRCNN('training', config, '../models')
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
            epochs=2,
            layers='all')

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = MaskRCNN(mode="inference",
                 config=inference_config,
                 model_dir='./logs/train')

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
#model_path = model.find_last()

# Load trained weights
trained_model = os.path.join('./logs', 'train', 'train', 'mask_rcnn_shapes_0001.h5')
model.load_weights(trained_model, by_name=True)

# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)


#visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
 #                           dataset_train.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'])