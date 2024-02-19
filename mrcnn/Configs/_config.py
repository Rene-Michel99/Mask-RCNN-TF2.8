from typing import Union, Callable, List, Tuple
import numpy as np


class Config(object):
    """Base configuration class. For custom configurations, pass the arguments in __init__
        or create a sub-class that inherits from this one and override properties
        that need to be changed.
       Params:
       - name: Name of the configuration, used for creating directory for training
       - optimizer: Optimizer for training, for now only SGD is allowed
       - gpu_count: Number of GPUs to use. When using only CPU sets to 1
       - images_per_gpu: Number of images to train on each GPU. A 12GB GPU can typically
         handle 2 images of 1024x1024px
       - class_names: Class names presents in the dataset. Including BG!
       - interpolation_method: Interpolation method used in resize, can be bicubic, bilinear and nearest
       - steps_per_epoch: Number of training steps per epoch. Doesn't need to match the
         size of the dataset.
       - validation_steps: Number of validation steps to run at the end of every training epoch
       - backbone: Backbone network architecture. Supported values are resnet50 and resnet101.
         Can also be a callable that should have the signature of model.resnet_graph. Use this
         obligate to set compute_backbone_shape as well.
       - compute_backbone_shape: Callable to backbone that calculate the shape of each layer of
         the FPN Pyramid. See model.compute_backbone_shapes
       - backbone_strides: The strides of each layer of the FPN Pyramid. These values
         are based on a Resnet101 backbone
       - fpn_classifier_fc_layers_size: Size of the fully-connected layers in the
         classification graph
       - top_down_pyramid_size: Size of the top-down layers used to build the
         feature pyramid
       - num_classes: Number of classification classes in dataset
       - rpn_anchor_scales: Length of square anchor side in pixels
       - rpn_anchor_ratios: Ratios of anchors at each cell (width/height).
         A value of 1 represents a square anchor, and 0.5 is a wide anchor
       - rpn_anchor_stride: Anchor stride, if 1 then anchors are created for
         each cell in the backbone feature map. If 2 then anchors are created
         for every other cell, and so on.
       - rpn_nms_threshold: Non-max suppression threshold to filter RPN proposals.
         You can increase this during training to generate more proposals      
       - rpn_train_anchors_per_image: How many anchors per image to use for RPN training
       - pre_nms_limit: ROIs kept after tf.nn.top_k and before non-maximum suppression
       - post_nms_rois_training: ROIs kept after non-maximum suppression training
       - post_nms_rois_inference: ROIs kept after non-maximum suppression inference   
       - use_mini_mask: If enabled, resizes instance masks to a smaller size to reduce
         memory load. Recommended when using high-resolution images
       - mini_mask_shape: (height, width) of the mini-mask
       - image_resize_mode: Input image resizing Generally, use the "square" resizing
         mode for training and predicting and it should work well in most cases. In
         this mode, images are scaled up such that the small side is = IMAGE_MIN_DIM,
         but ensuring that the scaling doesn't make the long side > IMAGE_MAX_DIM. Then
         the image is padded with zeros to make it a square so multiple images can be put
         in one batch. Available resizing modes:
         none:   No resizing or padding. Return the image unchanged.
         square: Resize and pad with zeros to get a square image
                 of size [max_dim, max_dim].
         pad64:  Pads width and height with zeros to make them multiples of 64.
                 If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
                 up before padding. IMAGE_MAX_DIM is ignored in this mode.
                 The multiple of 64 is needed to ensure smooth scaling of feature
                 maps up and down the 6 levels of the FPN pyramid (2**6=64).
         crop:   Picks random crops from the image. First, scales the image based
                 on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
                 size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
                 IMAGE_MAX_DIM is not used in this mode.
       - image_min_dim: Used in image_resize_mode
       - image_max_dim: Used in image_resize_mode
       - image_min_scale: Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
         up scaling. For example, if set to 2 then images are scaled up to double
         the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
         However, in 'square' mode, it can be overruled by IMAGE_MAX_DIM
       - image_channel_count: Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
         Changing this requires other changes in the code. See the WIKI for more
         details: https://github.com/matterport/Mask_RCNN/wiki
       - mean_pixel: Image mean (RGB)
       - train_rois_per_image: Number of ROIs per image to feed to classifier/mask heads
         The Mask RCNN paper uses 512 but often the RPN doesn't generate
         enough positive proposals to fill this and keep a positive:negative
         ratio of 1:3. You can increase the number of proposals by adjusting
         the RPN NMS threshold
       - roi_positive_ratio: Percent of positive ROIs used to train classifier/mask heads
       - pool_size: Output shape to use for pooling ROIs
       - mask_pool_size: Output shape for pooling ROIs of mask prediction
       - mask_shape: Shape of output mask, to change this you also need to change
         the neural network mask branch     
       - max_gt_instances: Maximum number of ground truth instances to use in one image
       - rpn_bbox_std_dev: Bounding box refinement standard deviation for RPN and final detections
       - bbox_std_dev: Bounding box refinement standard deviation for RPN and final detections
       - detection_max_instances: Max number of final detections
       - detection_min_confidence: Minimum probability value to accept a detected instance
       - detection_nms_threshold: Non-maximum suppression threshold for detection
       - learning_rate: Learning rate used in training, the Mask RCNN paper uses lr=0.02,
         but on TensorFlow it causes weights to explode. Likely due to differences in optimizer
       - learning_momentum: Learning momentum used in training
       - weight_decay: Weight decay regularization
       - rpn_class_loss_decay: Loss weights for more precise optimization
       - rpn_bbox_loss_decay: Loss weights for more precise optimization
       - mrcnn_class_loss_decay: Loss weights for more precise optimization
       - mrcnn_bbox_loss_decay: Loss weights for more precise optimization
       - mrcnn_mask_loss_decay: Loss weights for more precise optimization
       - use_rpn_rois: Use RPN ROIs or externally generated ROIs for training
         Keep this True for most situations. Set to False if you want to train
         the head branches on ROI generated by code rather than the ROIs from
         the RPN. For example, to debug the classifier head without having to
         train the RPN
       - train_bn: Train or freeze batch normalization layers
         None: Train BN layers. This is the normal mode
         False: Freeze BN layers. Good when using a small batch size
         True: (don't use). Set layer in training mode even when predicting
       - gradient_clip_norm: Gradient norm clipping
       - use_parallel_model: Use or not a parallel model, this requires a second GPU
       - use_mask_modifier: Use a mask method manipulator to make mask looks like polygon
         or other shape
       - verbose_mode: Define mode of the verbose, debug is default and will log almost every action
       - generate_lod: Define if will generate a log file with all actions executed. Default is False
    """

    def __init__(
            self,
            num_classes,                           # type: int
            name,                                  # type: str
            detection_min_confidence=0.75,         # type: float
            class_names=None,                      # type: List[str]
            images_per_gpu=1,                      # type: int
            steps_per_epoch=100,                   # type: int
            validation_steps=5,                    # type: int
            backbone="resnet101",                  # type: Union[str, Callable]
            optimizer="SGD",                       # type: str
            interpolation_method="bilinear",       # type: str
            gpu_count=1,                           # type: int
            compute_backbone_shape=None,           # type: Callable
            backbone_strides=None,                 # type: List[int]
            fpn_classifier_fc_layers_size=1024,     # type: int
            top_down_pyramid_size=256,             # type: int
            rpn_anchor_scales=None,                # type: Tuple[int]
            rpn_anchor_ratios=None,                # type: List[float]
            rpn_anchor_stride=1,                   # type: int
            rpn_nms_threshold=0.5,                 # type: float
            rpn_train_anchors_per_image=256,       # type: int
            pre_nms_limit=6000,                    # type: int
            post_nms_rois_training=2000,           # type: int
            post_nms_rois_inference=1000,          # type: int
            use_mini_mask=True,                    # type: bool
            mini_mask_shape=(56, 56),              # type: Tuple[int, int]
            image_resize_mode="square",            # type: str
            image_min_dim=800,                     # type: int
            image_max_dim=1024,                    # type: int
            image_min_scale=0,                     # type: float
            image_channel_count=3,                 # type: int
            mean_pixel=None,                       # type: List[float, float, float]
            train_rois_per_image=200,              # type: int
            roi_positive_ratio=0.33,               # type: float
            pool_size=7,                           # type: int
            mask_pool_size=14,                     # type: int
            mask_shape=None,                       # type: List[int, int]
            max_gt_instances=100,                  # type: int
            rpn_bbox_std_dev=None,                 # type: List[float, float, float, float]
            bbox_std_dev=None,                     # type: List[float, float, float, float]
            detection_max_instances=100,           # type: int
            detection_nms_threshold=0.3,           # type: float
            learning_rate=0.001,                   # type: float
            learning_momentum=0.9,                 # type: float
            weight_decay=0.0001,                   # type: float
            rpn_class_loss_decay=1.,               # type: float
            rpn_bbox_loss_decay=1.,                # type: float
            mrcnn_class_loss_decay=1.,             # type: float
            mrcnn_bbox_loss_decay=1.,              # type: float
            mrcnn_mask_loss_decay=1.,              # type: float
            use_rpn_rois=True,                     # type: bool
            train_bn=False,                        # type: bool
            gradient_clip_norm=5.0,                # type: float
            use_parallel_model=False,              # type: bool
            use_mask_modifier=None,                # type: Union[str, Callable]
            verbose_mode='debug',                  # type: str
            generate_log=False,                    # type: bool
    ):
        """Set values of computed attributes."""
        self.NUM_CLASSES = 1 + num_classes
        self.NAME = name
        self.IMAGES_PER_GPU = images_per_gpu
        self.CLASS_NAMES = class_names
        self.STEPS_PER_EPOCH = steps_per_epoch
        self.VALIDATION_STEPS = validation_steps
        self.INTERPOLATION_METHOD = interpolation_method
        self.BACKBONE = backbone
        self.DETECTION_MIN_CONFIDENCE = detection_min_confidence
        self.GPU_COUNT = gpu_count
        self.COMPUTE_BACKBONE_SHAPE = compute_backbone_shape
        self.FPN_CLASSIFIER_FC_LAYERS_SIZE = fpn_classifier_fc_layers_size
        self.TOP_DOWN_PYRAMID_SIZE = top_down_pyramid_size
        self.RPN_ANCHOR_STRIDE = rpn_anchor_stride
        self.RPN_NMS_THRESHOLD = rpn_nms_threshold
        self.RPN_TRAIN_ANCHORS_PER_IMAGE = rpn_train_anchors_per_image
        self.PRE_NMS_LIMIT = pre_nms_limit
        self.POST_NMS_ROIS_TRAINING = post_nms_rois_training
        self.POST_NMS_ROIS_INFERENCE = post_nms_rois_inference
        self.USE_MINI_MASK = use_mini_mask
        self.MINI_MASK_SHAPE = mini_mask_shape
        self.IMAGE_RESIZE_MODE = image_resize_mode
        self.IMAGE_MIN_DIM = image_min_dim
        self.IMAGE_MAX_DIM = image_max_dim
        self.IMAGE_MIN_SCALE = image_min_scale
        self.IMAGE_CHANNEL_COUNT = image_channel_count
        self.TRAIN_ROIS_PER_IMAGE = train_rois_per_image
        self.ROI_POSITIVE_RATIO = roi_positive_ratio
        self.POOL_SIZE = pool_size
        self.MASK_POOL_SIZE = mask_pool_size
        self.MAX_GT_INSTANCES = max_gt_instances
        self.DETECTION_MAX_INSTANCES = detection_max_instances
        self.DETECTION_NMS_THRESHOLD = detection_nms_threshold
        self.LEARNING_RATE = learning_rate
        self.LEARNING_MOMENTUM = learning_momentum
        self.WEIGHT_DECAY = weight_decay
        self.USE_RPN_ROIS = use_rpn_rois
        self.TRAIN_BN = train_bn
        self.GRADIENT_CLIP_NORM = gradient_clip_norm
        self.USE_PARALLEL_MODEL = use_parallel_model
        self.USE_MASK_MODIFIER = use_mask_modifier

        self.verbose_mode = verbose_mode
        self.generate_log = generate_log

        assert optimizer in ["SGD"], "Optimizer not allowed"
        self.OPTIMIZER = optimizer

        if use_mask_modifier:
            self.INTERPOLATION_METHOD = 'bilinear'

        if not backbone_strides:
            self.BACKBONE_STRIDES = [4, 8, 16, 32, 64]

        if not rpn_anchor_scales:
            self.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

        if not rpn_anchor_ratios:
            self.RPN_ANCHOR_RATIOS = [0.5, 1, 2]

        if not mean_pixel:
            self.MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

        if not mask_shape:
            self.MASK_SHAPE = [28, 28]

        if not rpn_bbox_std_dev:
            self.RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

        if not bbox_std_dev:
            self.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

        self.LOSS_WEIGHTS = {
            "rpn_class_loss": rpn_class_loss_decay,
            "rpn_bbox_loss": rpn_bbox_loss_decay,
            "mrcnn_class_loss": mrcnn_class_loss_decay,
            "mrcnn_bbox_loss": mrcnn_bbox_loss_decay,
            "mrcnn_mask_loss": mrcnn_mask_loss_decay
        }

        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([
                self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, self.IMAGE_CHANNEL_COUNT
            ])
        else:
            self.IMAGE_SHAPE = np.array([
                self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, self.IMAGE_CHANNEL_COUNT
            ])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def to_dict(self):
        return {a: getattr(self, a)
                for a in sorted(dir(self))
                if not a.startswith("__") and not callable(getattr(self, a))}

    def display(self):
        """Display Configuration values."""
        log = "\nConfigurations:"
        for key, val in self.to_dict().items():
            log += "{} {}\n".format(key, val)
        return log
