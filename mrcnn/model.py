import os
import re
import json
import logging
import datetime
import numpy as np
import multiprocessing
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
from typing import List, Union, Tuple, Callable, Dict, Optional
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.callbacks import History

from .Dataset import Dataset
from .Configs import Config
from .Utils import utilfunctions
from .Utils.DataUtils import mold_image, compose_image_meta
from .Utils.RpnUtils import build_rpn_model
from .CustomLayers import (
    ROIPoolingLayer,
    GetAnchors,
    NormBoxesGraph,
    RPNClassLoss,
    RPNBboxLoss,
    MRCNNClassLossGraph,
    MRCNNBboxLossGraph,
    MRCNNMaskLossGraph
)
from .MRCNNComponents import RPNComponent, MaskSegmentationComponent, ResnetComponent
from .CustomKerasModel import MaskRCNNModel
from .CustomCallbacks import ClearMemory
from .DataGenerator import DataGenerator

# Requires TensorFlow 2.8+
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("2.8")


class MaskRCNN:
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode: str, config: Config, model_dir: str = '') -> None:
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config

        if not model_dir:
            self.model_dir = utilfunctions.get_root_dir_path() + '/logs'
        else:
            self.model_dir = model_dir

        self._log_dir = ""
        self._checkpoint_path = ""
        self.set_log_dir()
        self._build_logger()
        self.epoch = 0
        self._anchor_cache = {}
        self.is_compiled = False
        self.keras_model = self.build(mode=mode, config=config)

    def _build_logger(self) -> None:
        if self.config.verbose_mode.lower() != "debug":
            return

        self._logger = logging.getLogger('MaskRCNN')
        self._logger.setLevel(logging.DEBUG)

        if self.config.generate_log:
            level = logging.DEBUG
            if not os.path.exists(self._log_dir):
                os.system("mkdir '{}'".format(self._log_dir))
            path_to_log = os.path.join(self._log_dir, self.mode + '.log')
            handler = logging.FileHandler(filename=path_to_log, encoding='utf-8')
        else:
            level = logging.INFO
            handler = logging.StreamHandler()

        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    def rebuild_as(self, mode: str, config: Config = None):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.epoch = 0
        self._anchor_cache.clear()
        self.is_compiled = False

        tf.keras.backend.clear_session()
        del self.keras_model

        config = config if config is not None else self.config
        self.keras_model = self.build(mode=mode, config=config)

    @staticmethod
    def _build_rpn_model(
            config: Config,
            rpn_feature_maps: List[KL.Conv2D]
    ):
        """ Build rpn model.
        Wwe take the feature maps obtained in the previous step and apply a region proposal network (RPM).
        This basically predicts if an object is present in that region (or not).
        In this step, we get those regions or feature maps which the model predicts contain some object.

        Params:
        - config: Configuration object
        - rpn_feature_maps: List of Convolutional layers for RPN

        Returns: Tuple of
            - rpn_class_logits: KL.Lambda. Anchor classifier logits (before softmax)
            - rpn_class: KL.Activation. RPN Classes classifier
            - rpn_bbox: KL.Lambda. Deltas to be applied to anchors
        """
        # RPN Model
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs
        return rpn_class_logits, rpn_class, rpn_bbox

    def _get_anchors_layer(self, input_image, image_shape, batch_size):
        anchors = self._get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (batch_size,) + anchors.shape)
        anchors = GetAnchors(  # noqa
            anchors,
            name="anchors"
        )(input_image)

        return anchors

    def _build_training_architecture(
            self,
            config: Config,
            input_image: KL.Input,
            input_image_meta: KL.Input
    ) -> MaskRCNNModel:
        """Build trainining architecture of Mask R-CNN.

        Params:
        - config: Configuration object
        - input_image: Input layer of tensorflow for image
        - input_image_meta: Input layer of tensorflow for data about image

        Returns: An MaskRCNNModel object
        """

        # INPUT LAYERS
        # RPN GT
        input_rpn_match = KL.Input(
            shape=[None, 1], name="input_rpn_match", dtype=tf.int32
        )
        input_rpn_bbox = KL.Input(
            shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32
        )

        # Detection GT (class IDs, bounding boxes, and masks)
        # 1. GT Class IDs (zero padded)
        input_gt_class_ids = KL.Input(
            shape=[None], name="input_gt_class_ids", dtype=tf.int32
        )
        # 2. GT Boxes in pixels (zero padded)
        # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
        input_gt_boxes = KL.Input(
            shape=[None, 4], name="input_gt_boxes", dtype=tf.float32
        )
        # Normalize coordinates
        gt_boxes = NormBoxesGraph()([input_gt_boxes, K.shape(input_image)[1:3]])  # noqa
        # 3. GT Masks (zero padded)
        # [batch, height, width, MAX_GT_INSTANCES]
        if config.USE_MINI_MASK:
            input_gt_masks = KL.Input(
                shape=[config.MINI_MASK_SHAPE[0],
                       config.MINI_MASK_SHAPE[1], None],
                name="input_gt_masks", dtype=bool
            )
        else:
            input_gt_masks = KL.Input(
                shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                name="input_gt_masks", dtype=bool
            )

        # BACKBONE COMPONENT
        # Note that P6 is used in RPN, but not in the classifier heads.
        resnet_component = ResnetComponent(input_image, config)

        # RPN COMPONENT
        anchors = self._get_anchors_layer(input_image, config.IMAGE_SHAPE, config.BATCH_SIZE)
        rpn_component = RPNComponent(
            'training',
            input_image,
            resnet_component.rpn_feature_maps,
            config,
            anchors
        )

        # Class ID mask to mark class IDs supported by the dataset the image
        # came from.
        active_class_ids = KL.Lambda(
            lambda x: utilfunctions.parse_image_meta_graph(x)["active_class_ids"]
        )(input_image_meta)

        # ROI POOLING
        # Generate detection targets
        # Subsamples proposals and generates target outputs for training
        # Note that proposal class IDs, gt_boxes, and gt_masks are zero
        # padded. Equally, returned rois and targets are zero padded.
        rois, target_class_ids, target_bbox, target_mask = ROIPoolingLayer(  # noqa
            config,
            name="proposal_targets"
        )([rpn_component.target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

        # MASK SEGMENTATION COMPONENT
        # Network Heads
        mask_component = MaskSegmentationComponent(
            'training',
            rois,
            resnet_component.mrcnn_feature_maps,
            input_image_meta,
            config
        )

        output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

        # LOSSES
        # TODO: Can this losses be converted to a truly loss function from tensorflow?
        rpn_class_loss = RPNClassLoss(  # noqa
            name="rpn_class_loss"
        )([input_rpn_match, rpn_component.rpn_class_logits])
        rpn_bbox_loss = RPNBboxLoss(  # noqa
            config.IMAGES_PER_GPU,
            name="rpn_bbox_loss"
        )([input_rpn_bbox, input_rpn_match, rpn_component.rpn_bbox])
        class_loss = MRCNNClassLossGraph(  # noqa
            name="mrcnn_class_loss"
        )([target_class_ids, mask_component.mrcnn_class_logits, active_class_ids])
        bbox_loss = MRCNNBboxLossGraph(  # noqa
            name="mrcnn_bbox_loss"
        )([target_bbox, target_class_ids, mask_component.mrcnn_bbox])
        mask_loss = MRCNNMaskLossGraph(  # noqa
            name="mrcnn_mask_loss"
        )([target_mask, target_class_ids, mask_component.mrcnn_mask])

        # Model
        inputs = [
            input_image, input_image_meta, input_rpn_match,
            input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks
        ]
        if not config.USE_RPN_ROIS:
            inputs.append(rpn_component.input_rois)

        outputs = [
            rpn_component.rpn_class_logits, rpn_component.rpn_class, rpn_component.rpn_bbox,
            mask_component.mrcnn_class_logits, mask_component.mrcnn_class,
            mask_component.mrcnn_bbox, mask_component.mrcnn_mask,
            rpn_component.rpn_rois, output_rois, rpn_class_loss,
            rpn_bbox_loss, class_loss, bbox_loss, mask_loss
        ]
        return MaskRCNNModel(inputs, outputs, name='mask_rcnn')

    @staticmethod
    def _build_inference_architecture(
            config: Config,
            input_image: KL.Input,
            input_image_meta: KL.Input
    ) -> MaskRCNNModel:
        """Build trainining architecture of Mask R-CNN.

        Params:
        - config: Configuration object
        - input_image: Input layer of tensorflow for image
        - input_image_meta: Input layer of tensorflow for data about image

        Returns: An MaskRCNNModel object
        """
        # Anchors in normalized coordinates
        input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # BACKBONE
        resnet_component = ResnetComponent(input_image, config)

        # RPN
        anchors = input_anchors
        rpn_component = RPNComponent(
            'inference',
            input_image,
            resnet_component.rpn_feature_maps,
            config,
            anchors
        )

        # MASK SEGMENTATION
        mask_component = MaskSegmentationComponent(
            'inference',
            rpn_component.rpn_rois,
            resnet_component.mrcnn_feature_maps,
            input_image_meta,
            config
        )

        inputs = [input_image, input_image_meta, input_anchors]
        outputs = [
            mask_component.detections, mask_component.mrcnn_class,
            mask_component.mrcnn_bbox, mask_component.mrcnn_mask,
            rpn_component.rpn_rois, rpn_component.rpn_class, rpn_component.rpn_bbox
        ]

        return MaskRCNNModel(
            inputs,
            outputs,
            name='mask_rcnn'
        )

    def build(self, mode: str, config: Config) -> MaskRCNNModel:
        """Build Mask R-CNN architecture.

            Params:
            - mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
            - config: Configuration object of Mask R-CNN

            Returns: Object of MaskRCNNModel
        """
        assert mode in ['training', 'inference']
        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            log_message = '''Image size must be dividable by 2 at least 6 times
                            to avoid fractions when downscaling and upscaling.
                            For example, use 256, 320, 384, 448, 512, ... etc.'''
            self._logger.error(log_message)
            raise Exception(log_message)

        self._logger.info('Using config - {}'.format(self.config.display()))
        self._logger.info('Building Mask R-CNN in {} architecture'.format(mode))
        # Inputs
        input_image = KL.Input(
            shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image"
        )

        input_image_meta = KL.Input(
            shape=[config.IMAGE_META_SIZE],
            name="input_image_meta"
        )

        try:
            if mode == "training":
                model = self._build_training_architecture(
                    config, input_image, input_image_meta
                )
            else:
                model = self._build_inference_architecture(
                    config, input_image, input_image_meta
                )
        except Exception as ex:
            self._logger.error(
                'Error building Mask R-CNN architecture in mode {} - {}'.format(mode, ex),
                exc_info=True
            )
            raise ex

        # Add multi-GPU support.
        if config.GPU_COUNT > 1 and config.USE_PARALLEL_MODEL:
            from .parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def find_last(self) -> str:
        """Finds the last checkpoint file of the last trained model in the
        model directory.

        Returns: The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            self._logger.error(
                "Could not find model directory under {}".format(self.model_dir)
            )
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir)
            )
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            self._logger.error(
                "Could not find weight files in {}".format(dir_name)
            )
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name)
            )
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(
            self,
            init_with='',
            filepath: str = None,
            by_name=False
    ) -> None:
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        Params:
        - init_with: String. Type of weights to load automatically, can be coco,
        imagenet and last (last trained weights)
        - filepath: Optional String. Specify the path of weights to be loaded.
        if architecture don't match must set by_name to True
        - by_name: Optional Bool. Set true when architecture of weights don't match
        with Mask R-CNN architecture

        Returns: None
        """
        self._logger.info('Starting load weights - filepath: {} - type: {}'.format(
            filepath, init_with
        ))
        if not filepath and not init_with:
            raise Exception('Must be feeded one of the params')
        if not filepath:
            by_name = True
            assert init_with in ["coco", "imagenet", "last"],\
                "init_with para must be coco, imagenet or last for the last trained model"
            if init_with == "imagenet":
                self._logger.info('Downloading imagenet weights')
                filepath = self.get_imagenet_weights()
            elif init_with == "coco":
                # Load weights trained on MS COCO, but skip layers that
                # are different due to the different number of classes
                # See README for instructions to download the COCO weights
                self._logger.info('Downloading COCO weights')
                filepath = utilfunctions.download_trained_weights(filepath)
            elif init_with == "last":
                # Load the last model you trained and continue training
                filepath = self.find_last()

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        self._logger.info("Using weights {}".format(filepath))
        skip_mismatch = self.mode == 'training'
        try:
            self.keras_model.load_weights(filepath, by_name=by_name, skip_mismatch=skip_mismatch)
            # Update the log directory
            self.set_log_dir(filepath)
        except Exception as ex:
            self._logger.error(
                'Error when loading weights to MaskRCNNModel - {}'.format(ex),
                exc_info=True
            )
            raise ex

    def load_weights_h5py(self, filepath: str, by_name=False, exclude=None) -> None:
        """!Deprecated. Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        from tensorflow.python.keras.saving import hdf5_format

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        with h5py.File(filepath, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']

            # In multi-GPU training, we wrap the model. Get layers
            # of the inner model because they have the weights.
            keras_model = self.keras_model
            layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
                else keras_model.layers

            # Exclude some layers
            if exclude:
                layers = filter(lambda l: l.name not in exclude, layers)

            if by_name:
                hdf5_format.load_weights_from_hdf5_group_by_name(f, layers)
            else:
                hdf5_format.load_weights_from_hdf5_group(f, layers)

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self, include_top=False) -> str:
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file

        BASE_WEIGHTS_PATH = (
            "https://storage.googleapis.com/tensorflow/keras-applications/resnet/"
        )
        WEIGHTS_HASHES = {
            "resnet50": (
                "2cb95161c43110f7111970584f804107",
                "4d473c1dd8becc155b73f8504c6f6626",
            ),
            "resnet101": (
                "f1aeb4b969a6efcfb50fad2f0c20cfc5",
                "88cf7a10940856eca736dc7b7e228a21",
            ),
            "resnet152": (
                "100835be76be38e30d865e96f2aaae62",
                "ee4c566cf9a93f14d82f913c2dc6dd0c",
            ),
            "resnet50v2": (
                "3ef43a0b657b3be2300d5770ece849e0",
                "fac2f116257151a9d068a22e544a4917",
            ),
            "resnet101v2": (
                "6343647c601c52e1368623803854d971",
                "c0ed64b8031c3730f411d2eb4eea35b5",
            ),
            "resnet152v2": (
                "a49b44d1979771252814e80f8ec446f9",
                "ed17cf2e0169df9d443503ef94b23b33",
            ),
            "resnext50": (
                "67a5b30d522ed92f75a1f16eef299d1a",
                "62527c363bdd9ec598bed41947b379fc",
            ),
            "resnext101": (
                "34fb605428fcc7aa4d62f44404c11509",
                "0f678c91647380debd923963594981b3",
            ),
        }

        model_name = self.config.BACKBONE.replace('V2', '')
        if include_top:
            file_name = model_name + "_weights_tf_dim_ordering_tf_kernels.h5"
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = (
                model_name + "_weights_tf_dim_ordering_tf_kernels_notop.h5"
            )
            file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = get_file(
            file_name,
            BASE_WEIGHTS_PATH + file_name,
            cache_subdir="models",
            file_hash=file_hash,
        )
        return weights_path

    def _get_optimizer(self, learning_rate: float, momentum: float) -> Optimizer:
        """Get the optimizer from config. Only SGD and RMSprop is working
        with actual architecture.
        """
        optimizer = self.config.OPTIMIZER
        self._logger.info("Using {} optimizer".format(optimizer))
        if optimizer == 'SGD':
            return keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=momentum,
                clipnorm=self.config.GRADIENT_CLIP_NORM,
            )
        elif optimizer == 'ADAM':
            return keras.optimizers.Adam(
                learning_rate=learning_rate,
                clipnorm=1.,
                epsilon=1e-08,
            )
        elif optimizer == 'RMSPROP':
            return keras.optimizers.RMSprop(
                learning_rate=learning_rate,
                momentum=momentum,
                clipnorm=self.config.GRADIENT_CLIP_NORM,
            )

    def compile(self, learning_rate: float, momentum: float, limit_device=False):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        self._logger.info("Building compiler")
        if limit_device:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
                )
            except Exception as ex:
                self._logger.error('Error when limiting device - {}'.format(ex), exc_info=True)
                raise ex

        try:
            # Optimizer object
            optimizer = self._get_optimizer(learning_rate, momentum)

            # Add Losses
            loss_names = [
                "rpn_class_loss", "rpn_bbox_loss",
                "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"
            ]
            if not self.is_compiled:
                for name in loss_names:
                    layer = self.keras_model.get_layer(name)
                    if name == "mrcnn_class_loss":
                        calc_loss = (tf.reshape(
                            tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.), []
                        ))
                    else:
                        calc_loss = (tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.))
                    self.keras_model.add_loss(
                        calc_loss
                    )

            losses_functions = [None] * len(self.keras_model.outputs)
            # Add L2 Regularization
            # Skip gamma and beta weights of batch normalization layers.
            if self.config.OPTIMIZER == 'SGD' and not self.is_compiled:
                self.keras_model.add_loss(
                    lambda: tf.add_n([
                        keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(input=w), tf.float32)
                        for w in self.keras_model.trainable_weights
                        if 'gamma' not in w.name and 'beta' not in w.name]
                    )
                )
            else:
                losses_functions = [
                    "categorical_crossentropy" if output.name in loss_names else None
                    for output in self.keras_model.outputs
                ]
            # Compile
            self.keras_model.compile(
                optimizer=optimizer,
                loss=losses_functions
            )
            self.is_compiled = True
            self._logger.info("Model compiled successfully!")
        except Exception as ex:
            self._logger.error(
                "Error when building compiler - {}".format(ex),
                exc_info=True
            )
            raise ex

    def set_trainable(self, layer_regex: str, keras_model: MaskRCNNModel = None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            utilfunctions.log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                utilfunctions.log("{}{:20}   ({})".format(" " * indent, layer.name,
                                                          layer.__class__.__name__))

    def set_log_dir(self, model_path: str = None) -> None:
        """Sets the model log directory and epoch counter.

        Params:
        - model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self._log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(),
            now
        ))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self._checkpoint_path = os.path.join(self._log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self._checkpoint_path = self._checkpoint_path.replace(
            "*epoch*", "{epoch:04d}"
        )

    def _write_history(self, history: dict) -> None:
        file_path = os.path.join(
            self._log_dir, 'history{}.json'.format(self.epoch)
        )
        with open(file_path, 'w') as f:
            f.write(json.dumps(history))

    def train(
            self,
            train_dataset: Dataset,
            val_dataset: Dataset,
            learning_rate: float,
            epochs: int,
            layers: str,
            use_early_stopping=True,
            augmentation: Callable = None,
            custom_callbacks=None,
            use_clear_memory=False,
            debug=False,
    ) -> dict:
        """Train the model.

        Params:
        - train_dataset, val_dataset: Training and validation Dataset objects.
        - learning_rate: The learning rate to train with
        - epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        - layers: Allows selecting wich layers to train. It can be:
              A regular expression to match layer names to train
              One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        - augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
	    - custom_callbacks: Optional. Add custom callbacks to be called
	        with the keras fit_generator method. Must be list of type keras.callbacks.
        - no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.

        Returns: A dict with data about epochs histogram
        """
        assert self.mode == "training", "Create model in training mode."
        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = DataGenerator(train_dataset, self.config, shuffle=True,
                                        augmentation=augmentation)
        val_generator = DataGenerator(val_dataset, self.config, shuffle=True)

        # Create log_dir if it does not exist
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)

        # TODO: Move callbacks to config class
        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self._log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self._checkpoint_path,
                                            verbose=0, save_weights_only=True),

        ]

        if use_clear_memory:
            callbacks.append(ClearMemory())

        if use_early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3
            ))

        if debug:
            tf.config.set_soft_device_placement(True)
            tf.debugging.experimental.enable_dump_debug_info(
                self._log_dir + '/train-ops-logs',
                tensor_debug_mode="FULL_HEALTH"
            )

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        self._logger.info("Starting at epoch {}. LR={}".format(self.epoch, learning_rate))
        self._logger.info("Checkpoint Path: {}".format(self._checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name == 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        try:
            history = self.keras_model.fit(
                train_generator,
                epochs=epochs,
                steps_per_epoch=self.config.STEPS_PER_EPOCH,
                callbacks=callbacks,
                validation_data=val_generator,
                validation_steps=self.config.VALIDATION_STEPS,
                max_queue_size=100,
                workers=workers,
                use_multiprocessing=True,
                verbose=1,
                batch_size=self.config.BATCH_SIZE,
            )
            self.epoch = max(self.epoch, epochs)
            self._write_history(history.history)
            return history.history
        except Exception as ex:
            self._logger.error('Error when training Mask R-CNN {}'.format(ex), exc_info=True)
            raise ex

    def train_directly(self, x, y, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None) -> History:
        """!Deprecated. Use train instead."""
        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Create log_dir if it does not exist
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self._log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self._checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        self._logger.info("Starting at epoch {}. LR={}".format(self.epoch, learning_rate))
        self._logger.info("Checkpoint Path: {}".format(self._checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name == 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        history = self.keras_model.fit(
            x,
            y,
            batch_size=self.config.BATCH_SIZE,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
            verbose=1
        )
        self.epoch = max(self.epoch, epochs)
        return history

    def mold_inputs(self, images: List[list]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.

        Params:
        - images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns: 3 Numpy matrices:
        - molded_images: [N, h, w, 3]. Images resized and normalized.
        - image_metas: [N, length of meta data]. Details about each image.
        - windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        self._logger.info('Molding input images')
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            molded_image, window, scale, padding, crop = utilfunctions.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE
            )
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        self._logger.info('Input image molded successfully!')
        return molded_images, image_metas, windows

    def unmold_detections(
            self,
            detections: np.ndarray,
            mrcnn_mask: np.ndarray,
            original_image_shape: list,
            image_shape: list,
            window: list
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        Params:
        - detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        - mrcnn_mask: [N, height, width, num_classes]
        - original_image_shape: [H, W, C] Original image shape before resizing
        - image_shape: [H, W, C] Shape of the image after resizing and padding
        - window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns: Tuple of
        - boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        - class_ids: [N] Integer class IDs for each bounding box
        - scores: [N] Float probability scores of the class_id
        - masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utilfunctions.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utilfunctions.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utilfunctions.unmold_mask(
                masks[i], boxes[i], original_image_shape,
                interpolation_method=self.config.INTERPOLATION_METHOD,
                use_skimage_resize=self.config.USE_MASK_MODIFIER,
                skimage_resize_order=0 if self.config.USE_MASK_MODIFIER else 1,
                mask_modifier=self.config.USE_MASK_MODIFIER
            )
            full_masks.append(full_mask)

        full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def detect(self, images, verbose=0) -> Dict[str, np.ndarray]:
        """Runs the detection pipeline.

        Params:
        - images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        - rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        - class_ids: [N] int class IDs
        - scores: [N] float probability scores for the class IDs
        - masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        self._logger.info("Processing {} images".format(len(images)))

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            if g.shape == image_shape:
                log_msg = '''After resizing, all images must have the same size.
                Check IMAGE_RESIZE_MODE and image sizes.'''
                self._logger.info(log_msg)
                raise Exception(log_msg)

        # Anchors
        anchors = self._get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        self._logger.info("molded_images data ->  shape: {} - min: {} - max: {}".format(
            molded_images.shape, molded_images.min(), molded_images.max())
        )
        self._logger.info("image_metas: {}".format(image_metas))
        self._logger.info("anchors: {}".format(anchors))

        # Run object detection
        return self._process_detections(
            images, molded_images,
            image_metas, anchors, windows=windows, verbose=verbose
        )

    def _process_detections(
            self,
            images: np.ndarray,
            molded_images: np.ndarray,
            image_metas: np.ndarray,
            anchors: np.ndarray,
            windows: np.ndarray = None,
            verbose=0
    ) -> Dict[str, np.ndarray]:
        self._logger.info('Inserting input data into Mask RCNN')
        try:
            detections, _, _, mrcnn_mask, _, _, _ = self.keras_model.predict(
                [molded_images, image_metas, anchors], verbose=verbose
            )
        except Exception as ex:
            self._logger.error(
                'Error when calling predict from MaskRCNNModel - {}'.format(ex),
                exc_info=True,
            )
            raise ex
        self._logger.info('Predicted successfully!')

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks = self.unmold_detections(
                detections[i], mrcnn_mask[i], image.shape,
                molded_images[i].shape, windows[i]
            )
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def detect_molded(
            self,
            molded_images: np.ndarray,
            image_metas: np.ndarray,
            verbose=0
    ) -> Dict[str, np.ndarray]:
        """Runs the detection pipeline, but expect inputs that are
        molded already. Used mostly for debugging and inspecting
        the model.

        molded_images: List of images loaded using load_image_gt()
        image_metas: image meta data, also returned by load_image_gt()

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.config.BATCH_SIZE,\
            "Number of images must be equal to BATCH_SIZE"

        if verbose:
            utilfunctions.log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                utilfunctions.log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Anchors
        anchors = self._get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            utilfunctions.log("molded_images", molded_images)
            utilfunctions.log("image_metas", image_metas)
            utilfunctions.log("anchors", anchors)

        # Run object detection
        return self._process_detections(
            molded_images, molded_images,
            image_metas, anchors
        )

    def _get_anchors(self, image_shape: list):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = utilfunctions.compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utilfunctions.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utilfunctions.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def ancestor(self, tensor: tf.Tensor, name: str, checked: list = None) -> Optional[tf.Tensor]:
        """Finds the ancestor of a TF tensor in the computation graph.

        Params:
        - tensor: TensorFlow symbolic tensor.
        - name: Name of ancestor tensor to find
        - checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self) -> list:
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for layer in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            trainable_layer = self.find_trainable_layer(layer)
            # Include layer if it has weights
            if trainable_layer.get_weights():
                layers.append(trainable_layer)
        return layers
