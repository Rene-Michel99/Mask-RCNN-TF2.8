import os
import re
import datetime
from collections import OrderedDict
import multiprocessing
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

from resources import utils
from Custom_layers.Common import norm_boxes_graph
from resources.resnet_utils import resnet_graph
from resources.Data_utils import *
from resources.RPN_utils import *
from Custom_layers import (
    ProposalLayer,
    DetectionLayer,
    DetectionTargetLayer,
    GetAnchors,
    NormBoxesGraph,
    RPNClassLoss,
    RPNBboxLoss,
    MRCNNClassLossGraph,
    MRCNNBboxLossGraph,
    MRCNNMaskLossGraph
)

# Requires TensorFlow 2.8+
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("2.2")
#tf.compat.v1.disable_eager_execution()


class MaskRCNN:
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.epoch = 0
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(
            shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = NormBoxesGraph()([input_gt_boxes, input_image])
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
                                                train_bn=config.TRAIN_BN)
        else:
            _, C2, C3, C4, C5 = resnet_graph(input_image, config.BACKBONE,
                                             stage5=True, train_bn=config.TRAIN_BN)
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # A hack to get around Keras's bad support for constants
            anchors = GetAnchors(
                np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape),
                name="anchors")(input_image)
        else:
            anchors = input_anchors

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

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(
                lambda x: utils.parse_image_meta_graph(x)["active_class_ids"]
                )(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.int32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask =\
                DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])
            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            '''rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
             rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
               [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])
                '''
            rpn_class_loss = RPNClassLoss(name="rpn_class_loss")([input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = RPNBboxLoss(config.IMAGES_PER_GPU, name="rpn_bbox_loss")([input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = MRCNNClassLossGraph(name="mrcnn_class_loss")([target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = MRCNNBboxLossGraph(name="mrcnn_bbox_loss")([target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = MRCNNMaskLossGraph(name="mrcnn_mask_loss")([target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [input_image, input_image_meta,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES,
                                              train_bn=config.TRAIN_BN)

            model = KM.Model([input_image, input_image_meta, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from .parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        skip_mismatch = self.mode == 'training'
        keras_model = self.keras_model
        keras_model.load_weights(filepath, by_name=by_name, skip_mismatch=skip_mismatch)

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                 'releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        #self.keras_model.metrics_tensors = []
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        #self.keras_model._losses = []
        #self.keras_model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss",  "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        '''for name in loss_names:
            layer = self.keras_model.get_layer(name)
            #if layer.output in self.keras_model.losses:
             #   continue
            self.keras_model.add_loss(
                lambda: tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.)
            )'''

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        self.keras_model.add_loss(
            lambda: tf.add_n([
                keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(input=w), tf.float32)
                for w in self.keras_model.trainable_weights
                if 'gamma' not in w.name and 'beta' not in w.name])
        )

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs)
        )
        # Add metrics for losses
        '''for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)

            loss = KL.Lambda(lambda x: tf.reduce_mean(input_tensor=x, keepdims=True)
            )(layer.output)
            #self.keras_model.metrics_tensors.append(loss)
            self.keras_model.add_metric(
                loss,
                name=name,
                aggregation='mean'
            )'''

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            utils.log("Selecting layers to train")

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
                utils.log("{}{:20}   ({})".format(" " * indent, layer.name,
                                                  layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
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
#        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
#            self.config.NAME.lower(), now))
        self.log_dir = "logs/train"

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
	    custom_callbacks: Optional. Add custom callbacks to be called
	        with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
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
        train_generator = self._data_generator(
            train_dataset, self.config, shuffle=True,
            augmentation=augmentation, batch_size=self.config.BATCH_SIZE,
            no_augmentation_sources=no_augmentation_sources, detection_targets=True,
            random_rois=1)
        val_generator = self._data_generator(
            val_dataset, self.config, shuffle=True,
            batch_size=self.config.BATCH_SIZE,
            detection_targets=True, random_rois=1)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        utils.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        utils.log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)
        print("Model Compiled Successfully")

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name == 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        history = self.keras_model.fit(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
            verbose=1
        )
        self.epoch = max(self.epoch, epochs)
        return history

    def train_directly(self, x, y, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
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
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        utils.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        utils.log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)
        print("Model Compiled Successfully")

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

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
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
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
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
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

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
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            utils.log("Processing {} images".format(len(images)))
            for image in images:
                utils.log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        # np.save("image_metas.npy", image_metas)
        # np.save("anchors.npy", anchors)
        # np.save("molded_images.npy", molded_images)

        if verbose:
            utils.log("molded_images", molded_images)
            utils.log("image_metas", image_metas)
            utils.log("anchors", anchors)

        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
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
            utils.log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                utils.log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            utils.log("molded_images", molded_images)
            utils.log("image_metas", image_metas)
            utils.log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       window)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = utils.compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
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

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        image_metas: If provided, the images are assumed to be already
            molded (i.e. resized, padded, and normalized)

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Prepare inputs
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        # Run inference
        if not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            utils.log(k, v)
        return outputs_np

    @staticmethod
    def _build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
        """Given the anchors and GT boxes, compute overlaps and identify positive
        anchors and deltas to refine them to match their corresponding GT boxes.

        anchors: [num_anchors, (y1, x1, y2, x2)]
        gt_class_ids: [num_gt_boxes] Integer class IDs.
        gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

        Returns:
        rpn_match: [N] (int32) matches between anchors and GT boxes.
                   1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        """
        # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
        rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
        # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
        rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances. Exclude
        # them from training. A crowd box is given a negative class ID.
        crowd_ix = np.where(gt_class_ids < 0)[0]
        if crowd_ix.shape[0] > 0:
            # Filter out crowds from ground truth class IDs and boxes
            non_crowd_ix = np.where(gt_class_ids > 0)[0]
            crowd_boxes = gt_boxes[crowd_ix]
            gt_class_ids = gt_class_ids[non_crowd_ix]
            gt_boxes = gt_boxes[non_crowd_ix]
            # Compute overlaps with crowd boxes [anchors, crowds]
            crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
            crowd_iou_max = np.amax(crowd_overlaps, axis=1)
            no_crowd_bool = (crowd_iou_max < 0.001)
        else:
            # All anchors don't intersect a crowd
            no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

        # Compute overlaps [num_anchors, num_gt_boxes]
        overlaps = utils.compute_overlaps(anchors, gt_boxes)

        # Match anchors to GT Boxes
        # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
        # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
        # Neutral anchors are those that don't match the conditions above,
        # and they don't influence the loss function.
        # However, don't keep any GT box unmatched (rare, but happens). Instead,
        # match it to the closest anchor (even if its max IoU is < 0.3).
        #
        # 1. Set negative anchors first. They get overwritten below if a GT box is
        # matched to them. Skip boxes in crowd areas.
        anchor_iou_argmax = np.argmax(overlaps, axis=1)
        anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
        rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
        # 2. Set an anchor for each GT box (regardless of IoU value).
        # If multiple anchors have the same IoU match all of them
        gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
        rpn_match[gt_iou_argmax] = 1
        # 3. Set anchors with high overlap as positive.
        rpn_match[anchor_iou_max >= 0.7] = 1

        # Subsample to balance positive and negative anchors
        # Don't let positives be more than half the anchors
        ids = np.where(rpn_match == 1)[0]
        extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
        if extra > 0:
            # Reset the extra ones to neutral
            ids = np.random.choice(ids, extra, replace=False)
            rpn_match[ids] = 0
        # Same for negative proposals
        ids = np.where(rpn_match == -1)[0]
        extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                            np.sum(rpn_match == 1))
        if extra > 0:
            # Rest the extra ones to neutral
            ids = np.random.choice(ids, extra, replace=False)
            rpn_match[ids] = 0

        # For positive anchors, compute shift and scale needed to transform them
        # to match the corresponding GT boxes.
        ids = np.where(rpn_match == 1)[0]
        ix = 0  # index into rpn_bbox
        # TODO: use box_refinement() rather than duplicating the code here
        for i, a in zip(ids, anchors[ids]):
            # Closest gt box (it might have IoU < 0.7)
            gt = gt_boxes[anchor_iou_argmax[i]]

            # Convert coordinates to center plus width/height.
            # GT Box
            gt_h = gt[2] - gt[0]
            gt_w = gt[3] - gt[1]
            gt_center_y = gt[0] + 0.5 * gt_h
            gt_center_x = gt[1] + 0.5 * gt_w
            # Anchor
            a_h = a[2] - a[0]
            a_w = a[3] - a[1]
            a_center_y = a[0] + 0.5 * a_h
            a_center_x = a[1] + 0.5 * a_w

            # Compute the bbox refinement that the RPN should predict.
            rpn_bbox[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w),
            ]
            # Normalize
            rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
            ix += 1

        return rpn_match, rpn_bbox

    @staticmethod
    def _build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
        """Generate targets for training Stage 2 classifier and mask heads.
        This is not used in normal training. It's useful for debugging or to train
        the Mask RCNN heads without using the RPN head.

        Inputs:
        rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
        gt_class_ids: [instance count] Integer class IDs
        gt_boxes: [instance count, (y1, x1, y2, x2)]
        gt_masks: [height, width, instance count] Ground truth masks. Can be full
                  size or mini-masks.

        Returns:
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
        bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
                bbox refinements.
        masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
               to bbox boundaries and resized to neural network output size.
        """
        assert rpn_rois.shape[0] > 0
        assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
            gt_class_ids.dtype)
        assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(
            gt_boxes.dtype)
        assert gt_masks.dtype == bool, "Expected bool but got {}".format(
            gt_masks.dtype)

        # It's common to add GT Boxes to ROIs but we don't do that here because
        # according to XinLei Chen's paper, it doesn't help.

        # Trim empty padding in gt_boxes and gt_masks parts
        instance_ids = np.where(gt_class_ids > 0)[0]
        assert instance_ids.shape[0] > 0, "Image must contain instances."
        gt_class_ids = gt_class_ids[instance_ids]
        gt_boxes = gt_boxes[instance_ids]
        gt_masks = gt_masks[:, :, instance_ids]

        # Compute areas of ROIs and ground truth boxes.
        rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
                       (rpn_rois[:, 3] - rpn_rois[:, 1])
        gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
                      (gt_boxes[:, 3] - gt_boxes[:, 1])

        # Compute overlaps [rpn_rois, gt_boxes]
        overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
        for i in range(overlaps.shape[1]):
            gt = gt_boxes[i]
            overlaps[:, i] = utils.compute_iou(
                gt, rpn_rois, gt_box_area[i], rpn_roi_area)

        # Assign ROIs to GT boxes
        rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
        rpn_roi_iou_max = overlaps[np.arange(
            overlaps.shape[0]), rpn_roi_iou_argmax]
        # GT box assigned to each ROI
        rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
        rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

        # Positive ROIs are those with >= 0.5 IoU with a GT box.
        fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

        # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
        # TODO: To hard example mine or not to hard example mine, that's the question
        # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
        bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

        # Subsample ROIs. Aim for 33% foreground.
        # FG
        fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
        if fg_ids.shape[0] > fg_roi_count:
            keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
        else:
            keep_fg_ids = fg_ids
        # BG
        remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
        if bg_ids.shape[0] > remaining:
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
        else:
            keep_bg_ids = bg_ids
        # Combine indices of ROIs to keep
        keep = np.concatenate([keep_fg_ids, keep_bg_ids])
        # Need more?
        remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
        if remaining > 0:
            # Looks like we don't have enough samples to maintain the desired
            # balance. Reduce requirements and fill in the rest. This is
            # likely different from the Mask RCNN paper.

            # There is a small chance we have neither fg nor bg samples.
            if keep.shape[0] == 0:
                # Pick bg regions with easier IoU threshold
                bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
                assert bg_ids.shape[0] >= remaining
                keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
                assert keep_bg_ids.shape[0] == remaining
                keep = np.concatenate([keep, keep_bg_ids])
            else:
                # Fill the rest with repeated bg rois.
                if len(keep_bg_ids) == 0:
                    keep_bg_ids = np.array([0], dtype=np.int32)
                keep_extra_ids = np.random.choice(
                    keep_bg_ids, remaining, replace=True)
                keep = np.concatenate([keep, keep_extra_ids])
        assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
            "keep doesn't match ROI batch size {}, {}".format(
                keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

        # Reset the gt boxes assigned to BG ROIs.
        rpn_roi_gt_boxes[keep_bg_ids, :] = 0
        rpn_roi_gt_class_ids[keep_bg_ids] = 0

        # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
        rois = rpn_rois[keep]
        roi_gt_boxes = rpn_roi_gt_boxes[keep]
        roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
        roi_gt_assignment = rpn_roi_iou_argmax[keep]

        # Class-aware bbox deltas. [y, x, log(h), log(w)]
        bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                           config.NUM_CLASSES, 4), dtype=np.float32)
        pos_ids = np.where(roi_gt_class_ids > 0)[0]
        bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(
            rois[pos_ids], roi_gt_boxes[pos_ids, :4])
        # Normalize bbox refinements
        bboxes /= config.BBOX_STD_DEV

        # Generate class-specific target masks
        masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                         dtype=np.float32)
        for i in pos_ids:
            class_id = roi_gt_class_ids[i]
            assert class_id > 0, "class id must be greater than 0"
            gt_id = roi_gt_assignment[i]
            class_mask = gt_masks[:, :, gt_id]

            if config.USE_MINI_MASK:
                # Create a mask placeholder, the size of the image
                placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
                # GT box
                gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
                gt_w = gt_x2 - gt_x1
                gt_h = gt_y2 - gt_y1
                # Resize mini mask to size of GT box
                placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
                    np.round(utils.resize(class_mask, (gt_h, gt_w))).astype(bool)
                # Place the mini batch in the placeholder
                class_mask = placeholder

            # Pick part of the mask and resize it
            y1, x1, y2, x2 = rois[i].astype(np.int32)
            m = class_mask[y1:y2, x1:x2]
            mask = utils.resize(m, config.MASK_SHAPE)
            masks[i, :, :, class_id] = mask

        return rois, roi_gt_class_ids, bboxes, masks

    @staticmethod
    def _generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
        """Generates ROI proposals similar to what a region proposal network
        would generate.

        image_shape: [Height, Width, Depth]
        count: Number of ROIs to generate
        gt_class_ids: [N] Integer ground truth class IDs
        gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.

        Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
        """
        # placeholder
        rois = np.zeros((count, 4), dtype=np.int32)

        # Generate random ROIs around GT boxes (90% of count)
        rois_per_box = int(0.9 * count / gt_boxes.shape[0])
        for i in range(gt_boxes.shape[0]):
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
            h = gt_y2 - gt_y1
            w = gt_x2 - gt_x1
            # random boundaries
            r_y1 = max(gt_y1 - h, 0)
            r_y2 = min(gt_y2 + h, image_shape[0])
            r_x1 = max(gt_x1 - w, 0)
            r_x2 = min(gt_x2 + w, image_shape[1])

            # To avoid generating boxes with zero area, we generate double what
            # we need and filter out the extra. If we get fewer valid boxes
            # than we need, we loop and try again.
            while True:
                y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
                x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
                # Filter out zero area boxes
                threshold = 1
                y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                            threshold][:rois_per_box]
                x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                            threshold][:rois_per_box]
                if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                    break

            # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
            # into x1, y1, x2, y2 order
            x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
            y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
            box_rois = np.hstack([y1, x1, y2, x2])
            rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

        # Generate random ROIs anywhere in the image (10% of count)
        remaining_count = count - (rois_per_box * gt_boxes.shape[0])
        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
            x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                        threshold][:remaining_count]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                        threshold][:remaining_count]
            if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        global_rois = np.hstack([y1, x1, y2, x2])
        rois[-remaining_count:] = global_rois
        return rois

    def _data_generator(self, dataset, config, shuffle=True, augment=False, augmentation=None,
                       random_rois=0, batch_size=1, detection_targets=False,
                       no_augmentation_sources=None):
        """A generator that returns images and corresponding target class ids,
        bounding box deltas, and masks.

        dataset: The Dataset object to pick data from
        config: The model config object
        shuffle: If True, shuffles the samples before every epoch
        augment: (deprecated. Use augmentation instead). If true, apply random
            image augmentation. Currently, only horizontal flipping is offered.
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
            For example, passing imgaug.augmenters.Fliplr(0.5) flips images
            right/left 50% of the time.
        random_rois: If > 0 then generate proposals to be used to train the
                     network classifier and mask heads. Useful if training
                     the Mask RCNN part without the RPN.
        batch_size: How many images to return in each call
        detection_targets: If True, generate detection targets (class IDs, bbox
            deltas, and masks). Typically for debugging or visualizations because
            in trainig detection targets are generated by DetectionTargetLayer.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.

        Returns a Python generator. Upon calling next() on it, the
        generator returns two lists, inputs and outputs. The contents
        of the lists differs depending on the received arguments:
        inputs list:
        - images: [batch, H, W, C]
        - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
        - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
        - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
        - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
        - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                    are those of the image unless use_mini_mask is True, in which
                    case they are defined in MINI_MASK_SHAPE.

        outputs list: Usually empty in regular training. But if detection_targets
            is True then the outputs list contains target class_ids, bbox deltas,
            and masks.
        """
        b = 0  # batch item index
        image_index = -1
        image_ids = np.copy(dataset.image_ids)
        error_count = 0
        no_augmentation_sources = no_augmentation_sources or []

        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        backbone_shapes = utils.compute_backbone_shapes(config, config.IMAGE_SHAPE)
        anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                 config.RPN_ANCHOR_RATIOS,
                                                 backbone_shapes,
                                                 config.BACKBONE_STRIDES,
                                                 config.RPN_ANCHOR_STRIDE)

        # Keras requires a generator to run indefinitely.
        while True:
            try:
                # Increment index to pick next image. Shuffle if at the start of an epoch.
                image_index = (image_index + 1) % len(image_ids)
                if shuffle and image_index == 0:
                    np.random.shuffle(image_ids)

                # Get GT bounding boxes and masks for image.
                image_id = image_ids[image_index]

                # If the image source is not to be augmented pass None as augmentation
                if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                    image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                        load_image_gt(dataset, config, image_id, augment=augment,
                                      augmentation=None,
                                      use_mini_mask=config.USE_MINI_MASK)
                else:
                    image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                        load_image_gt(dataset, config, image_id, augment=augment,
                                      augmentation=augmentation,
                                      use_mini_mask=config.USE_MINI_MASK)

                # Skip images that have no instances. This can happen in cases
                # where we train on a subset of classes and the image doesn't
                # have any of the classes we care about.
                if not np.any(gt_class_ids > 0):
                    continue

                # RPN Targets
                rpn_match, rpn_bbox = self._build_rpn_targets(image.shape, anchors,
                                                        gt_class_ids, gt_boxes, config)

                # Mask R-CNN Targets
                if random_rois:
                    rpn_rois = self._generate_random_rois(
                        image.shape, random_rois, gt_class_ids, gt_boxes)
                    if detection_targets:
                        rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = \
                            self._build_detection_targets(
                                rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)

                # Init batch arrays
                if b == 0:
                    batch_image_meta = np.zeros(
                        (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                    batch_rpn_match = np.zeros(
                        [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                    batch_rpn_bbox = np.zeros(
                        [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                    batch_images = np.zeros(
                        (batch_size,) + image.shape, dtype=np.float32)
                    batch_gt_class_ids = np.zeros(
                        (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                    batch_gt_boxes = np.zeros(
                        (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                    batch_gt_masks = np.zeros(
                        (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                         config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                    if random_rois:
                        batch_rpn_rois = np.zeros(
                            (batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                        if detection_targets:
                            batch_rois = np.zeros(
                                (batch_size,) + rois.shape, dtype=rois.dtype)
                            batch_mrcnn_class_ids = np.zeros(
                                (batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                            batch_mrcnn_bbox = np.zeros(
                                (batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                            batch_mrcnn_mask = np.zeros(
                                (batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

                # If more instances than fits in the array, sub-sample from them.
                if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                    ids = np.random.choice(
                        np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                    gt_class_ids = gt_class_ids[ids]
                    gt_boxes = gt_boxes[ids]
                    gt_masks = gt_masks[:, :, ids]

                # Add to batch
                batch_image_meta[b] = image_meta
                batch_rpn_match[b] = rpn_match[:, np.newaxis]
                batch_rpn_bbox[b] = rpn_bbox
                batch_images[b] = mold_image(image.astype(np.float32), config)
                batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
                batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
                batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
                if random_rois:
                    batch_rpn_rois[b] = rpn_rois
                    if detection_targets:
                        batch_rois[b] = rois
                        if len(batch_mrcnn_class_ids.shape) == 3:
                            batch_mrcnn_class_ids = batch_mrcnn_class_ids.reshape(8, 32)
                        batch_mrcnn_class_ids[b] = mrcnn_class_ids
                        batch_mrcnn_bbox[b] = mrcnn_bbox
                        batch_mrcnn_mask[b] = mrcnn_mask
                b += 1

                # Batch full?
                if b >= batch_size - 1:
                    inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                              batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                    padding = np.ones(shape=batch_images.shape)
                    outputs = [padding, padding, padding, padding, batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                              batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]


                    # TODO: adaptar classe para o seguinte formato https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit#wrapping_up_an_end-to-end_gan_example
                    if random_rois:
                        #inputs.extend([batch_rpn_rois])
                        if detection_targets:
                            #inputs.extend([batch_rois])
                            # Keras requires that output and targets have the same number of dimensions
                            batch_mrcnn_class_ids = np.expand_dims(
                                batch_mrcnn_class_ids, -1)
                            outputs.extend([batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

                    yield inputs, outputs

                    # start a new batch
                    b = 0
            except (GeneratorExit, KeyboardInterrupt):
                raise
            except:
                # Log it and skip the image
                logging.exception("Error processing image {}".format(
                    dataset.image_info[image_id]))
                error_count += 1
                if error_count > 5:
                    raise

