from typing import List, Union, Any
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as K

from mrcnn.Configs import Config
from mrcnn.CustomLayers import (
    BatchNorm,
    PyramidROIAlign,
    DetectionLayer,
    ROIPoolingLayer,
    ROILayer
)


class MaskSegmentationComponent:
    def __init__(
            self,
            mode: str,
            rois: Union[ROILayer, Any],
            mrcnn_feature_maps: List[KL.Conv2D],
            input_image_meta: KL.Input,
            config: Config
    ):
        self.mrcnn_mask = None
        self.detections = None
        self.mrcnn_class = None
        self.mrcnn_bbox = None
        self.mrcnn_class_logits = None
        self.detection_boxes = None

        self.build(
            mode,
            rois,
            mrcnn_feature_maps,
            input_image_meta,
            config
        )

    def build(
            self,
            mode: str,
            rois: Union[ROILayer, Any],
            mrcnn_feature_maps: List[KL.Conv2D],
            input_image_meta: KL.Input,
            config: Config,
    ):
        # Network Heads
        # Proposal classifier and BBox regressor heads
        # TODO: verify that this handles zero padded ROIs
        self.mrcnn_class_logits, self.mrcnn_class, self.mrcnn_bbox = self._fpn_classifier_graph(
            rois, mrcnn_feature_maps, input_image_meta,
            config.POOL_SIZE, config.NUM_CLASSES,
            config.INTERPOLATION_METHOD,
            train_bn=config.TRAIN_BN,
            fc_layers_size=config.FPN_CLASSIFIER_FC_LAYERS_SIZE
        )

        if mode == "inference":
            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            self.detections = DetectionLayer(  # noqa
                config,
                name="mrcnn_detection"
            )([rois, self.mrcnn_class, self.mrcnn_bbox, input_image_meta])

            # Create masks for detections
            self.detection_boxes = KL.Lambda(lambda x: x[..., :4])(self.detections)
            self.mrcnn_mask = self._build_fpn_mask_generator(
                self.detection_boxes, mrcnn_feature_maps,
                input_image_meta,
                config.MASK_POOL_SIZE,
                config.NUM_CLASSES,
                config.INTERPOLATION_METHOD,
                train_bn=config.TRAIN_BN
            )
        else:
            self.mrcnn_mask = self._build_fpn_mask_generator(
                rois, mrcnn_feature_maps,
                input_image_meta,
                config.MASK_POOL_SIZE,
                config.NUM_CLASSES,
                config.INTERPOLATION_METHOD,
                train_bn=config.TRAIN_BN
            )

    @staticmethod
    def _build_fpn_mask_generator(
            rois: KL.Lambda,
            feature_maps: List[KL.Conv2D],
            image_meta: KL.Input,
            pool_size: int,
            num_classes: int,
            interpolation_method: str,
            train_bn=True
    ):
        """Builds the computation graph of the mask head of Feature Pyramid Network.

        Params:
        - rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates.
        - feature_maps: List of feature maps from different layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
        - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        - pool_size: The width of the square feature map generated from ROI Pooling.
        - num_classes: number of classes, which determines the depth of the results
        - train_bn: Boolean. Train or freeze Batch Norm layers

        Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
        """
        # ROI Pooling
        # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
        x = PyramidROIAlign(  # noqa
            [pool_size, pool_size],
            interpolation_method=interpolation_method,
            name="roi_align_mask"
        )([rois, image_meta] + feature_maps)

        # Conv layers
        x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                               name="mrcnn_mask_conv1")(x)
        x = KL.TimeDistributed(BatchNorm(),
                               name='mrcnn_mask_bn1')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                               name="mrcnn_mask_conv2")(x)
        x = KL.TimeDistributed(BatchNorm(),
                               name='mrcnn_mask_bn2')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                               name="mrcnn_mask_conv3")(x)
        x = KL.TimeDistributed(BatchNorm(),
                               name='mrcnn_mask_bn3')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                               name="mrcnn_mask_conv4")(x)
        x = KL.TimeDistributed(BatchNorm(),
                               name='mrcnn_mask_bn4')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                               name="mrcnn_mask_deconv")(x)
        x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                               name="mrcnn_mask")(x)
        return x

    @staticmethod
    def _fpn_classifier_graph(
            rois: Union[ROIPoolingLayer, DetectionLayer],
            feature_maps: List[KL.Conv2D],
            image_meta: KL.Input,
            pool_size: int,
            num_classes: int,
            interpolation_method: str,
            train_bn=True,
            fc_layers_size=1024
    ):
        """Builds the computation graph of the feature pyramid network classifier
        and regressor heads.

        Params:
        - rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates.
        - feature_maps: List of feature maps from different layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
        - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        - pool_size: The width of the square feature map generated from ROI Pooling.
        - num_classes: number of classes, which determines the depth of the results
        - interpolation_method: String. Interpolation method to be use in resize function.
            Can be bilinear, nearest, bicubic, area and lanczos4. To see more about search
            cv2.resize method
        - train_bn: Boolean. Train or freeze Batch Norm layers
        - fc_layers_size: Size of the 2 FC layers

        Returns: Tuple of
            - logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
            - probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
            - bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                         proposal boxes
        """
        # ROI Pooling
        # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
        x = PyramidROIAlign(  # noqa
            [pool_size, pool_size],
            interpolation_method=interpolation_method,
            name="roi_align_classifier"
        )([rois, image_meta] + feature_maps)
        # Two 1024 FC layers (implemented with Conv2D for consistency)
        x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                               name="mrcnn_class_conv1")(x)
        x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
        x = KL.Activation('relu')(x)
        x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
                               name="mrcnn_class_conv2")(x)
        x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
        x = KL.Activation('relu')(x)

        shared = KL.Lambda(lambda z: K.squeeze(K.squeeze(z, 3), 2), name="pool_squeeze")(x)

        # Classifier head
        mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                                name='mrcnn_class_logits')(shared)
        mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                         name="mrcnn_class")(mrcnn_class_logits)

        # BBox head
        # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
        x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                               name='mrcnn_bbox_fc')(shared)
        # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
        s = K.int_shape(x)
        if s[1] is None:
            mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)
        else:
            mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

        return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox
