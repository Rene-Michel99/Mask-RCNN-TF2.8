import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.backend as K
from typing import List

from mrcnn.Configs import Config
from mrcnn.CustomLayers import norm_boxes_graph, GetAnchors, ROILayer


class RPNComponent:
    def __init__(
            self,
            mode: str,
            input_image: KL.Input,
            rpn_feature_maps: List[KL.Conv2D],
            config: Config,
            anchors: GetAnchors,
            **kwargs
    ):
        self.target_rois = None
        self.rpn_rois = None
        self.rpn_class_logits = None
        self.rpn_class = None
        self.rpn_bbox = None
        self.input_rois = None
        self.anchors = anchors
        self.build(mode, rpn_feature_maps, input_image, config, anchors)

    def build(
            self,
            mode: str,
            rpn_feature_maps: List[KL.Conv2D],
            input_image: KL.Input,
            config: Config,
            anchors: GetAnchors,
    ):
        # RPN Model
        self.rpn_class_logits, self.rpn_class, self.rpn_bbox = self._get_rpn_layers(
            config, rpn_feature_maps
        )

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        self.rpn_rois = ROILayer(  # noqa
            mode=mode,
            config=config,
            name="ROI"
        )([self.rpn_class, self.rpn_bbox, anchors])

        if mode == "training":
            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                self.input_rois = KL.Input(
                    shape=[config.POST_NMS_ROIS_TRAINING, 4],
                    name="input_roi", dtype=np.int32
                )
                # Normalize coordinates
                self.target_rois = KL.Lambda(
                    lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3])
                )(self.input_rois)
            else:
                self.target_rois = self.rpn_rois
        else:
            pass

    def _get_rpn_layers(
            self,
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
        rpn = self._build_rpn_model(
            config.RPN_ANCHOR_STRIDE,
            len(config.RPN_ANCHOR_RATIOS),
            config.TOP_DOWN_PYRAMID_SIZE
        )
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

    def _build_rpn_model(self, anchor_stride: int, anchors_per_location: int, depth: int):
        """Builds a Keras model of the Region Proposal Network.
        It wraps the RPN graph so it can be used multiple times with shared
        weights.

        Params:
        - anchors_per_location: number of anchors per pixel in the feature map
        - anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                       every pixel in the feature map), or 2 (every other pixel).
        - depth: Depth of the backbone feature map.

        Returns: a Keras Model object. The model outputs, when called, are:
        - rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        - rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        - rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                    applied to anchors.
        """
        input_feature_map = KL.Input(shape=[None, None, depth],
                                     name="input_rpn_feature_map")
        outputs = self._rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
        return KM.Model([input_feature_map], outputs, name="rpn_model")

    @staticmethod
    def _rpn_graph(
            feature_map: KL.Input,
            anchors_per_location: int,
            anchor_stride: int
    ):
        """Builds the computation graph of Region Proposal Network.

        Params:
        - feature_map: backbone features [batch, height, width, depth]
        - anchors_per_location: number of anchors per pixel in the feature map
        - anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                       every pixel in the feature map), or 2 (every other pixel).

        Returns: Tuple of
            - rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
            - rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
            - rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                      applied to anchors.
        """
        # TODO: check if stride of 2 causes alignment issues if the feature map is not even.
        # Shared convolutional base of the RPN
        shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                           strides=anchor_stride,
                           name='rpn_conv_shared')(feature_map)

        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                      activation='linear', name='rpn_class_raw')(shared)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = KL.Lambda(
            lambda t: tf.reshape(t, [tf.shape(input=t)[0], -1, 2])
        )(x)

        # Softmax on last dimension of BG/FG.
        rpn_probs = KL.Activation(
            "softmax", name="rpn_class_xxx")(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                      activation='linear', name='rpn_bbox_pred')(shared)

        # Reshape to [batch, anchors, 4]
        rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(input=t)[0], -1, 4]))(x)

        return [rpn_class_logits, rpn_probs, rpn_bbox]

