import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context

from mrcnn.Utils.utilfunctions import batch_slice
from mrcnn.Utils.Interface import Interface
from ._Common import apply_box_deltas_graph, clip_boxes_graph


class ROILayerInterface(Interface):
    def __init__(self, config):
        self.NMS_THRESHOLD = config.RPN_NMS_THRESHOLD
        self.IMAGES_PER_GPU = config.IMAGES_PER_GPU
        self.PRE_NMS_LIMIT = config.PRE_NMS_LIMIT
        self.RPN_BBOX_STD_DEV = config.RPN_BBOX_STD_DEV


class ROILayer(tf.keras.layers.Layer):
    """
    The regions obtained from the RPN might be of different shapes, right? Hence,
    we apply a pooling layer and convert all the regions to the same shape.
    Next, these regions are passed through a fully connected network so that the class
    label and bounding boxes are predicted.
    Till this point, the steps are almost similar to how Faster R-CNN works. Now comes
    the difference between the two frameworks. In addition to this, Mask R-CNN also generates the
    segmentation mask.

    For that, we first compute the region of interest so that the computation time can be reduced.
    For all the predicted regions, we compute the Intersection over Union (IoU) with the ground
    truth boxes. We can computer IoU like this IoU = Area if intersection / Area of the union.

    Now, only if the IoU is greater than or equal to 0.5, we consider that as a region of interest.
    Otherwise, we neglect that particular region. We do this for all the regions and then select only
    a set of regions for which the IoU is greater than 0.5.

    Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.


    Params:
        - rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        - rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        - anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns: Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(
            self,
            mode,
            config,
            **kwargs
    ):
        super(ROILayer, self).__init__(**kwargs)

        self.proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else config.POST_NMS_ROIS_INFERENCE
        self.interface = ROILayerInterface(config)

        self.batch_slice = batch_slice
        self.apply_box_deltas_graph = apply_box_deltas_graph
        self.clip_boxes_graph = clip_boxes_graph

    @tf.function
    def call(self, inputs):
        """Receives anchor scores and selects a subset to pass as proposals
            to the second stage. Filtering is done based on anchor scores and
            non-max suppression to remove overlaps. It also applies bounding
            box refinement deltas to anchors.

            Params:
                - rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
                - rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
                - anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

            Returns: Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
            """
        # TODO: Check if nms overlap is broken
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.interface.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.interface.PRE_NMS_LIMIT, tf.shape(input=anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = self.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.interface.IMAGES_PER_GPU)
        deltas = self.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.interface.IMAGES_PER_GPU)
        pre_nms_anchors = self.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                            self.interface.IMAGES_PER_GPU,
                                            names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = self.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.interface.IMAGES_PER_GPU,
                                  names=["refined_anchors"])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = self.batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.interface.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.interface.NMS_THRESHOLD,
                name="rpn_non_max_suppression"
            )
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(input=proposals)[0], 0)
            proposals = tf.pad(tensor=proposals, paddings=[(0, padding), (0, 0)])
            return proposals

        proposals = self.batch_slice([boxes, scores], nms,
                                      self.interface.IMAGES_PER_GPU)

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(None)
            proposals.set_shape(out_shape)

        return proposals

    def compute_output_shape(self, input_shape):
        return None, self.proposal_count, 4

    def get_config(self):
        config = super(ROILayer, self).get_config()
        config.update({
            "proposal_count": self.proposal_count,
            "interface": self.interface.to_dict(),
        })
        return config
