import tensorflow as tf
from resources.utils import batch_slice
from ._Common import detection_targets_graph


class Interface:
    def __init__(self, config):
        self.IMAGES_PER_GPU = config.IMAGES_PER_GPU
        self.TRAIN_ROIS_PER_IMAGE = config.TRAIN_ROIS_PER_IMAGE
        self.MASK_SHAPE = config.MASK_SHAPE
        self.ROI_POSITIVE_RATIO = config.ROI_POSITIVE_RATIO
        self.BBOX_STD_DEV = config.BBOX_STD_DEV
        self.USE_MINI_MASK = config.USE_MINI_MASK


class DetectionTargetLayer(tf.keras.layers.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(
            self,
            config,
            **kwargs
    ):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.interface = Interface(config)

        self.detection_targets_graph = detection_targets_graph
        self.batch_slice = batch_slice

    @tf.function
    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = self.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: self.detection_targets_graph(
                w, x, y, z, self.interface
            ), self.interface.IMAGES_PER_GPU, names=names
        )
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.interface.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.interface.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.interface.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.interface.TRAIN_ROIS_PER_IMAGE, self.interface.MASK_SHAPE[0],
             self.interface.MASK_SHAPE[1])  # masks
        ]

    @staticmethod
    def compute_mask(inputs, mask=None):
        return [None, None, None, None]

    def get_config(self):
        config = super().get_config()
        config.update({
            "interface": self.interface
        })
        return config
