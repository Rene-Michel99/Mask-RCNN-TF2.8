import tensorflow as tf
from resources.utils import batch_slice
from ._Common import detection_targets_graph


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
            images_per_gpu,
            train_rois_per_image,
            mask_shape,
            roi_positive_ratio,
            bbox_std_dev,
            use_mini_mask,
            **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.images_per_gpu = images_per_gpu
        self.train_rois_per_image = train_rois_per_image
        self.mask_shape = mask_shape
        self.roi_positive_ratio = roi_positive_ratio
        self.bbox_std_dev = bbox_std_dev
        self.use_mini_mask = use_mini_mask

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
                w, x, y, z,
                train_rois_per_image=self.train_rois_per_image,
                roi_positive_ratio=self.roi_positive_ratio,
                bbox_std_dev=self.bbox_std_dev,
                use_mini_mask=self.use_mini_mask,
                mask_shape=self.mask_shape
            ), self.images_per_gpu, names=names
        )
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.train_rois_per_image, 4),  # rois
            (None, self.train_rois_per_image),  # class_ids
            (None, self.train_rois_per_image, 4),  # deltas
            (None, self.train_rois_per_image, self.mask_shape[0],
             self.mask_shape[1])  # masks
        ]

    @staticmethod
    def compute_mask(inputs, mask=None):
        return [None, None, None, None]

    def get_config(self):
        config = super().get_config()
        config.update({
            "config": "config",
        })
        return config
