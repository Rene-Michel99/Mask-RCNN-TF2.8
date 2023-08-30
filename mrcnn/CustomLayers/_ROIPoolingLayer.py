import tensorflow as tf
from mrcnn.Utils.utilfunctions import batch_slice
from mrcnn.Utils.Interface import Interface
from ._Common import trim_zeros_graph, overlaps_graph, resize_and_crop


class ROIPoolingInterface(Interface):
    def __init__(self, config):
        self.IMAGES_PER_GPU = config.IMAGES_PER_GPU
        self.TRAIN_ROIS_PER_IMAGE = config.TRAIN_ROIS_PER_IMAGE
        self.MASK_SHAPE = config.MASK_SHAPE
        self.ROI_POSITIVE_RATIO = config.ROI_POSITIVE_RATIO
        self.BBOX_STD_DEV = config.BBOX_STD_DEV
        self.USE_MINI_MASK = config.USE_MINI_MASK
        self.INTERPOLATION_METHOD = config.INTERPOLATION_METHOD


class ROIPoolingLayer(tf.keras.layers.Layer):
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
        super(ROIPoolingLayer, self).__init__(**kwargs)

        self.interface = ROIPoolingInterface(config)

        self.batch_slice = batch_slice

    @staticmethod
    @tf.function
    def box_refinement_graph(box, gt_box):
        """Compute refinement needed to transform box to gt_box.

        box and gt_box are [N, (y1, x1, y2, x2)]
        """
        box = tf.cast(box, tf.float32)
        gt_box = tf.cast(gt_box, tf.float32)

        height = box[:, 2] - box[:, 0]
        width = box[:, 3] - box[:, 1]
        center_y = box[:, 0] + 0.5 * height
        center_x = box[:, 1] + 0.5 * width

        gt_height = gt_box[:, 2] - gt_box[:, 0]
        gt_width = gt_box[:, 3] - gt_box[:, 1]
        gt_center_y = gt_box[:, 0] + 0.5 * gt_height
        gt_center_x = gt_box[:, 1] + 0.5 * gt_width

        dy = (gt_center_y - center_y) / height
        dx = (gt_center_x - center_x) / width
        dh = tf.math.log(gt_height / height)
        dw = tf.math.log(gt_width / width)

        result = tf.stack([dy, dx, dh, dw], axis=1, name="stacked_box_refinement_graph")
        return result

    @tf.function
    def detection_targets_graph(self, proposals, gt_class_ids, gt_boxes, gt_masks):
        """Generates detection targets for one image. Subsamples proposals and
        generates target class IDs, bounding box deltas, and masks for each.

        Inputs:
        proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals.
        gt_class_ids: [MAX_GT_INSTANCES] int class IDs
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
        gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

        Returns: Target ROIs and corresponding class IDs, bounding box shifts,
        and masks.
        rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
        class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
        deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
        masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
               boundaries and resized to neural network output size.

        Note: Returned arrays might be zero padded if not enough target ROIs.
        """
        # Assertions
        asserts = [
            tf.Assert(tf.greater(tf.shape(input=proposals)[0], 0), [proposals],
                      name="roi_assertion"),
        ]
        with tf.control_dependencies(asserts):
            proposals = tf.identity(proposals)

        # Remove zero padding
        proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
        gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
        gt_class_ids = tf.boolean_mask(tensor=gt_class_ids, mask=non_zeros,
                                       name="trim_gt_class_ids")
        gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                             name="trim_gt_masks")

        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances. Exclude
        # them from training. A crowd box is given a negative class ID.
        crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
        non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
        crowd_boxes = tf.gather(gt_boxes, crowd_ix)
        gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
        gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
        gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

        # Compute overlaps matrix [proposals, gt_boxes]
        overlaps = overlaps_graph(proposals, gt_boxes)

        # Compute overlaps with crowd boxes [proposals, crowd_boxes]
        crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
        crowd_iou_max = tf.reduce_max(input_tensor=crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)

        # Determine positive and negative ROIs
        roi_iou_max = tf.reduce_max(input_tensor=overlaps, axis=1)
        # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
        positive_roi_bool = (roi_iou_max >= 0.5)
        positive_indices = tf.where(positive_roi_bool)[:, 0]
        # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
        negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = int(self.interface.TRAIN_ROIS_PER_IMAGE *
                             self.interface.ROI_POSITIVE_RATIO)
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(input=positive_indices)[0]
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / self.interface.ROI_POSITIVE_RATIO
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
        # Gather selected ROIs
        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)

        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.cond(
            pred=tf.greater(tf.shape(input=positive_overlaps)[1], 0),
            true_fn=lambda: tf.argmax(input=positive_overlaps, axis=1),
            false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
        )
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

        # Compute bbox refinement for positive ROIs
        deltas = self.box_refinement_graph(positive_rois, roi_gt_boxes)
        deltas /= self.interface.BBOX_STD_DEV

        # Assign positive ROIs to GT masks
        # Permute masks to [N, height, width, 1]
        transposed_masks = tf.expand_dims(tf.transpose(a=gt_masks, perm=[2, 0, 1]), -1)
        # Pick the right mask for each ROI
        roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

        # Compute mask targets
        boxes = positive_rois
        if self.interface.USE_MINI_MASK:
            # Transform ROI coordinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = tf.concat([y1, x1, y2, x2], 1)
        box_ids = tf.range(0, tf.shape(input=roi_masks)[0])
        if self.interface.INTERPOLATION_METHOD == "bicubic":
            masks = resize_and_crop(
                img=tf.cast(roi_masks, tf.float32),
                boxes=boxes,
                box_indices=box_ids,
                pool_shape=self.interface.MASK_SHAPE,
                method=self.interface.INTERPOLATION_METHOD
            )
        else:
            masks = tf.image.crop_and_resize(
                tf.cast(roi_masks, tf.float32), boxes,
                box_ids, self.interface.MASK_SHAPE,
                method='bilinear'
            )

        # Remove the extra dimension from masks.
        masks = tf.squeeze(masks, axis=3)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = tf.round(masks)

        # Append negative ROIs and pad bbox deltas and masks that
        # are not used for negative ROIs with zeros.
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        N = tf.shape(input=negative_rois)[0]
        P = tf.maximum(self.interface.TRAIN_ROIS_PER_IMAGE - tf.shape(input=rois)[0], 0)
        rois = tf.pad(tensor=rois, paddings=[(0, P), (0, 0)])
        roi_gt_boxes = tf.pad(tensor=roi_gt_boxes, paddings=[(0, N + P), (0, 0)])
        roi_gt_class_ids = tf.pad(tensor=roi_gt_class_ids, paddings=[(0, N + P)])
        deltas = tf.pad(tensor=deltas, paddings=[(0, N + P), (0, 0)])
        masks = tf.pad(tensor=masks, paddings=[[0, N + P], (0, 0), (0, 0)])

        return rois, roi_gt_class_ids, deltas, masks

    @tf.function
    def call(self, inputs):
        """ The inputs is a list of params.

        Params:
        - proposals: Can be ProposalLayer or KL.Lambda with norm_boxes_graph call
        - gt_class_ids: KL.Input. Detection GT (class IDs, bounding boxes, and masks)
        - gt_boxes: NormBoxesGraph layer of input_gt_boxes
        - gt_masks: KL.Input. With shape of [MINI_MASK_SHAPE[0], MINI_MASK_SHAPE[1], None]
        or [IMAGE_SHAPE[0], IMAGE_SHAPE[1], None], both with dtype bool

        """
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = self.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: self.detection_targets_graph(
                w, x, y, z
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
            "interface": self.interface.to_dict()
        })
        return config
