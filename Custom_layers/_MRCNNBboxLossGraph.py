import tensorflow as tf
from tensorflow.keras import backend as K
from ._Common import smooth_l1_loss


class MRCNNBboxLossGraph(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(MRCNNBboxLossGraph, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        """Loss for Mask R-CNN bounding box refinement.

            target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
            target_class_ids: [batch, num_rois]. Integer class IDs.
            pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
            """
        target_bbox = inputs[0]
        target_class_ids = inputs[1]
        pred_bbox = inputs[2]
        # Reshape to merge batch and roi dimensions for simplicity.
        target_class_ids = K.reshape(target_class_ids, (-1,))
        target_bbox = K.reshape(target_bbox, (-1, 4))
        pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indices.
        positive_roi_ix = tf.where(target_class_ids > 0, name="where_roi_ix")[:, 0]
        positive_roi_class_ids = tf.cast(
            tf.gather_nd(
                [target_class_ids],
                tf.stack([tf.zeros_like(positive_roi_ix), positive_roi_ix], axis=-1), name="gather_nd_roi_class_ids"
            ),
            tf.int64
        )
        indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1, name="stacked_indices")

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = tf.gather_nd(
            [target_bbox],
            tf.stack([tf.zeros_like(positive_roi_ix), positive_roi_ix], axis=-1, name="gather_nd_target_bbox")
        )
        pred_bbox = tf.gather_nd(pred_bbox, indices)

        metric = smooth_l1_loss(target_bbox, pred_bbox, name="MRCNNBboxLossGraph")
        self.add_metric(metric, name="mrcnn_bbox_loss")

        # Smooth-L1 Loss
        #loss = metric if tf.size(target_bbox) > 0 else tf.constant(0.0)
        #mean_loss = K.mean(loss)
        return K.mean(metric)
