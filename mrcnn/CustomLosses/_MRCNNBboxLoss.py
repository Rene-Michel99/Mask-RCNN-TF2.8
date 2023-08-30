import tensorflow as tf
from tensorflow.keras import backend as K
from ..CustomLayers import smooth_l1_loss


class MRCNNBboxLoss(tf.keras.losses.Loss):
    def __init__(self, *args, **kwargs):
        super(MRCNNBboxLoss, self).__init__(**kwargs)

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
        positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = tf.cast(
            tf.gather(target_class_ids, positive_roi_ix),
            tf.int64
        )
        indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = tf.gather(target_bbox, positive_roi_ix)
        pred_bbox = tf.gather_nd(pred_bbox, indices)

        # Smooth-L1 Loss
        loss = K.switch(
            tf.size(input=target_bbox) > 0,
            smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox, name="MRCNNBboxLoss"),
            tf.constant(0.0)
        )
        loss = K.mean(loss)
        return loss
