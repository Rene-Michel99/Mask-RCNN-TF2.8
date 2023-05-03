import tensorflow as tf
from tensorflow.keras import backend as K


class MRCNNMaskLossGraph(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(MRCNNMaskLossGraph, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        """Mask binary cross-entropy loss for the masks head.

            target_masks: [batch, num_rois, height, width].
                A float32 tensor of values 0 or 1. Uses zero padding to fill array.
            target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
            pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                        with values from 0 to 1.
            """
        target_masks = inputs[0]
        # Reshape for simplicity. Merge first two dimensions into one.
        target_class_ids = tf.reshape(inputs[1], (-1,))
        pred_masks = inputs[2]

        mask_shape = tf.shape(target_masks)
        target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
        pred_shape = tf.shape(pred_masks)
        pred_masks = tf.reshape(pred_masks,
                                (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
        # Permute predicted masks to [N, num_classes, height, width]
        pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = tf.where(target_class_ids > 0)[:, 0]
        positive_class_ids = tf.cast(
            tf.gather(target_class_ids, positive_ix), tf.int64)
        indices = tf.stack([positive_ix, positive_class_ids], axis=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = tf.gather(target_masks, positive_ix)
        y_pred = tf.gather_nd(pred_masks, indices)

        # Compute binary cross entropy. If no positive ROIs, then return 0.
        # shape: [batch, roi, num_classes]

        loss = K.switch(
            tf.size(y_true) > 0,
            K.binary_crossentropy(target=y_true, output=y_pred),
            tf.constant(0.0)
        )
        loss = K.mean(loss)
        self.add_metric(tf.reduce_mean(loss) * 1., name="mrcnn_mask_loss")
        return loss
