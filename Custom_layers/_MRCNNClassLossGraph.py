import tensorflow as tf


class MRCNNClassLossGraph(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(MRCNNClassLossGraph, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        """Loss for the classifier head of Mask RCNN.

            target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
                padding to fill in the array.
            pred_class_logits: [batch, num_rois, num_classes]
            active_class_ids: [batch, num_classes]. Has a value of 1 for
                classes that are in the dataset of the image, and 0
                for classes that are not in the dataset.
            """
        target_class_ids = inputs[0]
        pred_class_logits = inputs[1]
        active_class_ids = inputs[2]
        # During model building, Keras calls this function with
        # target_class_ids of type float32. Unclear why. Cast it
        # to int to get around it.
        target_class_ids = tf.cast(target_class_ids, tf.int64)

        # Find predictions of classes that are not in the dataset.
        pred_class_ids = tf.argmax(pred_class_logits, axis=2)
        # TODO: Update this line to work with batch > 1. Right now it assumes all
        #       images in a batch have the same active_class_ids
        pred_active = tf.gather(active_class_ids[0], pred_class_ids)

        # Loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_class_ids, logits=pred_class_logits)

        # Erase losses of predictions of classes that are not in the active
        # classes of the image.
        loss = loss * pred_active
        self.add_metric(loss, name="mrcnn_class_loss")

        # Computer loss mean. Use only predictions that contribute
        # to the loss to get a correct mean.
        return tf.reduce_sum(loss) / tf.reduce_sum(pred_active)

