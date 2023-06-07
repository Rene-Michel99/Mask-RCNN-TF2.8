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
        pred_class_logits = inputs[1]
        active_class_ids = inputs[2]
        # During model building, Keras calls this function with
        # target_class_ids of type float32. Unclear why. Cast it
        # to int to get around it.
        target_class_ids = tf.cast(inputs[0], tf.int64)
        pred_class_ids = tf.argmax(pred_class_logits, axis=2)
        # Find predictions of classes that are not in the dataset.
        # TODO: Make shape to fit batch size
        #active_class_ids = tf.reshape(active_class_ids, (pred_class_ids.shape[0], -1))
        pred_active = tf.gather(
            active_class_ids[0],
            pred_class_ids
        )

        # Loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_class_ids, logits=pred_class_logits
        )

        # Erase losses of predictions of classes that are not in the active
        # classes of the image.
        # Computer loss mean. Use only predictions that contribute
        # to the loss to get a correct mean.
        loss = loss * pred_active
        loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
        self.add_metric(tf.reduce_mean(loss) * 1., name="mrcnn_class_loss")
        return loss
