import tensorflow as tf
import tensorflow.keras.backend as K


class RPNBBOXLoss(tf.keras.layers.Layer):
    def __init__(self, images_per_gpu, target_bbox, rpn_match, rpn_bbox, **kwargs):
        super(RPNBBOXLoss, self).__init__(**kwargs)
        self.IMAGES_PER_GPU = images_per_gpu
        self.target_bbox = target_bbox
        self.rpn_match = rpn_match
        self.rpn_bbox = rpn_bbox

    def call(self):
        """Return the RPN bounding box loss graph.

            config: the model config object.
            target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
                Uses 0 padding to fill in unsed bbox deltas.
            rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
                       -1=negative, 0=neutral anchor.
            rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
            """
        # Positive anchors contribute to the loss, but negative and
        # neutral anchors (match value of 0 or -1) don't.
        rpn_match = K.squeeze(self.rpn_match, -1)
        indices = tf.where(K.equal(rpn_match, 1))

        # Pick bbox deltas that contribute to the loss
        rpn_bbox = tf.gather_nd(self.rpn_bbox, indices)

        # Trim target bounding box deltas to the same length as rpn_bbox.
        batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
        target_bbox = self.batch_pack_graph(self.target_bbox, batch_counts,
                                       self.IMAGES_PER_GPU)

        loss = self.smooth_l1_loss(target_bbox, rpn_bbox)

        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        return loss

    @staticmethod
    def batch_pack_graph(x, counts, num_rows):
        """Picks different number of values from each row
        in x depending on the values in counts.
        """
        outputs = []
        for i in range(num_rows):
            outputs.append(x[i, :counts[i]])
        return tf.concat(outputs, axis=0)

    @staticmethod
    def smooth_l1_loss(y_true, y_pred):
        """Implements Smooth-L1 loss.
        y_tru
        e and y_pred are typically: [N, 4], but could be any shape.
        """
        diff = K.abs(y_true - y_pred)
        less_than_one = K.cast(K.less(diff, 1.0), "float32")
        loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
        return loss
