import tensorflow as tf
from tensorflow.keras import backend as K
from resources.utils import smooth_l1_loss
from resources.Data_utils import batch_pack_graph


class RPNBboxLoss(tf.keras.layers.Layer):
    def __init__(self, images_per_gpu, *args, **kwargs):
        super(RPNBboxLoss, self).__init__(**kwargs)
        self.images_per_gpu = images_per_gpu

    def call(self, inputs):
        """Return the RPN bounding box loss graph.

            config: the model config object.
            target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
                Uses 0 padding to fill in unsed bbox deltas.
            rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
                       -1=negative, 0=neutral anchor.
            rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
            """
        target_bbox = inputs[0]
        rpn_match = inputs[1]
        rpn_bbox = inputs[2]
        # Positive anchors contribute to the loss, but negative and
        # neutral anchors (match value of 0 or -1) don't.
        rpn_match = K.squeeze(rpn_match, -1)
        indices = tf.where(tf.equal(rpn_match, 1))

        # Pick bbox deltas that contribute to the loss
        rpn_bbox = tf.gather_nd(rpn_bbox, indices)

        # Trim target bounding box deltas to the same length as rpn_bbox.
        batch_counts = K.sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)
        target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                       self.images_per_gpu)

        loss = smooth_l1_loss(target_bbox, rpn_bbox)
        self.add_metric(loss, name="rpn_bbox_loss")

        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        self.add_loss(tf.reduce_mean(loss, keepdims=True) * 1.)
        return loss
