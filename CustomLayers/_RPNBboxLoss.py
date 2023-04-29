import tensorflow as tf
from tensorflow.keras import backend as K
from resources.Data_utils import batch_pack_graph
from ._Common import smooth_l1_loss


class RPNBboxLoss(tf.keras.layers.Layer):
    def __init__(self, images_per_gpu, *args, **kwargs):
        super(RPNBboxLoss, self).__init__(**kwargs)
        self.images_per_gpu = images_per_gpu

    @tf.function
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
        # Positive anchors contribute to the loss, but negative and
        # neutral anchors (match value of 0 or -1) don't.
        rpn_match = K.squeeze(inputs[1], -1)
        rpn_bbox = inputs[2]
        indices = tf.where(tf.equal(rpn_match, 1))
        del inputs
        # Pick bbox deltas that contribute to the loss
        rpn_bbox = tf.gather_nd(rpn_bbox, indices)

        # Trim target bounding box deltas to the same length as rpn_bbox.
        batch_counts = K.sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)
        target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                       self.images_per_gpu)

        loss = smooth_l1_loss(target_bbox, rpn_bbox, name="RPNBboxLoss")
        self.add_metric(loss, name="rpn_bbox_loss")
        return K.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "images_per_gpu": self.images_per_gpu,
        })
        return config