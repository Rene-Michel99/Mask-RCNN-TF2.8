import tensorflow as tf
import tensorflow.keras.backend as K


class RPNClassLoss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RPNClassLoss, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        """RPN anchor classifier loss.
            rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
                       -1=negative, 0=neutral anchor.
            rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
            """
        # Squeeze last dim to simplify
        rpn_match = tf.squeeze(inputs[0], -1)

        # Get anchor classes. Convert the -1/+1 match to 0/1 values.
        anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)
        # Positive and Negative anchors contribute to the loss,
        # but neutral anchors (match value = 0) don't.
        indices = tf.where(tf.not_equal(rpn_match, 0))
        # Pick rows that contribute to the loss and filter out the rest.
        rpn_class_logits = tf.gather_nd(inputs[1], indices)
        anchor_class = tf.gather_nd(anchor_class, indices)
        # Cross entropy loss
        loss = K.sparse_categorical_crossentropy(
            target=anchor_class,
            output=rpn_class_logits,
            from_logits=True
        )
        '''metric = tf.keras.metrics.sparse_categorical_crossentropy(
            anchor_class,
            rpn_class_logits,
            from_logits=True
        )'''
        self.add_metric(loss, name="rpn_class_loss")

        loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
        return loss
