import tensorflow as tf


class GetAnchors(tf.keras.layers.Layer):
    def __int__(self):
        super(GetAnchors, self).__int__()

    def call(self, inputs, anchors):
        return tf.Variable(anchors)
