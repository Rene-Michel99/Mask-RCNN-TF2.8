import tensorflow as tf


class GetAnchors(tf.keras.layers.Layer):
    def __init__(self, anchors, name="anchors", **kwargs):
        super(GetAnchors, self).__init__(name=name, **kwargs)
        self.anchors = tf.Variable(anchors)

    @tf.function
    def call(self, dummy):
        return self.anchors

    def get_config(self):
        config = super(GetAnchors, self).get_config()
        return config
