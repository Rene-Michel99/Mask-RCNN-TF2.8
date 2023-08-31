import tensorflow as tf
from tensorflow.keras.models import Model

from mrcnn.CustomLosses import *


class MaskRCNNModel(Model):

    def build(self, config):
        self.rpn_class_loss = RPNClassLoss()
        self.rpn_bbox_loss = RPNBboxLoss(config.IMAGES_PER_GPU)
        self.mrcnn_class_loss = MRCNNClassLoss()
        self.mrcnn_bbox_loss = MRCNNBboxLoss()
        self.mrcnn_mask_loss = MRCNNMaskLoss()

    def same_rank_losses(self):
        for i in range(len(self.losses)):
            if self.losses[i].shape != tf.TensorShape([]):
                self.losses[i] = tf.reshape(self.losses[i], [])

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True) # noqa
            '''if y in [None, [], ()]:
                y_pred = []'''
            #loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            loss = self.rpn_class_loss(x, y_pred)
            loss += self.rpn_bbox_loss(x, y_pred)
            loss += self.mrcnn_class_loss(x, y_pred)
            loss += self.mrcnn_bbox_loss(x, y_pred)
            loss += self.mrcnn_mask_loss(x, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, [])
        metrics = {
            'rpn_class_loss': self.rpn_class_loss.metric,
            'rpn_bbox_loss': self.rpn_bbox_loss.metric,
            'mrcnn_class_loss': self.mrcnn_class_loss.metric,
            'mrcnn_bbox_loss': self.mrcnn_bbox_loss.metric,
            'mrcnn_mask_loss': self.mrcnn_mask_loss.metric
        }

        return metrics
