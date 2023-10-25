import tensorflow as tf
from tensorflow.keras.models import Model

from mrcnn.CustomLosses import *


class MaskRCNNModel(Model):

    def build_losses(self, config):
        self.custom_losses = []
        self.custom_losses.append(RPNClassLoss())
        self.custom_losses.append(RPNBboxLoss(config.IMAGES_PER_GPU))
        self.custom_losses.append(MRCNNClassLoss())
        self.custom_losses.append(MRCNNBboxLoss())
        self.custom_losses.append(MRCNNBboxLoss())
        self.custom_losses.append(MRCNNMaskLoss())

    def calc_losses(self, x, y):
        total_loss = 0
        for loss_fn in self.custom_losses:
            if hasattr(loss_fn, "metric"):
                total_loss += loss_fn(x, y)
            else:
                loss_fn()
        return total_loss

    def same_rank_losses(self):
        for i in range(len(self.losses)):
            if self.losses[i].shape != tf.TensorShape([]):
                self.losses[i] = tf.reshape(self.losses[i], [])

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True) # noqa
            #loss = self.compiled_loss(x, y_pred, regularization_losses=self.losses)
            loss = self.calc_losses(x, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, [])
        metrics = {loss.name: loss.metric for loss in self.custom_losses if hasattr(loss, 'metric')}
        metrics['loss'] = loss

        return metrics
