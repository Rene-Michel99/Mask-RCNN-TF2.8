import tensorflow as tf
from tensorflow.keras.models import Model


class MaskRCNNModel(Model):

    def same_rank_losses(self):
        for i in range(len(self.losses)):
            if self.losses[i].shape != tf.TensorShape([]):
                self.losses[i] = tf.reshape(self.losses[i], [])

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True) # noqa
            if y in [None, [], ()]:
                y_pred = []
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
