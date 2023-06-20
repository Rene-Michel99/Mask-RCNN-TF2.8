from logging import Logger
from tensorflow.keras.callbacks import Callback


class LoggerTraining(Callback):
    def __init__(self, logger: Logger):
        self.logger = logger

    def on_train_batch_end(self, batch, logs):
        self.logger.info('{} batch ended successfully - '.format(batch, logs))

    def on_epoch_end(self, epoch, logs):
        self.logger.info('{} epoch ended successfully - {}'.format(epoch, str(logs)))

