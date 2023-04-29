import gc
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback


class ClearMemory(Callback):
    def __int__(self):
        pass

    @staticmethod
    def on_epoch_end(epoch, logs=None):
        print("Cleaning memory for next epoch")
        gc.collect()
        K.clear_session()
        print("Cleaned!")
