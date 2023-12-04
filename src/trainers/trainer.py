import datetime
import os

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from base.base_trainer import BaseTrain


class ModelTrainer(BaseTrain):
    def __init__(self, model, data_train, data_validate, config):
        """_summary_

        Args:
            model (_type_): compiled model
            data_train (_type_): training data
            data_validate (_type_): validation data
        """
        super(ModelTrainer, self).__init__(model, data_train, data_validate, config)
        self.callbacks = []
        self.log_dir = (
            "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        )
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(
                    self.config.callbacks.checkpoint_dir,
                    "%s-{epoch:02d}-{val_loss:.4f}.tf" % self.config.exp.name,
                ),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                verbose=self.config.callbacks.checkpoint_verbose,
                save_format="tf",
            )
        )
        self.callbacks.append(
            EarlyStopping(patience=self.config.callbacks.ESPatience, monitor="val_loss")
        )
        self.callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=self.config.callbacks.lrSPatience,
                min_lr=self.config.callbacks.lrSmin_lr,
            )
        )

        self.callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        )

    def train(self):
        history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=self.config.trainer.EPOCHS,
            callbacks=self.callbacks,
        )

    if __name__ == "__main__":
        pass
