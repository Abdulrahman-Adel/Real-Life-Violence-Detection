import datetime

from dotmap import DotMap

from data_loader.data_loader_01 import *
from models.model_01 import VisionTransformer
from trainers.trainer import ModelTrainer

TRAIN_CONFIG = {
    "exp": {"name": "Experiment 1"},
    "trainer": {
        "name": "trainer.ModelTrainer",
        "EPOCHS": 100,
        "verbose_training": False,
        "save_pickle": True,
    },
    "callbacks": {
        "checkpoint_dir": "/checkpoint/"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        + "/",
        "checkpoint_monitor": "val_loss",
        "checkpoint_mode": "min",
        "checkpoint_save_best_only": True,
        "checkpoint_save_weights_only": False,
        "checkpoint_verbose": 1,
        "ESPatience": 5,
        "lrSPatience": 3,
        "lrSmin_lr": 1e-6,
    },
}
TRAIN_CONFIG = DotMap(TRAIN_CONFIG)

SEED = 42
CLASSES = 2
N_HEADS = 3
CHANNELS = 3
DROPOUT = 0.1
IMG_SIZE = 224
BATCH_SIZE = 16
PATCH_SIZE = 16
EMBED_SIZE = 198
MLP_HIDDEN = EMBED_SIZE * 4
ENCODER_BLOCKS = 5
SEQUENCE_LENGTH = 16
AUTOTUNE = tf.data.AUTOTUNE
DATADIR = "./data/Real Life Violence Dataset"


def main():
    # Build Dataset
    train, test, validation = build_dataframe(path=DATADIR)
    tr, ts, val = build_dataset(train, test, validation)
    # Build Model

    vit_model = VisionTransformer(
        n_heads=N_HEADS,
        n_classes=CLASSES,
        img_size=IMG_SIZE,
        mlp_dropout=DROPOUT,
        pos_dropout=0.0,
        attn_dropout=0.0,
        embed_size=EMBED_SIZE,
        patch_size=PATCH_SIZE,
        n_blocks=ENCODER_BLOCKS,
        mlpHidden_size=MLP_HIDDEN,
    )

    vit_model.build([None, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, CHANNELS])

    vit_model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1.1),
        metrics=[
            "accuracy",
        ],
    )

    TRAINER = ModelTrainer(vit_model, ts, val, TRAIN_CONFIG)
    TRAINER.train()


if __name__ == "__main__":
    main()
