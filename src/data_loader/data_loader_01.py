import itertools
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
CHANNELS = 3
IMG_SIZE = 224
BATCH_SIZE = 16
PATCH_SIZE = 16
SEQUENCE_LENGTH = 16
AUTOTUNE = tf.data.AUTOTUNE


class VideoDataset:
    """
    Dataset generator class for the violance - nonviolace  video dataset
    """

    def __init__(self, df: pd.DataFrame, n_frames: int = 32):
        self.n_frames = n_frames
        self.dataframe = df
        self.class_names = {"NonViolence": 0, "Violence": 1}

    def load_video(self, path: Path, SEQUENCE_LENGTH: int):
        """Creates list of video frames tensors

        Returns:
            frames: numpy array of tensor frames
        """
        frames = list()

        Video_Caption = cv2.VideoCapture(str(path))

        frame_count = int(Video_Caption.get(cv2.CAP_PROP_FRAME_COUNT))
        pad_tensor = tf.constant(
            value=-2, shape=[IMG_SIZE, IMG_SIZE, CHANNELS], dtype=tf.float32
        )

        skip_frames_window = max(int(frame_count / SEQUENCE_LENGTH), 1)
        while Video_Caption.isOpened():
            current_frame = Video_Caption.get(1)
            ret, frame = Video_Caption.read()

            if ret != True:
                break

            if (
                current_frame % skip_frames_window == 0
                and len(frames) < SEQUENCE_LENGTH
            ):
                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
                image = tf.image.convert_image_dtype(rgb_img, dtype=tf.float32)
                image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
                frames.append(image)

        if len(frames) < SEQUENCE_LENGTH:
            for _ in range(SEQUENCE_LENGTH - len(frames)):
                frames.append(pad_tensor)
        return np.asarray(frames)

    def __len__(self):
        return len(self.dataframe)

    def __call__(self):
        for item in self.dataframe.iterrows():
            video_frames = self.load_video(item[1]["Video_Path"], self.n_frames)
            label = self.class_names[item[1]["Labels"]]
            yield video_frames, label


def build_dataframe(path: str):
    """create dataframes with [video_path, label] columns from videos directory,
    split with (60, 20, 20) percentages for train , test and validation data

    Args:
        path (str): directory path

    Returns:
        pd.DataFrame: _description_
    """
    data_path = Path(path)
    Video_Path = list(data_path.glob(r"*/*.mp4"))
    Video_Labels = list(
        map(lambda x: os.path.split(os.path.split(x)[0])[1], Video_Path)
    )
    dataframe = pd.DataFrame({"Video_Path": Video_Path, "Labels": Video_Labels})
    tr, val, ts = np.split(
        dataframe.sample(frac=1, random_state=SEED),
        [int(0.6 * len(dataframe)), int(0.8 * len(dataframe))],
    )
    return (
        tr.reset_index(drop=True),
        val.reset_index(drop=True),
        ts.reset_index(drop=True),
    )


def build_dataset(train, test, validation):
    """

    Args:
        train (Pandas DataFrame): [video_path, label] train set
        test (Pandas DataFrame): [video_path, label] test set
        validation (Pandas DataFrame): [video_path, label] validation set
    """
    output_signature = (
        tf.TensorSpec(shape=(None, None, None, CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int16),
    )

    train_ds = tf.data.Dataset.from_generator(
        VideoDataset(n_frames=SEQUENCE_LENGTH, df=train),
        output_signature=output_signature,
    )
    val_ds = tf.data.Dataset.from_generator(
        VideoDataset(n_frames=SEQUENCE_LENGTH, df=validation),
        output_signature=output_signature,
    )
    test_ds = tf.data.Dataset.from_generator(
        VideoDataset(n_frames=SEQUENCE_LENGTH, df=test),
        output_signature=output_signature,
    )

    tr = (
        train_ds.cache()
        .batch(batch_size=BATCH_SIZE, num_parallel_calls=AUTOTUNE)
        .prefetch(buffer_size=AUTOTUNE)
    )
    val = (
        val_ds.cache()
        .batch(batch_size=BATCH_SIZE, num_parallel_calls=AUTOTUNE)
        .prefetch(buffer_size=AUTOTUNE)
    )
    ts = (
        test_ds.cache()
        .batch(batch_size=BATCH_SIZE, num_parallel_calls=AUTOTUNE)
        .prefetch(buffer_size=AUTOTUNE)
    )

    return tr, ts, val


def plot_video(video: any, label: any):
    fig, axes = plt.subplots(6, 6, figsize=(20, 20))
    fig.suptitle(f"label: {label}", fontsize=30)
    fig.tight_layout()
    for frame, ax in enumerate(axes.ravel()):
        try:
            ax.imshow(video[frame])
        except:
            break
    plt.show()


if __name__ == "__main__":
    DIR_PATH = ".\Real Life Violence Dataset"
    train, test, validation = build_dataframe(path=DIR_PATH)

    gen = VideoDataset(n_frames=SEQUENCE_LENGTH, df=train)

    for video, label in itertools.islice(gen(), 5):
        plot_video(video, label)
