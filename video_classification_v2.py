"""
Title: Video Classification with a CNN-RNN Architecture
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2021/05/28
Last modified: 2021/06/05
Description: Training a video classifier with transfer learning and a recurrent model on the UCF101 dataset.
"""
"""
This example demonstrates video classification, an important use-case with
applications in recommendations, security, and so on.
We will be using the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php)
to build our video classifier. The dataset consists of videos categorized into different
actions, like cricket shot, punching, biking, etc. This dataset is commonly used to
build action recognizers, which are an application of video classification.

A video consists of an ordered sequence of frames. Each frame contains *spatial*
information, and the sequence of those frames contains *temporal* information. To model
both of these aspects, we use a hybrid architecture that consists of convolutions
(for spatial processing) as well as recurrent layers (for temporal processing).
Specifically, we'll use a Convolutional Neural Network (CNN) and a Recurrent Neural
Network (RNN) consisting of [GRU layers](https://keras.io/api/layers/recurrent_layers/gru/).
This kind of hybrid architecture is popularly known as a **CNN-RNN**.

This example requires TensorFlow 2.5 or higher, as well as TensorFlow Docs, which can be
installed using the following command:
"""

"""shell
pip install -q git+https://github.com/tensorflow/docs
"""

"""
## Data collection

In order to keep the runtime of this example relatively short, we will be using a
subsampled version of the original UCF101 dataset. You can refer to
[this notebook](https://colab.research.google.com/github/sayakpaul/Action-Recognition-in-TensorFlow/blob/main/Data_Preparation_UCF101.ipynb)
to know how the subsampling was done.
"""

"""shell
wget -q https://git.io/JGc31 -O ucf101_top5.tar.gz
tar xf ucf101_top5.tar.gz
"""

"""
## Setup
"""

# from tensorflow_docs.vis import embed
from tensorflow import keras
# from imutils import paths

# import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)
import pandas as pd
import numpy as np
# import imageio
import cv2
import os
# import multiprocessing
# import time

import sys
import os

"""
## Define hyperparameters
"""

# IMG_SIZE = 224
IMG_SIZE = 96
BATCH_SIZE = 64
# EPOCHS = 6
EPOCHS = 10

MAX_SEQ_LENGTH = 100
NUM_FEATURES = 2048
# NUM_FEATURES = 1024
# keras.utils.set_random_seed(1)

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


"""
We can use a pre-trained network to extract meaningful features from the extracted
frames. The [`Keras Applications`](https://keras.io/api/applications/) module provides
a number of state-of-the-art models pre-trained on the [ImageNet-1k dataset](http://image-net.org/).
We will be using the [InceptionV3 model](https://arxiv.org/abs/1512.00567) for this purpose.
"""


def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

def prepare_all_videos(df, root_dir, label_processor, feature_extractor):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(
            shape=(
                1,
                MAX_SEQ_LENGTH,
            ),
            dtype="bool",
        )
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[
            idx,
        ] = temp_frame_features.squeeze()
        frame_masks[
            idx,
        ] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels

def prepare_all_videos_3_col(df, label_processor, feature_extractor):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()
    video_p_s = df["path"].values.tolist()

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        print(idx, num_samples)
        frames = load_video(os.path.join(video_p_s[idx], path))
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(
            shape=(
                1,
                MAX_SEQ_LENGTH,
            ),
            dtype="bool",
        )
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[
            idx,
        ] = temp_frame_features.squeeze()
        frame_masks[
            idx,
        ] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels

# Utility for our sequence model.
def get_sequence_model_gru(label_processor):
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model

def get_sequence_model_lstm(label_processor):
    class_vocab = label_processor.get_vocabulary()

    print("class_vocab: ", class_vocab)

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.LSTM(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.LSTM(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model


# Utility for running experiments.
def run_experiment(label_processor, train_data, train_labels, test_data, test_labels, idx):
    filepath = "/tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model_lstm(label_processor)
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    # eva_ret, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    # print(f"Test accuracy: {idx},{round(accuracy * 100, 2)}%")
    #
    # print(eva_ret, accuracy)

    return history, seq_model

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(
        shape=(
            1,
            MAX_SEQ_LENGTH,
        ),
        dtype="bool",
    )
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def frames_prediction_p(v_classify, path):
    class_vocab = ['macrophage', 'myeloma']

    print(path)
    frames = load_video(os.path.join("test", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = v_classify.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return class_vocab[0] if probabilities[0] > probabilities[1] else class_vocab[1]

def test_data_predict(v_classify, test_data, i, class_vocab):
    # class_vocab = ['macrophage', 'myeloma'] # I need to change it.
    class_vocab = ['macrophage', 'monocyte', 'myeloma']

    # probabilities = v_classify.predict([frame_features, frame_mask])[0]
    probabilities = v_classify.predict([test_data[0][i][None, ...], test_data[1][i][None, ...]])[0]

    # for i in np.argsort(probabilities)[::-1]:
    #     print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    # return class_vocab[0] if probabilities[0] > probabilities[1] else class_vocab[1]
    class_vocab[np.argmax(probabilities)]

# def sequence_prediction(v_classify, path):
#     class_vocab = label_processor.get_vocabulary()
#
#     frames = load_video(os.path.join("test", path))
#     frame_features, frame_mask = prepare_single_video(frames)
#     probabilities = v_classify.predict([frame_features, frame_mask])[0]
#
#     for i in np.argsort(probabilities)[::-1]:
#         print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
#     return frames

def frames_prediction_f(v_classify, frames):
    class_vocab = ['macrophage', 'myeloma']

    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = v_classify.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return class_vocab[0] if probabilities[0] > probabilities[1] else class_vocab[1]


# This utility is for visualization.
# Referenced from:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
# def to_gif(images):
#     converted_images = images.astype(np.uint8)
#     imageio.mimsave("animation.gif", converted_images, fps=10)
#     return embed.embed_file("animation.gif")


def one_split(idx, train_p, test_p, out_path):
    """
    ## Data preparation
    """

    train_df = pd.read_csv(train_p)
    if(test_p):
        test_df = pd.read_csv(test_p)

    print(f"Total videos for training: {len(train_df)}")

    if (test_p):
        print(f"Total videos for testing: {len(test_df)}")

    sample = train_df.sample(10)
    print("sample: ", sample)



    # The following two methods are taken from this tutorial:
    # https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub



    feature_extractor = build_feature_extractor()


    label_processor = keras.layers.StringLookup(
        num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
    )
    print(label_processor.get_vocabulary())



    # train_data, train_labels = prepare_all_videos(train_df, "train")
    # test_data, test_labels = prepare_all_videos(test_df, "test")
    # train_data, train_labels = prepare_all_videos(train_df, "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/TimeLapseVideos/ML/dataset/", label_processor, feature_extractor)
    # test_data, test_labels = prepare_all_videos(test_df, "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/TimeLapseVideos/ML/dataset/", label_processor, feature_extractor)

    test_data, test_labels = None, None
    train_data, train_labels = prepare_all_videos_3_col(train_df, label_processor, feature_extractor)
    # test_data, test_labels = prepare_all_videos_3_col(test_df, label_processor, feature_extractor)

    print(f"Frame features in train set: {train_data[0].shape}")
    print(f"Frame masks in train set: {train_data[1].shape}")


    expt_ret, sequence_model = run_experiment(label_processor, train_data, train_labels, test_data, test_labels, idx)

    os.makedirs(out_path, exist_ok=True)

    sequence_model.save(out_path + "v_classify_" + str(idx) + ".h5")
    sequence_model.save("./v_classify.h5")


    # v_classify = tf.keras.models.load_model("v_classify.h5")
    # v_classify = sequence_model


    # f = open(out_path + "test_ret_" + str(idx) + ".txt", "w")
    # # vt = train_df["video_name"].values.tolist()
    # vl = test_df["video_name"].values.tolist()
    # # va = vl + vt
    #
    # for i in range(len(vl)):
    #     print(f"Test video path: {vl[i]}")
    #     # test_frames = sequence_prediction(v)
    #
    #     # ret = frames_prediction_p(sequence_model, vl[i])
    #     ret = test_data_predict(sequence_model, test_data, i)
    #     # f.write(vl[i] + "," + ret + "\n")
    #     f.write(test_df["path"][i] + vl[i] + "," + "g_" + str(test_labels[i][0]) + "," + ret + "\n")
    #
    # f.close()

    # test_video = np.random.choice(test_df["video_name"].values.tolist())
    # print(f"Test video path: {test_video}")
    # test_frames = sequence_prediction(test_video)
    # to_gif(test_frames[:MAX_SEQ_LENGTH])

# processes = []
# for i in range(100):
#     train_p = "/home/qibing/Work/cell_ml_3_to_1/splits_600/train_" + str(i) + ".csv"
#     test_p = "/home/qibing/Work/cell_ml_3_to_1/splits_600/test_" + str(i) + ".csv"
#     out_path = "/home/qibing/Work/cell_ml_3_to_1/split_600_ret/"
#     # train_p = "/home/qibing/Work/cell_ml_3_to_1/splits/train_" + str(i) + ".csv"
#     # test_p = "/home/qibing/Work/cell_ml_3_to_1/splits/test_" + str(i) + ".csv"
#     # out_path = "/home/qibing/Work/cell_ml_3_to_1/split_ret_test/"
#     os.makedirs(out_path, exist_ok=True)
#
#     # one_split(i, train_p, test_p, out_path)
#
#     try:
#         p = multiprocessing.Process(target=one_split, args=(i, train_p, test_p, out_path))
#         p.start()
#         processes.append(p)
#         # print(time.strftime("%d_%H_%M ", time.localtime()), p, input_path)
#
#     except Exception as e:  # work on python 3.x
#         print('Exception: ' + str(e))
#
#     while len(processes) > 4:
#         # print(time.strftime("%d_%H_%M ", time.localtime()), len(processes), " processes are running.", file=log_f)
#         for p in processes:
#             if(p.is_alive() == False):
#                 # print(time.strftime("%d_%H_%M ", time.localtime()), p, file=log_f)
#                 # log_f.flush()
#                 p.terminate()
#                 processes.remove(p)
#                 break
#             else:
#                 # print(time.strftime("%d_%H_%M ", time.localtime()), p, file=log_f)
#                 pass
#         time.sleep(60)
#
# while len(processes) > 0:
#     # print(time.strftime("%d_%H_%M ", time.localtime()), len(processes), " processes are running.", file=log_f)
#     for p in processes:
#         if(p.is_alive() == False):
#             # print(time.strftime("%d_%H_%M ", time.localtime()), p, file=log_f)
#             # log_f.flush()
#             p.terminate()
#             processes.remove(p)
#             break
#         else:
#             # print(time.strftime("%d_%H_%M ", time.localtime()), p, file=log_f)
#             pass
#     time.sleep(60)
#
#     # log_f.close()
#

def error_augment():

    out_path = "/home/qibing/Work/cell_ml_3_to_1/split_600_ret_v3_copy/split_600_ret/"
    new_dataset_p = "/home/qibing/Work/cell_ml_3_to_1/split_600_ret_v3_copy/splits_600/error_augment/"

    for i in range(10):
        sequence_model = tf.keras.models.load_model(out_path + "v_classify_" + str(i) + ".h5")
        v_classify = sequence_model

        train_p = "/home/qibing/Work/cell_ml_3_to_1/splits_600/train_" + str(i) + ".csv"
        test_p = "/home/qibing/Work/cell_ml_3_to_1/splits_600/test_" + str(i) + ".csv"
        # train_p = "/home/qibing/Work/cell_ml_3_to_1/splits_600/train_" + str(i) + "_test.csv"
        # test_p = "/home/qibing/Work/cell_ml_3_to_1/splits_600/test_" + str(i) + "_test.csv"

        train_df = pd.read_csv(train_p)
        test_df = pd.read_csv(test_p)

        print(f"Total videos for training: {len(train_df)}")
        print(f"Total videos for testing: {len(test_df)}")

        feature_extractor = build_feature_extractor()

        label_processor = keras.layers.StringLookup(
            num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
        )
        print(label_processor.get_vocabulary())


        # train_data, train_labels = prepare_all_videos(train_df, "train")
        # test_data, test_labels = prepare_all_videos(test_df, "test")
        train_data, train_labels = prepare_all_videos(train_df,
                                                      "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/TimeLapseVideos/ML/dataset/",
                                                      label_processor, feature_extractor)
        test_data, test_labels = prepare_all_videos(test_df,
                                                    "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/TimeLapseVideos/ML/dataset/",
                                                    label_processor, feature_extractor)

        print(f"Frame features in train set: {train_data[0].shape}")
        print(f"Frame masks in train set: {train_data[1].shape}")


        # probabilities = v_classify.predict([frame_features, frame_mask])[0]
        p0 = v_classify.predict([train_data[0], train_data[1]])
        p1 = v_classify.predict([test_data[0], test_data[1]])

        new_f = open(new_dataset_p + "train_" + str(i) + "_err_au.csv", "w")
        class_vocab = label_processor.get_vocabulary()
        for p in [p0, ]:
            cnt = 0
            for i in range(len(p)):
                ret_str = class_vocab[0] if p[i][0] > p[i][1] else class_vocab[1]
                if(ret_str == class_vocab[train_labels[i][0]]):
                    cnt += 1
                else:
                    # print(train_df["video_name"][i], ret_str)
                    if(class_vocab[train_labels[i][0]] == "macrophage"):
                        new_f.write(train_df["video_name"][i] + "," + class_vocab[train_labels[i][0]] + "\n")

            print("accuracy: ", cnt / len(p))

        for p in [p1, ]:
            cnt = 0
            for i in range(len(p)):
                ret_str = class_vocab[0] if p[i][0] > p[i][1] else class_vocab[1]
                if(ret_str == class_vocab[test_labels[i][0]]):
                    cnt += 1
                else:
                    # print(test_df["video_name"][i], ret_str)
                    # new_f.write(test_df["video_name"][i] + class_vocab[test_labels[i][0]])
                    pass

            print("accuracy: ", cnt / len(p))
def evaluate(idx, test_p, out_path):

    test_df = pd.read_csv(test_p)
    print(f"Total videos for testing: {len(test_df)}")

    feature_extractor = build_feature_extractor()

    label_processor = keras.layers.StringLookup(
        num_oov_indices=0, vocabulary=np.unique(test_df["tag"])
    )
    print(label_processor.get_vocabulary())

    test_data, test_labels = prepare_all_videos_3_col(test_df, label_processor, feature_extractor)


    # sequence_model.save(out_path + "v_classify_" + str(idx) + ".h5")
    v_classify = tf.keras.models.load_model(out_path + "v_classify_" + str(idx) + ".h5")
    sequence_model = v_classify

    f = open(out_path + "test_ret_" + str(idx) + ".txt", "w")
    # vt = train_df["video_name"].values.tolist()
    vl = test_df["video_name"].values.tolist()
    # va = vl + vt

    for i in range(len(vl)):
        print(f"Test video path: {vl[i]}")
        # test_frames = sequence_prediction(v)

        # ret = frames_prediction_p(sequence_model, vl[i])
        ret = test_data_predict(sequence_model, test_data, i)
        f.write(test_df["path"][i] + vl[i] + "," + "g_" + str(test_labels[i][0]) + "," + ret + "\n")

    f.close()

def evaluate_v2(cat, model_path, test_p, out_path):

    # v_classify = tf.keras.models.load_model(model_path)

    v_classify = tf.keras.models.load_model(model_path, compile=False)
    v_classify.compile()

    sequence_model = v_classify


    test_df = pd.read_csv(test_p)
    print(f"Total videos for testing: {len(test_df)}")

    feature_extractor = build_feature_extractor()

    label_processor = keras.layers.StringLookup(
        num_oov_indices=0, vocabulary=np.unique(test_df["tag"])
    )

    # class_vocab = label_processor.get_vocabulary()
    class_vocab = cat
    print(class_vocab)

    test_data, test_labels = prepare_all_videos_3_col(test_df, label_processor, feature_extractor)

    f_name = os.path.basename(test_p)
    f = open(out_path + f_name[:-4] + "_ret.txt", "w")
    # vt = train_df["video_name"].values.tolist()
    vl = test_df["video_name"].values.tolist()
    # va = vl + vt

    for i in range(len(vl)):
        print(f"Test video path: {vl[i]}")
        # test_frames = sequence_prediction(v)

        # ret = frames_prediction_p(sequence_model, vl[i])
        # ret = test_data_predict(sequence_model, test_data, i, class_vocab)

        probabilities = sequence_model.predict([test_data[0][i][None, ...], test_data[1][i][None, ...]])[0]
        print(probabilities)
        ret = class_vocab[np.argmax(probabilities)]
        print(ret)

        f.write(test_df["path"][i] + vl[i] + "," + "g_" + test_df["tag"][i] + "," + ret + "\n")

    f.close()

def accracy_analyse2(out_path):

    # out_path = '/home/qibing/disk_16t/qibing/work/cell_ml_3_to_1/one_bea_train/'

    for idx in range(1):
        f = open(out_path + "test_ret_" + str(idx) + ".txt", "r")
        data = f.read()
        lines = data.split("\n")
        # print(data.split(".avi"))
        ret = []

        myeloma = []
        macrophage = []

        for l in lines:
            b = []
            if ("myeloma" in l):
                b = [0, 1]
            elif ("macrophage" in l):
                b = [1, 0]
            else:
                print("another thing is wrong.")

            if ("g_1" in l):
                # a = "myeloma"
                myeloma.append(b)
            elif ("g_0" in l):
                # a = "macrophage"
                macrophage.append(b)
            else:
                print("something is wrong.")

        # print(macrophage, myeloma)
        ma_np = np.array(macrophage)
        my_np = np.array(myeloma)
        # print(len(macrophage), len(myeloma))

        # print("%d,%.1f,%.1f,%.1f,%.1f" % (idx, ma_np.sum(axis=0)[0], my_np.sum(axis=0)[1]/100 * 100, (ma_np.sum(axis=0)[0] + my_np.sum(axis=0)[1])/200 * 100), sep = ',', file = f_split_sum)
        # print("%d,%.1f,%.1f,%.1f,%.1f" % (idx, ma_np.sum(axis=0)[0], ma_np.sum(axis=0)[1], my_np.sum(axis=0)[0]/100 * 100, my_np.sum(axis=0)[1]), sep = ',', file = f_split_sum)
        print(idx, ma_np.sum(axis=0)[0], ma_np.sum(axis=0)[1], my_np.sum(axis=0)[0], my_np.sum(axis=0)[1])

def accracy_analyse2_v2_one_ret(out_path):

    f = open(out_path)
    data = f.read()
    lines = data.split("\n")
    # print(data.split(".avi"))
    ret = []

    myeloma = []
    macrophage = []

    for l in lines:
        b = []
        if ("myeloma" in l):
            b = [0, 1]
        elif ("macrophage" in l):
            b = [1, 0]
        else:
            print("another thing is wrong. The line is: ", l)

        if ("g_1" in l):
            # a = "myeloma"
            myeloma.append(b)
        elif ("g_0" in l):
            # a = "macrophage"
            macrophage.append(b)
        else:
            print("something is wrong. The line is: ", l)

    # print(macrophage, myeloma)
    ma_np = np.array(macrophage)
    my_np = np.array(myeloma)
    # print(len(macrophage), len(myeloma))

    # print("%d,%.1f,%.1f,%.1f,%.1f" % (idx, ma_np.sum(axis=0)[0], my_np.sum(axis=0)[1]/100 * 100, (ma_np.sum(axis=0)[0] + my_np.sum(axis=0)[1])/200 * 100), sep = ',', file = f_split_sum)
    # print("%d,%.1f,%.1f,%.1f,%.1f" % (idx, ma_np.sum(axis=0)[0], ma_np.sum(axis=0)[1], my_np.sum(axis=0)[0]/100 * 100, my_np.sum(axis=0)[1]), sep = ',', file = f_split_sum)
    # print(ma_np.sum(axis=0)[0], ma_np.sum(axis=0)[1], my_np.sum(axis=0)[0], my_np.sum(axis=0)[1])
    print(ma_np.sum(axis=0)[0], ma_np.sum(axis=0)[1])

def accuracy_table(cat, out_path):
    print(os.path.basename(out_path))

    idx = ["g_" + c for c in cat]
    ret = pd.DataFrame(index=idx, columns=cat)
    ret[:][:] = 0
    data = pd.read_csv(out_path, names = ["path", "gt", "prd"])#ground truth, prediction
    for i in data.index:
        ret[data["prd"][i]][data["gt"][i]] += 1

    ret.to_csv(out_path[:-4] + "_accuracy_table.txt")
    print(ret)




# for i in range(1, 10):
#     # train_p = "/home/qibing/Work/cell_ml_3_to_1/splits_600/train_" + str(i) + ".csv"
#     train_p = '/home/qibing/Work/cell_ml_3_to_1/split_600_ret_v3_copy_2/splits_600/error_augment/train_' + str(i) + '_err_au_2.csv'
#     test_p = "/home/qibing/Work/cell_ml_3_to_1/splits_600/test_" + str(i) + ".csv"
#     out_path = '/home/qibing/Work/cell_ml_3_to_1/split_600_ret_v3_copy_2/splits_600/error_augment/'
#     one_split(i, train_p, test_p, out_path)
# #
# train_p = '/home/qibing/disk_16t/qibing/work/cell_ml_3_to_1/extra_train_MAX_SEQ_LENGTH_100/train.csv'
# test_p = '/home/qibing/disk_16t/qibing/work/cell_ml_3_to_1/extra_train_MAX_SEQ_LENGTH_100/test.csv'
# out_path = '/home/qibing/disk_16t/qibing/work/cell_ml_3_to_1/extra_train_MAX_SEQ_LENGTH_100/'

# train_p = './two_bea_train/train.csv'
# test_p = './two_bea_train/test.csv'
# # out_path = './two_bea_train/two_bea_train_lstm/'
# out_path = './two_bea_train/two_bea_train_gru/'
# # one_split(0, train_p, test_p, out_path)
# evaluate(0, test_p, out_path)
# # accracy_analyse2(out_path)

# path = "./train_pt935_v8_7bea/"

# one_split(0, path + "train.csv", path + "test.csv", out_path=path)
# one_split(0, path + "train.csv", None, out_path=path)
# evaluate(0, path + "test.csv", out_path=path)

# model = "/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea/v_classify_0.h5"
# for i in range(6, 9):
#     evaluate_v2(model, path + "test_" + str(i) + ".csv", out_path=path)

# print(sys.argv[:])
# model = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v8_7bea/v_classify_0.h5'
# data = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v8_7bea_2lstm/mm_test_' + str(sys.argv[1]) + '.csv'
# out_path = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v8_7bea_2lstm/test_ret_b/'
# os.makedirs(out_path, exist_ok = True)
# evaluate_v2(model, data, out_path=out_path)


# accracy_analyse2(path)

# for i in range(2, 9):
#     accracy_analyse2_v2_one_ret('/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v8_7bea_2lstm/test_ret_b/mo_test_' + str(i) + '_ret.txt')

# for i in range(2, 6):
#     accracy_analyse2_v2_one_ret('/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v8_7bea_2lstm/test_ret_b/mm_test_' + str(i) + '_ret.txt')

# path = "./train_pt935_v8_7bea_2lstm_2cat_mm_mono/"
# one_split(0, path + "train.csv", None, out_path=path)

# print(sys.argv[:])
# model = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v8_7bea_2lstm_2cat_mm_mono/v_classify_0.h5'
# data = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v8_7bea_2lstm_2cat_mm_mono/test.csv'
# out_path = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v8_7bea_2lstm_2cat_mm_mono/'
# os.makedirs(out_path, exist_ok = True)
# evaluate_v2(model, data, out_path=out_path)# change something in test_data_predict



# accuracy_table(['macrophage', 'monocyte', 'myeloma'], '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v8_7bea_2lstm/test_ret.txt')

# accuracy_table(['monocyte', 'myeloma'], '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v8_7bea_2lstm_2cat_mm_mono/test_ret.txt')


# model = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v8_7bea/v_classify_0.h5'
# data = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v8_7bea_2lstm/all_raw_data/mo_test_2.csv'
# out_path = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v8_7bea/'
# os.makedirs(out_path, exist_ok = True)
# cat = ['macrophage', 'myeloma']
# # cat = ['macrophage', 'monocyte', 'myeloma']
# evaluate_v2(cat, model, data, out_path=out_path)# change something in test_data_predict

# accuracy_table(['macrophage', 'monocyte', 'myeloma'], '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v8_7bea_2lstm/train_7500_cat3/macro_test_ret.txt')

# path = "./train_pt935_v9_7bea_2lstm_layer1/"
# one_split(0, path + "train.csv", None, out_path=path)
#

# print(sys.argv[:])
# model = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v9_7bea_2lstm_layer1/v_classify_0.h5'
# # data = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v9_7bea_2lstm_layer1/raw_data/' + str(sys.argv[1])
# data = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v9_7bea_2lstm_layer1/mm_tmp.csv'
# out_path = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v9_7bea_2lstm_layer1/test_ret_b/'
# os.makedirs(out_path, exist_ok = True)
# cat = ['macrophage', 'myeloma']
# evaluate_v2(cat, model, data, out_path=out_path)

# for i in range(3, 13):
#     accuracy_table(['macrophage', 'myeloma'], '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v9_7bea_2lstm_layer1/test_ret_b/mo_test_' + str(i) + '_ret.txt')


# path = "./train_pt935_v9_7bea_2lstm_layer2_v2_mm_mono/"
# one_split(0, path + "train.csv", None, out_path=path)

# print(sys.argv[:])
# model = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v9_7bea_2lstm_layer2_mm_mono/v_classify_0.h5'
# out_path = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v9_7bea_2lstm_layer2_mm_mono/'
# os.makedirs(out_path, exist_ok = True)
# cat = ['monocyte', 'myeloma']
# # data = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v9_7bea_2lstm_layer1/macro_head_1000.csv'
# # evaluate_v2(cat, model, data, out_path=out_path)
# data = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v9_7bea_2lstm_layer1/mm_test_3_ret_mm.csv'
# evaluate_v2(cat, model, data, out_path=out_path)
#
# print(sys.argv[:])
# model = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea/v_classify_0.h5'
# out_path = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea_2nd_layer/test_b/'
# os.makedirs(out_path, exist_ok = True)
# cat = ['macrophage', 'myeloma']
# data = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v9_7bea_2lstm_layer1/raw_data/' + str(sys.argv[1])
# evaluate_v2(cat, model, data, out_path=out_path)

# for i in range(0, 13):
#     accuracy_table(['macrophage', 'myeloma'], '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea_2nd_layer/test_b/mo_test_' + str(i) + '_ret.txt')

# path = "./train_pt935_v7_7bea_2nd_layer/"
# path = "./train_pt935_v7_7bea_2nd_layer_with_blurred_bea/"
# one_split(0, path + "train.csv", None, out_path=path)

# print(sys.argv[:])
# model = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea_2nd_layer/v_classify_0.h5'
# out_path = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea_2nd_layer/'
# os.makedirs(out_path, exist_ok = True)
# cat = ['monocyte', 'myeloma']
# data = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea_2nd_layer/mm_mm_test.csv'
# evaluate_v2(cat, model, data, out_path=out_path)
# data = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea_2nd_layer/mo_mm_test.csv'
# evaluate_v2(cat, model, data, out_path=out_path)

# print(sys.argv[:])
# model = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea/v_classify_0.h5'
# out_path = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea_2nd_layer_with_blurred_bea/test_b/'
# os.makedirs(out_path, exist_ok = True)
# cat = ['macrophage', 'myeloma']
# for i in range(7):
#     data = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea_2nd_layer_with_blurred_bea/raw_data/mo_test_' + str(i) + '.csv'
#     evaluate_v2(cat, model, data, out_path=out_path)

# model = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea_2nd_layer_with_blurred_bea/v_classify_0.h5'
# out_path = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea_2nd_layer_with_blurred_bea/'
# os.makedirs(out_path, exist_ok = True)
# cat = ['monocyte', 'myeloma']
# data = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea_2nd_layer_with_blurred_bea/mm_mm_test.csv'
# evaluate_v2(cat, model, data, out_path=out_path)
# data = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea_2nd_layer_with_blurred_bea/mo_mm_test_0.csv'
# evaluate_v2(cat, model, data, out_path=out_path)
# data = '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea_2nd_layer_with_blurred_bea/mo_mm_test_1.csv'
# evaluate_v2(cat, model, data, out_path=out_path)

accuracy_table(['monocyte', 'myeloma'], '/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea_2nd_layer_with_blurred_bea/layer2_test.txt')


"""
## Next steps

* In this example, we made use of transfer learning for extracting meaningful features
from video frames. You could also fine-tune the pre-trained network to notice how that
affects the end results.
* For speed-accuracy trade-offs, you can try out other models present inside
`tf.keras.applications`.
* Try different combinations of `MAX_SEQ_LENGTH` to observe how that affects the
performance.
* Train on a higher number of classes and see if you are able to get good performance.
* Following [this tutorial](https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub), try a
[pre-trained action recognition model](https://arxiv.org/abs/1705.07750) from DeepMind.
* Rolling-averaging can be useful technique for video classification and it can be
combined with a standard image classification model to infer on videos.
[This tutorial](https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/)
will help understand how to use rolling-averaging with an image classifier.
* When there are variations in between the frames of a video not all the frames might be
equally important to decide its category. In those situations, putting a
[self-attention layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention) in the
sequence model will likely yield better results.
* Following [this book chapter](https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-11),
you can implement Transformers-based models for processing videos.
"""
