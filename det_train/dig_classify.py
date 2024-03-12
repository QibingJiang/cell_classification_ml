#!/usr/bin/env python3
# 安装 TensorFlow
import tensorflow as tf
import numpy as np
import os
import cv2
import time

import keras.backend as K
import random

def my_train():

    stroma_p_0 = "/home/qibing/disk_16t/qibing/bad_exp/Pt352_SOCCO/pt352_beacon_83_stroma_0/"
    stroma_p_1 = "/home/qibing/disk_16t/qibing/bad_exp/Pt352_SOCCO/pt352_beacon_83_stroma_1/"
    stroma_p_2 = "/home/qibing/disk_16t/qibing/bad_exp/Pt352_SOCCO/pt352_beacon_83_stroma_2/"
    stroma_p_3 = "/home/qibing/disk_16t/qibing/bad_exp/Pt352_SOCCO/pt352_beacon_83_stroma_3/"
    stroma_p = [stroma_p_0, stroma_p_1, stroma_p_2, stroma_p_3]

    background_p = "/home/qibing/disk_16t/Pt210/TimeLapseVideos/ML/background_2/"
    dark_p = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/TimeLapseVideos/ML/lstm_1/Beacon_49_cells_train_dark_myeloma/"
    bright_p = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/TimeLapseVideos/ML/lstm_1/Beacon_49_cells_train_bright_myeloma/"
    macrophage_p = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/TimeLapseVideos/ML/lstm_1/Beacon_1_cells_train_macrophage/"
    white_edge_p = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/TimeLapseVideos/ML/Beacon_61_cells_train_false_white_edge/"
    black_cell_out_focus_p = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/TimeLapseVideos/ML/Beacon_61_cells_train_black_cells_out_focus/"

    # ps = [background_p, dark_p, bright_p, white_edge_p, black_cell_out_focus_p]

    ps = [background_p, white_edge_p, dark_p, bright_p, black_cell_out_focus_p]

    stroma = []
    for p in stroma_p:
        f = os.listdir(p)
        for i in range(len(f)):
            cell = cv2.imread(p + f[i])[:,:,0:1]#, cv2.IMREAD_GRAYSCALE
            stroma.append(cell)


    cells = []
    for i in range(len(ps)):
        p = ps[i]
        f = os.listdir(p)
        cells.append([])
        for j in range(len(f)):
            cell = cv2.imread(p + f[j])[:,:,0:1]#, cv2.IMREAD_GRAYSCALE
            cells[i].append(cell)

    cells.insert(0, stroma)

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    # y = [np.nan]*1100
    for i in range(len(cells)):
        x_train += cells[i][0:1000]
        x_test += cells[i][1000:1100]
        y = [i]*1100
        y_train += y[0:1000]
        y_test += y[1000:1100]

    x_train = np.stack(x_train)
    x_test = np.stack(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    y_train[:3000] = 0
    y_train[3000:] = 1
    y_test[:300] = 0
    y_test[300:] = 1


    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(12, 12, 1)),
      # tf.keras.layers.Conv2D(6, (3,3), activation='relu', input_shape=(10, 10, 3)),
      tf.keras.layers.Flatten(),

      # tf.keras.layers.Flatten(input_shape=(10, 10)),

      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(len(cells), activation='softmax')
    ])

    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # sess = K.get_session()
    # print(model.get_layer("conv2d").kernel)


    # while(True):
    #     idx = random.randint(0, len(x_train))
    #     print(idx)
    #     cv2.imshow(str(y_train[idx]), x_train[idx])
    #     cv2.waitKey()

    model.fit(x_train, y_train, epochs=5)
    ret = model.evaluate(x_test,  y_test, verbose=2)
    print("evaluate: ", ret)

    # sess = K.get_session()
    # print(model.get_layer("conv2d").kernel)

    model.save("det.h5")

# def predict():
    det_ml = tf.keras.models.load_model("det.h5")

    temp_t = time.time()
    ret = det_ml.predict(x_test)
    # print(ret)
    np.savetxt("./pred_ret.txt", ret)
    # ret = det_ml.predict(x_test[400:500])
    # print(ret)
    print("predict time:", time.time() - temp_t)

def post_analyse():
    data = np.loadtxt("./pred_ret.txt")
    # print(data)

    cla = [np.argmax(ret) for ret in data]
    # print(cla)
    np.savetxt("./classify.txt", cla, fmt="%d")

    cla_np = np.array(cla)

    ret_1 = []
    num_categories = 2
    for i in range(num_categories):
        cla_one = cla_np[i*300:(i+1)*300]
        cla_one_4 = []

        for j in range(num_categories):
            cla_one_4.append(np.count_nonzero(cla_one == j))

        ret_1.append(cla_one_4)

    print(ret_1)



# my_train()
post_analyse()
# model.evaluate(x_test[0:3])
