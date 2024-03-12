#!/usr/bin/env python3
# 安装 TensorFlow
import tensorflow as tf
import numpy as np
import os
import cv2
import time

import keras.backend as K

# path = "/home/qibing/disk_16t/Pt210/TimeLapseVideos/ML/cells/Beacon_73/"
#
# cells = []
# for tra_i in range(167):
#     for t_i in range(100):
#         cell = np.zeros((10, 10), dtype=np.int8)
#         cell_path = path + "{0:0=4d}".format(tra_i) + "_" + str(t_i) + ".tif"
#         if(os.path.exists(cell_path)):
#             cell = cv2.imread(cell_path, cv2.IMREAD_GRAYSCALE)
#             cells.append(cell)
#
#
# path_not_cell = "/home/qibing/disk_16t/Pt210/TimeLapseVideos/ML/not_cells/"
#
# not_cells = []
# for i in range(7000):
#     not_cell = cv2.imread(path_not_cell + str(i) + ".tif", cv2.IMREAD_GRAYSCALE)
#     not_cells.append(not_cell)
#
# x_train = np.stack( cells[:5000] + not_cells[:5000], axis=0 )
# # x_train = np.vstack(cells[:5000] + not_cells[:5000])
# x_test = np.stack(cells[5000:7000] + not_cells[5000:7000], axis=0 )
#
# y_train = np.zeros(10000, dtype = np.int8)
# y_train[:5000] = 1
# y_train[5000:] = 0
#
# y_test = np.zeros(4000, dtype = np.int8)
# y_test[:2000] = 1
# y_test[2000:] = 0

p0 = "/home/qibing/disk_16t/Pt210/TimeLapseVideos/ML/bright_cells/"
# p1 = "/home/qibing/disk_16t/Pt210/TimeLapseVideos/ML/dark_cells/"
# false_positive = "/home/qibing/disk_16t/qibing/macrophage/Macrophage_VENDAR_Pt641/TimeLapseVideos/ML/Beacon_147/cells_train/false_negtive_50/"
path_not_cell = "/home/qibing/disk_16t/Pt210/TimeLapseVideos/ML/not_cells/"
f_n = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/Pt796_MO_CD138_03102022/TimeLapseVideos/ML/Beacon_15/Pt796_bea_15_sel_cells/"

f_n_1 = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/Pt796_MO_CD138_03102022/TimeLapseVideos/ML/sel_cells/"

p0 = f_n
p1 = path_not_cell
f0 = os.listdir(p0)
f1 = os.listdir(p1)

f_n_1_s = os.listdir(f_n_1)

bright_cells = []
dark_cells = []

f_n_c_s = []

for i in range(1000):
    bright_cell = cv2.imread(p0 + f0[i])[:,:,0:1]#, cv2.IMREAD_GRAYSCALE
    # dark_cell = cv2.imread(p1 + f1[i])[:,:,0:1]#, cv2.IMREAD_GRAYSCALE
    # bright_cell = cv2.imread(p0 + f0[i])
    # dark_cell = cv2.imread(p1 + f1[i])
    bright_cells.append(bright_cell)
    # dark_cells.append(dark_cell)

for i in range(1800):
    # bright_cell = cv2.imread(p0 + f0[i])[:,:,0:1]#, cv2.IMREAD_GRAYSCALE
    dark_cell = cv2.imread(p1 + f1[i])[:,:,0:1]#, cv2.IMREAD_GRAYSCALE
    # bright_cell = cv2.imread(p0 + f0[i])
    # dark_cell = cv2.imread(p1 + f1[i])
    # bright_cells.append(bright_cell)
    dark_cells.append(dark_cell)


for i in range(1000):
    f_n_c = cv2.imread(f_n_1 + f_n_1_s[i])[:,:,0:1]
    f_n_c_s.append(f_n_c)


# not_cells = []
# for i in range(2000):
#     not_cell = cv2.imread(path_not_cell + str(i) + ".tif")[:,:,0:1]#, cv2.IMREAD_GRAYSCALE
#     # not_cell = cv2.imread(path_not_cell + str(i) + ".tif")
#     not_cells.append(not_cell)


x_train = np.stack(bright_cells[:800] + f_n_c_s[0:1000] + dark_cells[:1800], axis=0)# half 1 half 0
x_test = np.stack(bright_cells[800:1000] + dark_cells[800:1000], axis=0)
#
y_train = np.zeros(3600, dtype = np.int8)
y_train[:1800] = 1
y_train[1800:] = 0
#
y_test = np.zeros(400, dtype = np.int8)
y_test[:200] = 1
y_test[200:] = 0



# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(6, (3,3), activation='relu', input_shape=(10, 10, 1)),
  # tf.keras.layers.Conv2D(6, (3,3), activation='relu', input_shape=(10, 10, 3)),
  tf.keras.layers.Flatten(),

  # tf.keras.layers.Flatten(input_shape=(10, 10)),

  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# sess = K.get_session()
# print(model.get_layer("conv2d").kernel)

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

# sess = K.get_session()
# print(model.get_layer("conv2d").kernel)

model.save("det.h5")

det_ml = tf.keras.models.load_model("det.h5")

temp_t = time.time()
ret = det_ml.predict(x_test[190:210])
print(ret)
# ret = det_ml.predict(x_test[400:500])
# print(ret)
print("predict time:", time.time() - temp_t)

# print(x_test[0])
# model.evaluate(x_test[0:3])