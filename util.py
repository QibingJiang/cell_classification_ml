#!/usr/bin/env python3
import cv2
import os
import numpy as np
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def read_frame(path, frame_count, data_type, scale, crop_width = 0, crop_height = 0, color = False):
    # global crop_width, crop_height

    if (data_type == 0):#read from raw image data
        video_full_path = path + "input_images/"
        image_path = video_full_path + str(frame_count) + ".npy" #In default case, npy files have been scaled 8 times.
        ret = os.path.exists(image_path)
        if ret != True:
            print("File not exists, ", image_path)
            return False, None

        frame = np.load(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame = frame[0:crop_height * scale, 0:crop_width * scale]
        return True, frame
    elif (data_type == 1 or data_type == 2):  # read from jpg png tiff "/img1/{0:0=6d}".format(frame_index) + ".jpg"
        # image_path = path + "t" + "{0:0=3d}".format(frame_count) + ".tif"
        image_path = path + "{0:0=6d}".format(frame_count) + ".jpg"
        if ((not os.path.exists(image_path))): #  or frame_count > 30
            print("file not exist: ", image_path)
            return False, None
        else:
            # print(frame_count, image_path)
            pass

        # frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)




        if(color == True):
            frame = cv2.imread(image_path)
            if (len(frame.shape) <= 2):
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # x = 619
        # y = 321
        # frame = frame[y:y + crop_height, x:x + crop_width]


        if(crop_width == 0 and crop_height == 0):
            crop_width = frame.shape[1]
            crop_height = frame.shape[0]


        frame = frame[0:crop_height, 0:crop_width]
        if (scale > 1):
            frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

        # frame = frame[154: 625, 748: 1340] # cells
        # frame = frame[154: 384, 748: 1040] # one ell
        # frame = frame[264: 494, 434: 1043] # one cell
        # frame = frame[315: 508, 2345: 2854] # one cell 272
        # frame = frame[1082: 1413, 2209: 2537] # one cell 159

        return True, frame

    # frame = frame[0:crop_height * scale, 0:crop_width * scale]
    # return True, frame
