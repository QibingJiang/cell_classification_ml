import numpy as np
import cv2
import time
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib
import os
from operator import add
from statistics import mean
import scipy as scipy
from scipy.signal import find_peaks
from scipy import optimize
# from astropy.modeling import functional_models as models
from astropy.modeling import models, fitting
import sys
import tensorflow as tf
import shutil
import generate_detections
import re

amount_limit = 790
# amount_limit = 20

line_thick = 8
draw = True
colors = [(255, 255, 0), (255, 0, 255)]

# x3 = self.tracks[i].trace[-1][0]
# y3 = self.tracks[i].trace[-1][1]
# ratio = self.tracks[i].trace[-1][2]
# area = self.tracks[i].trace[-1][3]
# le = int(self.tracks[i].trace[-1][4])
# loc_var = self.tracks[i].trace[-1][5]

new_hist = np.array([0,0,1,1,2,3,3,4,4,5,6,6,7,7,8,9,9,10
,10,11,12,12,13,13,14,15,15,16,16,17,18,18,19,19,20,21
,21,22,22,23,24,24,25,25,26,27,27,28,28,29,30,31,32,34
,35,37,38,39,41,42,44,45,46,48,49,51,52,53,55,56,58,59
,60,62,63,65,66,67,69,70,72,73,74,76,77,79,80,81,83,84
,86,87,88,90,91,93,94,95,97,98,100,101,102,103,105,106,107,108
,110,111,112,114,115,116,117,119,120,121,123,124,125,126,128,129,130,132
,133,134,135,137,138,139,141,142,143,144,146,147,148,150,151,152,153,155
,156,157,158,160,161,162,164,165,166,167,169,170,171,173,174,175,176,178
,179,180,182,183,184,185,187,188,189,191,192,193,194,196,197,198,200,200
,201,202,202,203,204,205,205,206,207,207,208,209,210,210,211,212,212,213
,214,215,215,216,217,217,218,219,220,220,221,222,222,223,224,225,225,226
,227,227,228,229,230,230,231,232,232,233,234,235,235,236,237,237,238,239
,240,240,241,242,242,243,244,245,245,246,247,247,248,249,250,250,251,252
,252,253,254,255], dtype=np.uint8)


def _1gaussian(x, amp1,cen1,sigma1):

    arr = np.array([amp1,cen1,sigma1])
    for data in arr:
        if data < 0:
            return float("inf")

    # return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))
    return amp1 * (np.exp((-1.0 / 2.0) * (((x - cen1) ** 2) / (sigma1 ** 2))))

def _2gaussian(x, amp1,cen1,sigma1, amp2,cen2,sigma2):

    # arr = np.array([amp1,cen1,sigma1, amp2,cen2,sigma2])
    # for data in arr:
    #     if data < 0:
    #         return float("inf")
    #
    # return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + \
    #         amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen2)/sigma2)**2)))

    return _1gaussian(x, amp1, cen1, sigma1) + _1gaussian(x, amp2,cen2,sigma2)

class cell_d:
    def __init__(self):
        self.horizontal_x = 0
        self.vertical_y = 0
        self.area = 0
        self.max_pixel = []
        self.contour = None
        self.radius = 0
        self.frame_idx = 0
        self.score = 0
        self.feature = None
        self.tlwh = None
        self.img = None
        self.fluo = False

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        # ret = self.tlwh.copy()
        # ret[:2] += ret[2:] / 2
        # ret[2] /= ret[3]
        return np.array([self.horizontal_x, self.vertical_y, 1, 18])

class CellDetector(object):

    def __init__(self):
        self.background_pixel = 0
        self.background_pixel_mean = 0
        self.background_pixel_std = 0
        self.cell_core_r = 0
        self.cell_core_r_mean = 0
        self.cell_core_r_std = 0
        self.bg_gau_mean = 0
        self.bg_gau_std = 0
        self.image_amount = 0
        self.edge_thr = 0
        self.core_thr = 0
        self.radius_thr = []
        self.max_pixel = 0
        pass

    def detect_by_edge_core_and_level(self, path, frame, frame_index, scale):
        # print("enter detect_and_level")

        debug = 0
        draw = False

        # debug = 1
        # draw = True

        frame_draw = None

        cell_size = 5

        if (len(frame.shape) > 2):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_draw = frame.copy()
        else:
            gray = frame.copy()
            frame_draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        gray_org = gray.copy()

        if (debug == 1):
            cv2.namedWindow('gray_org', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('gray_org', 900, 900)
            cv2.imshow('gray_org', gray_org)
            # cv2.waitKey()


        centers = []  # vector of object centroids in a frame
        very_white_cell = []

        ##########***** detect black edge *****######################
        if (debug == 2):
            temp_t = time.time()

        # print("black edge thresh: ", self.edge_thr)

        ret, black = cv2.threshold(gray, min(self.edge_thr, 99), 255, cv2.THRESH_BINARY_INV)
        # ret, black = cv2.threshold(gray, 0.95 * self.background_pixel, 255, cv2.THRESH_BINARY_INV)
        # print("black edge thresh: ", 0.95 * self.background_pixel)

        black_white = black

        if (debug == 1):
            cv2.namedWindow('black edge', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black edge', 900, 900)
            cv2.imshow('black edge', black)
            # cv2.waitKey()
            pass

        # t1 = time.time()
        contours, hierarchy = cv2.findContours(black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(time.time() - t1)

        if (debug == 1):
            contours_image = np.zeros_like(gray)
            cv2.drawContours(contours_image, contours, -1, (255, 255, 255), 1)
            cv2.namedWindow('black_edge contours_image2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black_edge contours_image2', 900, 900)
            cv2.imshow('black_edge contours_image2', contours_image)
            # cv2.waitKey()

        frame_0 = np.zeros_like(gray)

        # frame_0

        t1 = time.time()
        black_contour = []
        for i in range(len(contours)):
            try:
                if (hierarchy[0][i][3] == -1 and hierarchy[0][i][2] != -1):#
                    # (x, y), radius = cv2.minEnclosingCircle(contours[i])
                    # if (radius > 3 * scale):
                    # cv2.drawContours(frame_0, contours, i, (255, 255, 255), -1)

                    # loc_0 = np.argmax(contours[i][:, 0][:, 0])
                    # loc_1 = np.argmax(contours[i][:, 0][:, 1])
                    # loc_2 = np.argmin(contours[i][:, 0][:, 0])
                    # loc_3 = np.argmin(contours[i][:, 0][:, 1])
                    # x = contours[i][loc_0][0][0] - contours[i][loc_2][0][0]
                    # y = contours[i][loc_1][0][1] - contours[i][loc_3][0][1]
                    # if(x < (frame.shape[1] >> 1) and y < (frame.shape[0] >> 1)):
                    #     black_contour.append(contours[i])

                    black_contour.append(contours[i])
                    # cv2.namedWindow('black frame', cv2.WINDOW_NORMAL)
                    # cv2.resizeWindow('black frame', 900, 900)
                    # cv2.imshow('black frame', frame_0)
                    # cv2.waitKey()
            except ZeroDivisionError:
                pass

        t1 = time.time()
        cv2.drawContours(frame_0, black_contour, -1, (255, 255, 255), -1)

        if (debug == 1):
            cv2.namedWindow('black frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black frame', 900, 900)
            cv2.imshow('black frame', frame_0)
            # cv2.waitKey()

        if (debug == 2):
            print("detect black edge", time.time() - temp_t)
        ##########***** detect black edge end *****######################


        #####********** detect white core start ********#########
        # print("detect white core start")
        if (debug == 2):
            temp_t = time.time()

        cell_with_edge = []
        cell_with_half_edge = []
        # thresh = self.bg_gau_mean + 3.0 * self.bg_gau_std

        ret, th4 = cv2.threshold(gray, self.core_thr, 255, cv2.THRESH_BINARY)
        # print("cell core thresh: ", self.core_thr)

        if (debug == 1):
            cv2.namedWindow('white_core', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('white_core', 900, 900)
            cv2.imshow('white_core', th4)
            # cv2.waitKey()

        contours, hierarchy = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if (debug == 1):
            contours_image = np.zeros_like(gray)
            cv2.drawContours(contours_image, contours, -1, (255, 255, 255), 1)
            cv2.namedWindow('white point contour', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('white point contour', 900, 900)
            cv2.imshow('white point contour', contours_image)
            # cv2.waitKey()

        fake_cell = 0
        black_new = np.zeros_like(gray)


        for i in range(len(contours)):
            try:
                loc_0 = np.argmax(contours[i][:,0][:,0])
                loc_1 = np.argmax(contours[i][:,0][:,1])
                loc_2 = np.argmin(contours[i][:,0][:,0])
                loc_3 = np.argmin(contours[i][:,0][:,1])

                flag_rm = 0
                for idx in [loc_0, loc_1, loc_2, loc_3]:
                    x = contours[i][idx][0][0]
                    y = contours[i][idx][0][1]
                    if(frame_0[y][x] == 0):
                        flag_rm = 1
                        break
                if(flag_rm == 1):
                    continue


                (x, y), radius = cv2.minEnclosingCircle(contours[i])
                x = int(x)
                y = int(y)
                centeroid = (int(x), int(y))

                if (x > cell_size * scale and x < (frame.shape[1] - cell_size * scale) and y > cell_size * scale and y < (frame.shape[0] - cell_size * scale)):
                    pass
                else:
                    continue

                    # if (radius > (np.max(1.0, self.cell_core_r - 3 * self.cell_core_r_std)) * scale and frame_0[y][x] == 255):  # radius < 2 * scale and gray[y][x] > 125 and and gray[y][x] > 125
                # if (radius > self.noise_radius_thresh * scale and frame_0[y][x] == 255):  # radius < 2 * scale and gray[y][x] > 125 and and gray[y][x] > 125
                # if (self.noise_radius_thresh * scale < radius < (self.cell_core_r + 3 * self.cell_core_r_std) * scale and frame_0[y][x] == 255):
                # if (max(1, self.radius_thr[0]) * scale < radius and frame_0[y][x] == 255): # < self.radius_thr[1] * scale
                if (max(1, self.radius_thr[0]) * scale < radius < 5 * self.cell_core_r * scale and frame_0[y][x] == 255):  # for Pt180 Beacon-32 engineering_code
                    cell = gray_org[(y - cell_size * scale):(y + cell_size * scale + 1), (x - cell_size * scale):(x + cell_size * scale + 1)]
                    black_new[(y - cell_size * scale):(y + cell_size * scale + 1), (x - cell_size * scale):(x + cell_size * scale + 1)] = gray_org[(y - cell_size * scale):(y + cell_size * scale + 1), (x - cell_size * scale):(x + cell_size * scale + 1)]

                    x, y, w, h = cv2.boundingRect(contours[i])
                    rect = gray_org[y:y + h, x:x + w]
                    local_cnt = contours[i]
                    local_cnt = np.array(local_cnt)
                    local_cnt[:, :, 0] = local_cnt[:, :, 0] - x
                    local_cnt[:, :, 1] = local_cnt[:, :, 1] - y
                    my_mask = np.zeros_like(rect)
                    my_mask[:, :] = 255
                    cv2.drawContours(my_mask, [local_cnt], -1, (0, 0, 0), -1)

                    mask_arr = ma.masked_array(rect, mask=my_mask)
                    brightest = ma.max(mask_arr)
                    bright_mean = ma.mean(mask_arr)

                    if (brightest < 150):  # self.max_pixel engineering_code
                        continue

                    retval = cv2.minAreaRect(contours[i])
                    a = max(retval[1][0], retval[1][1])  # long
                    b = min(retval[1][0], retval[1][1])  # short
                    area = cv2.contourArea(contours[i])
                    ratio = b / a
                    eccentricity = np.sqrt(1 - ratio ** 2)

                    ret = (retval[0], (retval[1][0], retval[1][1]), retval[2])
                    box = cv2.boxPoints(ret)
                    box = np.int0(box)

                    new_cell = np.where(cell > 175, cell, 0)
                    cell_sum = np.sum(new_cell)

                    # x3 = self.tracks[i].trace[-1][0]
                    # y3 = self.tracks[i].trace[-1][1]
                    # ratio = self.tracks[i].trace[-1][2]
                    # area = self.tracks[i].trace[-1][3]
                    # le = int(self.tracks[i].trace[-1][4])
                    # loc_var = self.tracks[i].trace[-1][5]

                    if(cell_sum > 100000):
                        level = 0
                        b = np.array([centeroid[0] / scale, centeroid[1] / scale, ratio, area, level, 0, contours[i], box, radius, frame_index, bright_mean, eccentricity], dtype=object)
                        very_white_cell.append(b)
                    else:
                        level = 1
                        b = np.array([centeroid[0] / scale, centeroid[1] / scale, ratio, area, level, 0, contours[i], box, radius, frame_index, bright_mean, eccentricity], dtype=object)
                        cell_with_edge.append(b)

                    if (draw == True):
                        # cv2.circle(frame_draw, centeroid, cell_size * scale, colors[level], int(0.5 * scale))
                        cv2.circle(frame_draw, centeroid, 5 * scale, (255, 255, 0), ((1 * scale) >> 2))
                        # cv2.putText(frame_draw, str(i), (int(x + 65), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # print(i, radius, self.cell_core_r, self.cell_core_r_std, self.noise_radius_thresh)
                        pass

                    # cell_r = 40
                    # one_cell = gray_org[centeroid[1] - cell_r:centeroid[1] + cell_r + 1, centeroid[0] - cell_r:centeroid[0] + cell_r + 1]
                    # cell_img_path = path + "cell_det" + str(frame_index) + ".tif"
                    # cv2.imwrite(cell_img_path, one_cell)
                    #
                    # core = th4[centeroid[1] - cell_r:centeroid[1] + cell_r + 1, centeroid[0] - cell_r:centeroid[0] + cell_r + 1]
                    # cell_img_path = path + "cell_core_det" + str(frame_index) + ".tif"
                    # cv2.imwrite(cell_img_path, core)

                    # cv2.imshow('one_cell', one_cell)
                    # cv2.imshow('core', core)
                    # cv2.waitKey()

                else:
                    if debug == 1:
                        # print('fake object number in the frame:')
                        pass
                    fake_cell = fake_cell + 1

            except ZeroDivisionError:
                pass

        # plt.hist(black_new.flatten(), 256, [0, 256], alpha=0.5, label='Image a')
        # plt.show()

        # cv2.namedWindow('black_new', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('black_new', 900, 900)
        # cv2.imshow('black_new', black_new)
        # cv2.waitKey()

        if (debug == 1):
            cv2.namedWindow('result', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('result', 900, 900)
            cv2.imshow('result', frame_draw)
            cv2.waitKey()

        if (debug == 2):
            print("detect white cores t:", time.time() - temp_t)

        ##########**** detect white cores end ******######################

        # print("cells: ", len(very_white_cell), len(cell_with_edge), len(cell_with_half_edge), len(centers))
        # print(very_white_cell, cell_with_edge, cell_with_half_edge, centers)

        very_white_cell = np.array(very_white_cell)
        cell_with_edge = np.array(cell_with_edge)
        cell_with_half_edge = np.array(cell_with_half_edge)
        centers.append(very_white_cell)
        centers.append(cell_with_edge)
        centers.append(cell_with_half_edge)

        if (draw == True):
            cv2.putText(frame, str(frame_index), (5*scale, 10*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (0, 255, 255), int(0.3 * scale))

        # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('result', 900, 900)
        # cv2.imshow('result', frame)
        # cv2.waitKey()

        # print("detect and level end")
        # cv2.putText(frame_draw, str(frame_index), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (138, 221, 48), 2)
        return frame_draw, centers

    def det_ml(self, path, frame, frame_index, scale, Beacon, f_det_txt, gen_cell = False, fluo = False, fluo_img = None):

        # print("det_ml")

        debug = 0
        draw = False

        # debug = 1
        # draw = True

        # frame_draw = None

        cell_size = 5

        # if (len(frame.shape) > 2):
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     frame_draw = frame.copy()
        # else:
        #     gray = frame.copy()
        #     frame_draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clr_org = frame.copy()
        frame_draw = frame.copy()

        draw_scale = 8
        frame_draw = cv2.resize(frame_draw, (frame_draw.shape[1] * draw_scale , frame_draw.shape[0] * draw_scale), interpolation=cv2.INTER_CUBIC)
        gray_org = gray.copy()
        clr_org_8 = frame_draw.copy()

        if (debug == 1):
            cv2.namedWindow('gray_org', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('gray_org', 900, 900)
            cv2.imshow('gray_org', gray_org)
            # cv2.waitKey()

        centers = []  # vector of object centroids in a frame
        very_white_cell = []

        ##########***** detect black edge *****######################

        # ret, black = cv2.threshold(gray, min(self.edge_thr, 99), 255, cv2.THRESH_BINARY_INV)
        ret, black = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        # ret, black = cv2.threshold(gray, 0.95 * self.background_pixel, 255, cv2.THRESH_BINARY_INV)
        # print("black edge thresh: ", 0.95 * self.background_pixel)

        black_white = black

        if (debug == 1):
            cv2.namedWindow('black edge', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black edge', 900, 900)
            cv2.imshow('black edge', black)
            # cv2.waitKey()
            pass

        # t1 = time.time()
        contours, hierarchy = cv2.findContours(black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print("num of contours: ", len(contours))

        # contours, hierarchy = cv2.findContours(black_white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print(time.time() - t1)

        # contours, _ = cv2.findContours(image=np.array(gray, dtype=np.int32), mode=cv2.RETR_FLOODFILL, method=cv2.CHAIN_APPROX_SIMPLE)

        if (debug == 1):
            contours_image = np.zeros_like(gray)
            cv2.drawContours(contours_image, contours, -1, (255, 255, 255), 1)
            cv2.namedWindow('black_edge contours_image2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black_edge contours_image2', 900, 900)
            cv2.imshow('black_edge contours_image2', contours_image)
            # cv2.waitKey()

        cells = []
        temp_t = time.time()
        cell_l = [[], [], []]
        cell_info_s = []
        rows = []

        # print("len(contours): ", len(contours))
        for i in range(len(contours)):
            try:
                t1 = time.time()
                # if(len(contours[i]) < 4):
                #     # print(contours[i])
                #     continue
                (f_x, f_y), radius = cv2.minEnclosingCircle(contours[i])
                # cell_info_s.append([[f_x, f_y], radius])
                x = round(f_x)
                y = round(f_y)
                t1 = time.time() - t1

                # Rect_x, Rect_y, Rect_w, Rect_h = cv2.boundingRect(contours[i])

                cr = 6
                cell_r = 9


                if (x - cell_r > 0 and x + cell_r < gray.shape[1] and y - cell_r > 0 and y + cell_r < gray.shape[0]):# 2 < Rect_w < 8 and 2 < Rect_h < 8

                    cell_2 = clr_org[y - cr:y + cr, x - cr:x + cr][:, :, 2:3]

                    Rect_x, Rect_y, Rect_w, Rect_h = cv2.boundingRect(contours[i])
                    #
                    # print(contours[i])
                    # print(Rect_x, Rect_y, Rect_w, Rect_h)
                    # print(x - cr, y - cr)

                    rect = gray_org[Rect_y:Rect_y + Rect_h, Rect_x:Rect_x + Rect_w]
                    cell_2_gray = gray_org[Rect_y:Rect_y + Rect_h, Rect_x:Rect_x + Rect_w]

                    local_cnt = contours[i]
                    local_cnt = np.array(local_cnt)
                    local_cnt[:, :, 0] = local_cnt[:, :, 0] - Rect_x
                    local_cnt[:, :, 1] = local_cnt[:, :, 1] - Rect_y
                    my_mask = np.zeros_like(rect)
                    my_mask[:, :] = 255
                    cv2.drawContours(my_mask, [local_cnt], -1, (0, 0, 0), -1)

                    mask_arr = ma.masked_array(rect, mask=my_mask)
                    brightest = ma.max(mask_arr)
                    bright_mean = ma.mean(mask_arr)

                    if(brightest != cell_2_gray.max()):
                        # print("max disagree")
                        continue

                    cell_l[0].append(i)
                    cell_l[1].append([[f_x, f_y], radius, brightest])
                    cell_l[2].append(cell_2)

                    cell_r = 9
                    rows.append([x - cell_r, y - cell_r, cell_r * 2, cell_r * 2])

                    pass
                else:
                    # print("not cell")
                    pass
            except ZeroDivisionError:
                pass

        det_ml = tf.keras.models.load_model("./det_train/det.h5")
        encoder = generate_detections.create_box_encoder("./deep_sort_2/resources/networks/cells.pb", batch_size=32)

        det_score = det_ml.predict(np.array(cell_l[2]))
        # print(len(det_score), det_score)

        # det_score_2 = [np.argmax(d) for d in det_score]

        det_score_2 = []

        for d in det_score:
            loc = np.argmax(d)
            if(d[loc] > 0.0):
                det_score_2.append(loc)
            else:
                det_score_2.append(-1)
                # print("The white object is nothing.")

        features = encoder(clr_org, rows)

        if(gen_cell == True):
            cells_path = path + "ML/" + "Beacon_" + str(Beacon) + "_cells_train/" + str(frame_index) + "/"
            # false_negtive = path + "ML/" + "Beacon_" + str(Beacon) + "_cells_train/false_negtive/"
            # if(os.path.exists(cells_path)):
            #     os.rmdir(cells_path)
            # os.remove()


            try:
                if(frame_index == 0 and os.path.exists(path + "ML/" + "Beacon_" + str(Beacon) + "_cells_train/")):
                    shutil.rmtree(path + "ML/" + "Beacon_" + str(Beacon) + "_cells_train/")
            except OSError as e:
                print("Error: %s : %s" % (cells_path, e.strerror))
                pass

            os.makedirs(cells_path, exist_ok=True)
            # os.makedirs(false_negtive, exist_ok=True)

        for s_i in range(len(det_score_2)):
            try:
                if(det_score_2[s_i] == 0 and cell_l[1][s_i][2] < 160):
                # if(det_score_2[s_i] == 0 and cell_l[1][s_i][2] < 110): #  and cell_l[1][s_i][2] < 160 # /data/qibing/Pt935_MACROPHAGE_plate1_08102022/RawData/Beacon-92/
                    continue

                # if(det_score_2[s_i] == 0 and cell_l[1][s_i][2] < 135):# I just temporarily use this one for Pt935 Beacon 337-341
                #     continue


                c_i = cell_l[0][s_i]
                level = 0
                area = cv2.contourArea(contours[c_i])

                b = np.array([cell_l[1][s_i][0][0], cell_l[1][s_i][0][1], 1, area, level, 0, contours[c_i], ["box"],
                              cell_l[1][s_i][1], frame_index, 230, 1, features[s_i]], dtype=object)
                c = cell_d()
                c.horizontal_x = cell_l[1][s_i][0][0]
                c.vertical_y = cell_l[1][s_i][0][1]
                c.area = area
                c.contour = contours[c_i]
                c.radius = cell_l[1][s_i][1]
                c.max_pixel = cell_l[1][s_i][2]
                c.frame_idx = frame_index
                c.feature = features[s_i]
                c.tlwh = np.array([c.vertical_y - 9, c.horizontal_x - 9, 18, 18])
                # c.score = det_score[s_i][1]
                c.score = det_score_2[s_i]

                x = int(c.horizontal_x)
                y = int(c.vertical_y)
                # b = clr_org[y][x][0]
                # g = clr_org[y][x][1]
                # r = clr_org[y][x][2]

                # if (b - max(g, r) > 3):
                #     c.fluo = True

                if (fluo and len(fluo_img) > 0 and fluo_img[y][x] > 25): # and frame_index > 96
                    c.fluo = True

                cells.append(c)

                # if(s_i == 69):
                #     print("s_i == 69 max: ", c.max_pixel, c.horizontal_x, c.vertical_y)


                f_x = cell_l[1][s_i][0][0]
                f_y = cell_l[1][s_i][0][1]

                cell_radius = 6
                x3 = f_x
                y3 = f_y

                # if (gen_cell == True):
                scale_cell = 1
                one_cell = gray_org[round((y3 - cell_radius) * scale_cell):round((y3 + cell_radius) * scale_cell),
                           round((x3 - cell_radius) * scale_cell):round((x3 + cell_radius) * scale_cell)]

                scale_cell = 8
                one_big_cell = clr_org_8[round((y3 - cell_radius) * scale_cell):round((y3 + cell_radius) * scale_cell),
                           round((x3 - cell_radius) * scale_cell):round((x3 + cell_radius) * scale_cell)]

                c.img = one_big_cell.copy()
                # cv2.imshow("cell", one_cell)
                # cv2.imshow("big cell", one_big_cell)
                # cv2.waitKey()

                if (gen_cell == True and one_cell.max() > 150):
                    cell_img_path = cells_path + "{0:0=4d}".format(frame_index) + "_" + str(s_i) + ".tif"
                    cv2.imwrite(cell_img_path, one_cell)

                    # print(x3, y3, round((x3 - cell_radius) * scale_cell), round((x3 + cell_radius) * scale_cell))
                    # cv2.imshow("gray", gray_org)
                    # cv2.imshow("img", one_cell)
                    # cv2.waitKey()

                if (det_score_2[s_i] == 2):

                    # t3 = time.time()
                    # c_i = cell_l[0][s_i]
                    # level = 0
                    # area = cv2.contourArea(contours[c_i])
                    # b = np.array([cell_l[1][s_i][0][0], cell_l[1][s_i][0][1], 1, area, level, 0, contours[c_i], ["box"], cell_l[1][s_i][1], frame_index, 230, 1], dtype=object)
                    # cells.append(b)

                    f_x = cell_l[1][s_i][0][0]
                    f_y = cell_l[1][s_i][0][1]

                    cell_radius = 5
                    x3 = f_x
                    y3 = f_y



                    # if(gen_cell == True):
                    #     scale_cell = 1
                    #     one_cell = gray_org[int((y3 - cell_radius) * scale_cell):int((y3 + cell_radius) * scale_cell),
                    #                int((x3 - cell_radius) * scale_cell):int((x3 + cell_radius) * scale_cell)]
                    #     cell_img_path = cells_path + "{0:0=4d}".format(frame_index) + "_" + str(s_i) + ".tif"
                    #     cv2.imwrite(cell_img_path, one_cell)

                    cell_r = 9
                    print(frame_index, -1, x3 - cell_r, y3 - cell_r, cell_r * 2, cell_r * 2, 1, -1, -1, -1, file=f_det_txt, sep=',')

                    if (draw == True):
                        cv2.circle(frame_draw, (round(f_x * draw_scale), round(f_y * draw_scale)), 5 * draw_scale, (255, 255, 0),
                                   ((1 * draw_scale) >> 2))
                        cv2.putText(frame_draw, str(s_i) + "_" + str(det_score_2[s_i]),
                                    (round(f_x + 8) * draw_scale, round(f_y + 2.5) * draw_scale), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.125 * draw_scale, (255, 255, 0), max(round(0.25 * draw_scale), 1))

                elif(det_score_2[s_i] == 1):

                    f_x = cell_l[1][s_i][0][0]
                    f_y = cell_l[1][s_i][0][1]

                    cell_radius = 6
                    x3 = f_x
                    y3 = f_y

                    # if(gen_cell == True):
                    #     scale_cell = 1
                    #     one_cell = gray_org[round((y3 - cell_radius) * scale_cell):round((y3 + cell_radius) * scale_cell),
                    #                round((x3 - cell_radius) * scale_cell):round((x3 + cell_radius) * scale_cell)]
                    #
                    #     if(one_cell.max() > 150):
                    #         cell_img_path = cells_path + "{0:0=4d}".format(frame_index) + "_" + str(s_i) + ".tif"
                    #         cv2.imwrite(cell_img_path, one_cell)

                    if (draw == True):
                        cv2.circle(frame_draw, (round(f_x * draw_scale), round(f_y * draw_scale)), 5 * draw_scale, (255, 0, 255),
                                   ((1 * draw_scale) >> 2))
                        cv2.putText(frame_draw, str(s_i) + "_" + str(det_score_2[s_i]),
                                    (round(f_x + 8) * draw_scale, round(f_y + 2.5) * draw_scale), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.125 * draw_scale, (255, 0, 255), max(int(0.25 * draw_scale), 1))
                else:

                    f_x = cell_l[1][s_i][0][0]
                    f_y = cell_l[1][s_i][0][1]

                    cell_radius = 6
                    x3 = f_x
                    y3 = f_y

                    # if(gen_cell == True):
                    #     scale_cell = 1
                    #     one_cell = gray_org[round((y3 - cell_radius) * scale_cell):round((y3 + cell_radius) * scale_cell),
                    #                round((x3 - cell_radius) * scale_cell):round((x3 + cell_radius) * scale_cell)]
                    #
                    #     if(one_cell.max() > 150):
                    #         cell_img_path = cells_path + "{0:0=4d}".format(frame_index) + "_" + str(s_i) + ".tif"
                    #         cv2.imwrite(cell_img_path, one_cell)

                    if (draw == True):
                        cv2.circle(frame_draw, (round(f_x * draw_scale), round(f_y * draw_scale)), 5 * draw_scale, (255, 255, 255),
                                   ((1 * draw_scale) >> 2))
                        cv2.putText(frame_draw, str(s_i) + "_" + str(det_score_2[s_i]),
                                    (round(f_x + 8) * draw_scale, round(f_y + 2.5) * draw_scale), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.125 * draw_scale, (255, 255, 255), max(int(0.25 * draw_scale), 1))

                    pass

                pass
            except ZeroDivisionError:
                pass

        cv2.putText(frame_draw, str(frame_index), (5*draw_scale, 10*draw_scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * draw_scale, (0, 255, 255), round(0.3 * draw_scale))

        # print("one loop:", time.time() - temp_t)

        if (debug == 1):
            cv2.namedWindow('frame_draw', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame_draw', 900, 900)
            cv2.imshow('frame_draw', frame_draw)
            cv2.waitKey()

        centers = []
        centers.append(cells)
        return frame_draw, centers

    #detect black edge first then white edge
    def det_out_focus(self, path, frame, frame_index, scale, Beacon, f_det_txt):
        # print("enter detect_and_level")

        debug = 0
        draw = False

        debug = 0
        draw = True

        frame_draw = None

        cell_size = 5

        # gen_cell = False
        gen_cell = True

        if (len(frame.shape) > 2):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_draw = frame.copy()
        else:
            gray = frame.copy()
            frame_draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        clr_org = frame_draw.copy()

        draw_scale = 8
        frame_draw = cv2.resize(frame_draw, (frame_draw.shape[1] * draw_scale , frame_draw.shape[0] * draw_scale), interpolation=cv2.INTER_CUBIC)
        gray_org = gray.copy()
        clr_org_8 = frame_draw.copy()

        if (debug == 1):
            cv2.namedWindow('gray_org', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('gray_org', 900, 900)
            cv2.imshow('gray_org', gray_org)
            # cv2.waitKey()

        centers = []  # vector of object centroids in a frame
        very_white_cell = []

        ##########***** detect black edge *****######################
        if (debug == 2):
            temp_t = time.time()

        # print("black edge thresh: ", self.edge_thr)

        # ret, black = cv2.threshold(gray, min(self.edge_thr, 99), 255, cv2.THRESH_BINARY_INV)
        ret, black = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
        # ret, black = cv2.threshold(gray, 0.95 * self.background_pixel, 255, cv2.THRESH_BINARY_INV)
        # print("black edge thresh: ", 0.95 * self.background_pixel)

        # num_labels, labels = cv2.connectedComponents(black)

        black_white = black

        if (debug == 1):
            cv2.namedWindow('black edge', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black edge', 900, 900)
            cv2.imshow('black edge', black)
            # cv2.waitKey()
            pass

        # t1 = time.time()
        contours, hierarchy = cv2.findContours(black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print("num of contours: ", len(contours))

        # contours, hierarchy = cv2.findContours(black_white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print(time.time() - t1)

        # contours, _ = cv2.findContours(image=np.array(gray, dtype=np.int32), mode=cv2.RETR_FLOODFILL, method=cv2.CHAIN_APPROX_SIMPLE)

        if (debug == 1):
            contours_image = np.zeros_like(gray)
            cv2.drawContours(contours_image, contours, -1, (255, 255, 255), 1)
            cv2.namedWindow('black_edge contours_image2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black_edge contours_image2', 900, 900)
            cv2.imshow('black_edge contours_image2', contours_image)
            # cv2.waitKey()

        frame_0 = np.zeros_like(gray)

        # frame_0

        t1 = time.time()
        black_contour = []

        cells = []
        temp_t = time.time()
        cell_l = [[], [], []]
        cell_info_s = []
        rows = []

        for i in range(len(contours)):
            try:
                t1 = time.time()
                # if(len(contours[i]) < 4):
                #     # print(contours[i])
                #     continue

                if(hierarchy[0][i][3] > -1):
                    continue

                (f_x, f_y), radius = cv2.minEnclosingCircle(contours[i])
                # cell_info_s.append([[f_x, f_y], radius])
                x = round(f_x)
                y = round(f_y)
                t1 = time.time() - t1

                Rect_x, Rect_y, Rect_w, Rect_h = cv2.boundingRect(contours[i])

                cr = 6
                cell_r = 9


                if (x - cell_r > 0 and x + cell_r < gray.shape[1] and y - cell_r > 0 and y + cell_r < gray.shape[0]):# 2 < Rect_w < 8 and 2 < Rect_h < 8
                    t2 = time.time()

                    # cell = gray[y - 5:y + 5, x - 5:x + 5]
                    # cell_2 = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)[:, :, 0:1]

                    cell_2 = clr_org[y - cr:y + cr, x - cr:x + cr][:, :, 0:1]

                    Rect_x, Rect_y, Rect_w, Rect_h = cv2.boundingRect(contours[i])
                    rect = gray_org[Rect_y:Rect_y + Rect_h, Rect_x:Rect_x + Rect_w]
                    local_cnt = contours[i]
                    local_cnt = np.array(local_cnt)
                    local_cnt[:, :, 0] = local_cnt[:, :, 0] - Rect_x
                    local_cnt[:, :, 1] = local_cnt[:, :, 1] - Rect_y
                    my_mask = np.zeros_like(rect)
                    my_mask[:, :] = 255
                    cv2.drawContours(my_mask, [local_cnt], -1, (0, 0, 0), -1)

                    mask_arr = ma.masked_array(rect, mask=my_mask)
                    brightest = ma.max(mask_arr)
                    bright_mean = ma.mean(mask_arr)

                    # if(brightest != cell_2.max()):
                    #     continue

                    cell_l[0].append(i)
                    cell_l[1].append([[f_x, f_y], radius, brightest])
                    cell_l[2].append(cell_2)

                    cell_r = 9
                    rows.append([x - cell_r, y - cell_r, cell_r * 2, cell_r * 2])

                    pass
            except ZeroDivisionError:
                pass

        gray_org_2 = gray_org.copy()
        draw_scale = 1
        for c in cell_l[1]:
            f_x, f_y = c[0]
            if(c[1] < 4):
                cv2.circle(gray_org_2, (round(f_x * draw_scale), round(f_y * draw_scale)), 3 * draw_scale, (0, 0, 0), -1)

        draw_scale = 8

        # debug = 1
        # if (debug == 1):
        #     cv2.namedWindow('gray_org_2', cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow('gray_org_2', 900, 900)
        #     cv2.imshow('gray_org_2', gray_org_2)
        #     cv2.waitKey()

        ret_frame, ret_centers = self.det_ml(path, gray_org_2, frame_index, scale, Beacon, f_det_txt)


        det_ml = tf.keras.models.load_model("./det_train/det.h5")
        encoder = generate_detections.create_box_encoder("./deep_sort_2/resources/networks/cells.pb", batch_size=32)

        det_score = det_ml.predict(np.array(cell_l[2]))

        # det_score_2 = [np.argmax(d) for d in det_score]

        det_score_2 = []
        for d in det_score:
            loc = np.argmax(d)
            if(d[loc] > 0.9):
                det_score_2.append(loc)
            else:
                det_score_2.append(-1)
                # print("The white object is nothing.")

        features = encoder(clr_org, rows)

        if(gen_cell == True):
            cells_path = path + "ML/" + "Beacon_" + str(Beacon) + "_cells_train/" + str(frame_index) + "/"
            # false_negtive = path + "ML/" + "Beacon_" + str(Beacon) + "_cells_train/false_negtive/"
            # if(os.path.exists(cells_path)):
            #     os.rmdir(cells_path)
            # os.remove()


            try:
                if(frame_index == 0 and os.path.exists(path + "ML/" + "Beacon_" + str(Beacon) + "_cells_train/")):
                    shutil.rmtree(path + "ML/" + "Beacon_" + str(Beacon) + "_cells_train/")
            except OSError as e:
                print("Error: %s : %s" % (cells_path, e.strerror))
                pass

            os.makedirs(cells_path, exist_ok=True)
            # os.makedirs(false_negtive, exist_ok=True)

        for s_i in range(len(det_score_2)):
            try:
                c_i = cell_l[0][s_i]
                level = 0
                area = cv2.contourArea(contours[c_i])

                b = np.array([cell_l[1][s_i][0][0], cell_l[1][s_i][0][1], 1, area, level, 0, contours[c_i], ["box"],
                              cell_l[1][s_i][1], frame_index, 230, 1, features[s_i]], dtype=object)
                c = cell_d()
                c.horizontal_x = cell_l[1][s_i][0][0]
                c.vertical_y = cell_l[1][s_i][0][1]
                c.area = area
                c.contour = contours[c_i]
                c.radius = cell_l[1][s_i][1]
                c.max_pixel = cell_l[1][s_i][2]
                c.frame_idx = frame_index
                c.feature = features[s_i]
                c.tlwh = np.array([c.vertical_y - 9, c.horizontal_x - 9, 18, 18])
                # c.score = det_score[s_i][1]
                c.score = det_score_2[s_i]

                cells.append(c)

                # if(s_i == 69):
                #     print("s_i == 69 max: ", c.max_pixel, c.horizontal_x, c.vertical_y)


                f_x = cell_l[1][s_i][0][0]
                f_y = cell_l[1][s_i][0][1]

                cell_radius = 6
                x3 = f_x
                y3 = f_y

                if (gen_cell == True):
                    scale_cell = 1
                    one_cell = gray_org[round((y3 - cell_radius) * scale_cell):round((y3 + cell_radius) * scale_cell),
                               round((x3 - cell_radius) * scale_cell):round((x3 + cell_radius) * scale_cell)]

                    scale_cell = 8
                    one_big_cell = clr_org_8[round((y3 - cell_radius) * scale_cell):round((y3 + cell_radius) * scale_cell),
                               round((x3 - cell_radius) * scale_cell):round((x3 + cell_radius) * scale_cell)]

                    c.img = one_big_cell.copy()
                    # cv2.imshow("cell", one_cell)
                    # cv2.imshow("big cell", one_big_cell)
                    # cv2.waitKey()

                    if (True or one_cell.max() > 150):
                        cell_img_path = cells_path + "{0:0=4d}".format(frame_index) + "_" + str(s_i) + ".tif"
                        cv2.imwrite(cell_img_path, one_cell)

                if (det_score_2[s_i] == 2):

                    f_x = cell_l[1][s_i][0][0]
                    f_y = cell_l[1][s_i][0][1]

                    cell_radius = 5
                    x3 = f_x
                    y3 = f_y

                    cell_r = 9
                    print(frame_index, -1, x3 - cell_r, y3 - cell_r, cell_r * 2, cell_r * 2, 1, -1, -1, -1, file=f_det_txt, sep=',')

                    if (draw == True):
                        cv2.circle(frame_draw, (round(f_x * draw_scale), round(f_y * draw_scale)), 5 * draw_scale, (255, 255, 0),
                                   ((1 * draw_scale) >> 2))
                        cv2.putText(frame_draw, str(s_i) + "_" + str(det_score_2[s_i]),
                                    (round(f_x + 8) * draw_scale, round(f_y + 2.5) * draw_scale), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.125 * draw_scale, (255, 255, 0), max(round(0.25 * draw_scale), 1))

                elif(det_score_2[s_i] == 1):

                    f_x = cell_l[1][s_i][0][0]
                    f_y = cell_l[1][s_i][0][1]

                    if (draw == True):
                        cv2.circle(frame_draw, (round(f_x * draw_scale), round(f_y * draw_scale)), 5 * draw_scale, (255, 0, 255),
                                   ((1 * draw_scale) >> 2))
                        cv2.putText(frame_draw, str(s_i) + "_" + str(det_score_2[s_i]),
                                    (round(f_x + 8) * draw_scale, round(f_y + 2.5) * draw_scale), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.125 * draw_scale, (255, 0, 255), max(int(0.25 * draw_scale), 1))
                else:

                    f_x = cell_l[1][s_i][0][0]
                    f_y = cell_l[1][s_i][0][1]

                    if (draw == True):
                        cv2.circle(frame_draw, (round(f_x * draw_scale), round(f_y * draw_scale)), 5 * draw_scale, (255, 255, 255),
                                   ((1 * draw_scale) >> 2))
                        cv2.putText(frame_draw, str(s_i) + "_" + str(det_score_2[s_i]),
                                    (round(f_x + 8) * draw_scale, round(f_y + 2.5) * draw_scale), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.125 * draw_scale, (255, 255, 255), max(int(0.25 * draw_scale), 1))

                    pass

                pass
            except ZeroDivisionError:
                pass

        cv2.putText(frame_draw, str(frame_index), (5*draw_scale, 10*draw_scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * draw_scale, (0, 255, 255), round(0.3 * draw_scale))

        # print("one loop:", time.time() - temp_t)

        debug = 0
        if (debug == 1):
            cv2.namedWindow('frame_draw', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame_draw', 900, 900)
            cv2.imshow('frame_draw', frame_draw)
            cv2.waitKey()

        centers = []
        centers.append(cells + ret_centers[0])
        # for()
        print(cells)
        for c in ret_centers[0]:
            f_x = c.horizontal_x
            f_y = c.vertical_y
            cv2.circle(frame_draw, (round(f_x * draw_scale), round(f_y * draw_scale)), 5 * draw_scale, (0, 255, 255),
                       ((1 * draw_scale) >> 2))
            # cv2.putText(frame_draw, str(s_i) + "_" + str(det_score_2[s_i]),
            #             (round(f_x + 8) * draw_scale, round(f_y + 2.5) * draw_scale), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.125 * draw_scale, (255, 255, 255), max(int(0.25 * draw_scale), 1))

        debug = 0
        if (debug == 1):
            cv2.namedWindow('frame_draw', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame_draw', 900, 900)
            cv2.imshow('frame_draw', frame_draw)
            cv2.waitKey()


        return frame_draw, centers

    #detect black edge. Call det_ml before this function to detect white core.
    def det_out_focus_2(self, path, frame, frame_index, scale, Beacon, f_det_txt, masked, gen_cell = False):
        # print("enter detect_and_level")

        debug = 0
        draw = False

        debug = 0
        draw = True

        frame_draw = None

        cell_size = 5

        # if (len(frame.shape) > 2):
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     frame_draw = frame.copy()
        # else:
        #     gray = frame.copy()
        #     frame_draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # clr_org = frame_draw.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clr_org = frame.copy()
        frame_draw = frame.copy()

        draw_scale = 8
        frame_draw = cv2.resize(frame_draw, (frame_draw.shape[1] * draw_scale , frame_draw.shape[0] * draw_scale), interpolation=cv2.INTER_CUBIC)
        gray_org = gray.copy()
        clr_org_8 = frame_draw.copy()

        if (debug == 1):
            cv2.namedWindow('gray_org', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('gray_org', 900, 900)
            cv2.imshow('gray_org', gray_org)
            # cv2.waitKey()

        centers = []  # vector of object centroids in a frame
        very_white_cell = []

        ##########***** detect black edge *****######################
        if (debug == 2):
            temp_t = time.time()

        # print("black edge thresh: ", self.edge_thr)

        # ret, black = cv2.threshold(gray, min(self.edge_thr, 99), 255, cv2.THRESH_BINARY_INV)
        ret, black = cv2.threshold(masked, 80, 255, cv2.THRESH_BINARY_INV)
        # ret, black = cv2.threshold(gray, 0.95 * self.background_pixel, 255, cv2.THRESH_BINARY_INV)
        # print("black edge thresh: ", 0.95 * self.background_pixel)

        # num_labels, labels = cv2.connectedComponents(black)

        black_white = black

        if (debug == 1):
            cv2.namedWindow('black edge', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black edge', 900, 900)
            cv2.imshow('black edge', black)
            # cv2.waitKey()
            pass

        # t1 = time.time()
        contours, hierarchy = cv2.findContours(black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print("num of contours: ", len(contours))

        # contours, hierarchy = cv2.findContours(black_white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print(time.time() - t1)

        # contours, _ = cv2.findContours(image=np.array(gray, dtype=np.int32), mode=cv2.RETR_FLOODFILL, method=cv2.CHAIN_APPROX_SIMPLE)

        if (debug == 1):
            contours_image = np.zeros_like(gray)
            cv2.drawContours(contours_image, contours, -1, (255, 255, 255), 1)
            cv2.namedWindow('black_edge contours_image2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black_edge contours_image2', 900, 900)
            cv2.imshow('black_edge contours_image2', contours_image)
            # cv2.waitKey()

        frame_0 = np.zeros_like(gray)

        # frame_0

        t1 = time.time()
        black_contour = []

        cells = []
        temp_t = time.time()
        cell_l = [[], [], []]
        cell_info_s = []
        rows = []

        for i in range(len(contours)):
            try:
                # if(i == 1136):
                #     print(i)
                t1 = time.time()
                # if(len(contours[i]) < 4):
                #     # print(contours[i])
                #     continue

                if(hierarchy[0][i][3] > -1):
                    continue

                (f_x, f_y), radius = cv2.minEnclosingCircle(contours[i])

                self.cell_core_r = 2.88
                self.cell_core_r_std = 0.46

                # maybe problem is here
                if(radius > 5 or radius < self.cell_core_r - self.cell_core_r_std * 3):
                    continue

                # cell_info_s.append([[f_x, f_y], radius])
                x = round(f_x)
                y = round(f_y)
                t1 = time.time() - t1

                Rect_x, Rect_y, Rect_w, Rect_h = cv2.boundingRect(contours[i])

                cr = 6
                cell_r = 9



                if (x - cell_r > 0 and x + cell_r < gray.shape[1] and y - cell_r > 0 and y + cell_r < gray.shape[0]):# 2 < Rect_w < 8 and 2 < Rect_h < 8


                    retval = cv2.minAreaRect(contours[i])
                    a = max(retval[1][0], retval[1][1])  # long
                    b = min(retval[1][0], retval[1][1])  # short
                    ratio = b / a
                    eccentricity = np.sqrt(1 - ratio ** 2)

                    # if(eccentricity > 0.5):
                    #     continue



                    t2 = time.time()

                    # cell = gray[y - 5:y + 5, x - 5:x + 5]
                    # cell_2 = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)[:, :, 0:1]

                    cell_2 = clr_org[y - cr:y + cr, x - cr:x + cr][:, :, 0:1]

                    Rect_x, Rect_y, Rect_w, Rect_h = cv2.boundingRect(contours[i])
                    rect = gray_org[Rect_y:Rect_y + Rect_h, Rect_x:Rect_x + Rect_w]
                    local_cnt = contours[i]
                    local_cnt = np.array(local_cnt)
                    local_cnt[:, :, 0] = local_cnt[:, :, 0] - Rect_x
                    local_cnt[:, :, 1] = local_cnt[:, :, 1] - Rect_y
                    my_mask = np.zeros_like(rect)
                    my_mask[:, :] = 255
                    cv2.drawContours(my_mask, [local_cnt], -1, (0, 0, 0), -1)

                    mask_arr = ma.masked_array(rect, mask=my_mask)
                    brightest = ma.max(mask_arr)
                    bright_mean = ma.mean(mask_arr)

                    # if(brightest != cell_2.max()):
                    #     continue

                    cell_l[0].append(i)
                    cell_l[1].append([[f_x, f_y], radius, brightest])
                    cell_l[2].append(cell_2)

                    cell_r = 9
                    rows.append([x - cell_r, y - cell_r, cell_r * 2, cell_r * 2])

                    pass
            except ZeroDivisionError:
                pass

        # gray_org_2 = gray_org.copy()
        # draw_scale = 1
        # for c in cell_l[1]:
        #     f_x, f_y = c[0]
        #     if(c[1] < 4):
        #         cv2.circle(gray_org_2, (round(f_x * draw_scale), round(f_y * draw_scale)), 3 * draw_scale, (0, 0, 0), -1)
        # draw_scale = 8


        # ret_frame, ret_centers = self.det_ml(path, gray_org_2, frame_index, scale, Beacon, f_det_txt)

        det_ml = tf.keras.models.load_model("./det_train/det.h5")
        encoder = generate_detections.create_box_encoder("./deep_sort_2/resources/networks/cells.pb", batch_size=32)

        if(len(cell_l[2]) > 0):
            det_score = det_ml.predict(np.array(cell_l[2]))
        else:
            det_score = []

        # det_score_2 = [np.argmax(d) for d in det_score]

        det_score_2 = []
        for d in det_score:
            loc = np.argmax(d)
            if(d[loc] > 0.9):
                det_score_2.append(loc)
            else:
                det_score_2.append(-1)
                # print("The white object is nothing.")

        features = encoder(clr_org, rows)

        if(gen_cell == True):
            cells_path = path + "ML/" + "Beacon_" + str(Beacon) + "_cells_train/" + str(frame_index) + "/"
            # false_negtive = path + "ML/" + "Beacon_" + str(Beacon) + "_cells_train/false_negtive/"
            # if(os.path.exists(cells_path)):
            #     os.rmdir(cells_path)
            # os.remove()


            try:
                if(frame_index == 0 and os.path.exists(path + "ML/" + "Beacon_" + str(Beacon) + "_cells_train/")):
                    shutil.rmtree(path + "ML/" + "Beacon_" + str(Beacon) + "_cells_train/")
            except OSError as e:
                print("Error: %s : %s" % (cells_path, e.strerror))
                pass

            os.makedirs(cells_path, exist_ok=True)
            # os.makedirs(false_negtive, exist_ok=True)

        # det_score_2[603] = 1

        for s_i in range(len(det_score_2)):
            try:
                c_i = cell_l[0][s_i]
                level = 0
                area = cv2.contourArea(contours[c_i])

                # if (s_i == 422):
                #     print(contours[c_i], hierarchy[0][c_i])

                b = np.array([cell_l[1][s_i][0][0], cell_l[1][s_i][0][1], 1, area, level, 0, contours[c_i], ["box"],
                              cell_l[1][s_i][1], frame_index, 230, 1, features[s_i]], dtype=object)
                c = cell_d()
                c.horizontal_x = cell_l[1][s_i][0][0]
                c.vertical_y = cell_l[1][s_i][0][1]
                c.area = area
                c.contour = contours[c_i]
                c.radius = cell_l[1][s_i][1]
                c.max_pixel = cell_l[1][s_i][2]
                c.frame_idx = frame_index
                c.feature = features[s_i]
                c.tlwh = np.array([c.vertical_y - 9, c.horizontal_x - 9, 18, 18])
                # c.score = det_score[s_i][1]
                c.score = det_score_2[s_i]

                x = int(c.horizontal_x)
                y = int(c.vertical_y)

                b = clr_org[y][x][0]
                g = clr_org[y][x][1]
                r = clr_org[y][x][2]

                if (b - max(g, r) > 3):
                    c.fluo = True

                cells.append(c)


                f_x = cell_l[1][s_i][0][0]
                f_y = cell_l[1][s_i][0][1]

                cell_radius = 6
                x3 = f_x
                y3 = f_y

                scale_cell = 8
                one_big_cell = clr_org_8[round((y3 - cell_radius) * scale_cell):round((y3 + cell_radius) * scale_cell),
                           round((x3 - cell_radius) * scale_cell):round((x3 + cell_radius) * scale_cell)]

                c.img = one_big_cell.copy()

                if (gen_cell == True):
                    scale_cell = 1
                    one_cell = gray_org[round((y3 - cell_radius) * scale_cell):round((y3 + cell_radius) * scale_cell),
                               round((x3 - cell_radius) * scale_cell):round((x3 + cell_radius) * scale_cell)]

                    if (True or one_cell.max() > 150):
                        cell_img_path = cells_path + "{0:0=4d}".format(frame_index) + "_" + str(s_i) + ".tif"
                        cv2.imwrite(cell_img_path, one_cell)

                if (det_score_2[s_i] >= 2):

                    f_x = cell_l[1][s_i][0][0]
                    f_y = cell_l[1][s_i][0][1]

                    cell_radius = 5
                    x3 = f_x
                    y3 = f_y

                    cell_r = 9
                    print(frame_index, -1, x3 - cell_r, y3 - cell_r, cell_r * 2, cell_r * 2, 1, -1, -1, -1, file=f_det_txt, sep=',')

                    if (draw == True):
                        cv2.circle(frame_draw, (round(f_x * draw_scale), round(f_y * draw_scale)), 5 * draw_scale, (255, 255, 0),
                                   ((1 * draw_scale) >> 2))
                        cv2.putText(frame_draw, str(s_i) + "_" + str(det_score_2[s_i]),
                                    (round(f_x + 8) * draw_scale, round(f_y + 2.5) * draw_scale), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.125 * draw_scale, (255, 255, 0), max(round(0.25 * draw_scale), 1))

                elif(det_score_2[s_i] == 1):

                    f_x = cell_l[1][s_i][0][0]
                    f_y = cell_l[1][s_i][0][1]

                    if (draw == True):
                        cv2.circle(frame_draw, (round(f_x * draw_scale), round(f_y * draw_scale)), 5 * draw_scale, (255, 0, 255),
                                   ((1 * draw_scale) >> 2))
                        cv2.putText(frame_draw, str(s_i) + "_" + str(det_score_2[s_i]),
                                    (round(f_x + 8) * draw_scale, round(f_y + 2.5) * draw_scale), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.125 * draw_scale, (255, 0, 255), max(int(0.25 * draw_scale), 1))
                else:

                    f_x = cell_l[1][s_i][0][0]
                    f_y = cell_l[1][s_i][0][1]

                    if (draw == True):
                        cv2.circle(frame_draw, (round(f_x * draw_scale), round(f_y * draw_scale)), 5 * draw_scale, (255, 255, 255),
                                   ((1 * draw_scale) >> 2))
                        cv2.putText(frame_draw, str(s_i) + "_" + str(det_score_2[s_i]),
                                    (round(f_x + 8) * draw_scale, round(f_y + 2.5) * draw_scale), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.125 * draw_scale, (255, 255, 255), max(int(0.25 * draw_scale), 1))

                    if (gen_cell == True):
                        scale_cell = 1
                        one_cell = gray_org[
                                   round((y3 - cell_radius) * scale_cell):round((y3 + cell_radius) * scale_cell),
                                   round((x3 - cell_radius) * scale_cell):round((x3 + cell_radius) * scale_cell)]

                        scale_cell = 8

                        cell_img_path = cells_path + "{0:0=4d}".format(frame_index) + "_" + str(s_i) + ".tif"
                        cv2.imwrite(cell_img_path, one_cell)

                    pass

                pass
            except ZeroDivisionError:
                pass

        cv2.putText(frame_draw, str(frame_index), (5*draw_scale, 10*draw_scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * draw_scale, (0, 255, 255), round(0.3 * draw_scale))

        # print("one loop:", time.time() - temp_t)

        centers = []
        centers.append(cells)
        # print(cells)
        # for c in ret_centers[0]:
        #     f_x = c.horizontal_x
        #     f_y = c.vertical_y
        #     cv2.circle(frame_draw, (round(f_x * draw_scale), round(f_y * draw_scale)), 5 * draw_scale, (0, 255, 255),
        #                ((1 * draw_scale) >> 2))
        #     # cv2.putText(frame_draw, str(s_i) + "_" + str(det_score_2[s_i]),
        #     #             (round(f_x + 8) * draw_scale, round(f_y + 2.5) * draw_scale), cv2.FONT_HERSHEY_SIMPLEX,
        #     #             0.125 * draw_scale, (255, 255, 255), max(int(0.25 * draw_scale), 1))

        if (debug == 1):
            cv2.namedWindow('frame_draw', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame_draw', 900, 900)
            cv2.imshow('frame_draw', frame_draw)
            cv2.waitKey()


        return frame_draw, centers

    def det_out_focus_3_radius(self, path, frame, frame_index, scale, Beacon, f_det_txt, masked):
        # print("enter detect_and_level")

        debug = 0
        draw = False

        debug = 0
        draw = True

        frame_draw = None

        cell_size = 5

        # gen_cell = False
        gen_cell = True

        if (len(frame.shape) > 2):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_draw = frame.copy()
        else:
            gray = frame.copy()
            frame_draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        clr_org = frame_draw.copy()

        draw_scale = 8
        frame_draw = cv2.resize(frame_draw, (frame_draw.shape[1] * draw_scale , frame_draw.shape[0] * draw_scale), interpolation=cv2.INTER_CUBIC)
        gray_org = gray.copy()
        clr_org_8 = frame_draw.copy()

        if (debug == 1):
            cv2.namedWindow('gray_org', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('gray_org', 900, 900)
            cv2.imshow('gray_org', gray_org)
            # cv2.waitKey()

        centers = []  # vector of object centroids in a frame
        very_white_cell = []

        ##########***** detect black edge *****######################

        # ret, black = cv2.threshold(gray, min(self.edge_thr, 99), 255, cv2.THRESH_BINARY_INV)
        ret, black = cv2.threshold(masked, 80, 255, cv2.THRESH_BINARY_INV)
        # ret, black = cv2.threshold(gray, 0.95 * self.background_pixel, 255, cv2.THRESH_BINARY_INV)
        # print("black edge thresh: ", 0.95 * self.background_pixel)

        # num_labels, labels = cv2.connectedComponents(black)

        black_white = black

        if (debug == 1):
            cv2.namedWindow('black edge', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black edge', 900, 900)
            cv2.imshow('black edge', black)
            # cv2.waitKey()
            pass

        # t1 = time.time()
        contours, hierarchy = cv2.findContours(black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if(frame_index == 0):

            select_contour = []
            cell_r_s = []

            for i in range(len(contours)):
                try:

                    # if(hierarchy[0][i][3] > -1):
                    #     continue

                    # (x, y), radius = cv2.minEnclosingCircle(contours[i])
                    # area = cv2.contourArea(contours[i])
                    # cell_r_s.append([radius / 8, area / 64])
                    ok = True
                    if (ok == True):
                        select_contour.append(contours[i])
                        points = contours[i]
                        pixels = gray[points[:, 0, 1], points[:, 0, 0]]
                        (x, y), radius = cv2.minEnclosingCircle(contours[i])
                        area = cv2.contourArea(contours[i])
                        # if(radius >= 1 * scale):
                        cell_r_s.append([radius / scale, area / (scale * scale), np.mean(pixels)])
                    else:
                        # print("point: ", contours[i][0][0][0], contours[i][0][0][1], gray[contours[i][0][0][1]][contours[i][0][0][0]])
                        pass

                except ZeroDivisionError:
                    pass

            frame_0 = np.zeros_like(gray)
            cv2.drawContours(frame_0, select_contour, -1, (255, 255, 255), -1)

            debug = 0
            if (debug == 1):
                cv2.namedWindow('select_contour', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('select_contour', 900, 900)
                cv2.imshow('select_contour', frame_0)
                # cv2.imwrite(out_path + 'select_contour.png', frame_0)
                cv2.waitKey()

            cell_r_s = np.array(cell_r_s)
            max_r = max(cell_r_s[:, 0])
            bins = int(max_r/0.1)
            # arr = np.zeros((bins, 2))

            # for i in range(bins):
            #     arr[i][0] = max_r / bins * i

            # for i in range(len(cell_r_s)):
            #     # arr[int(cell_r_s[i][0])] += cell_r_s[i][1]
            #     for j in range(bins):
            #         if cell_r_s[i][0] <= arr[j][0]:
            #             arr[j][1] += cell_r_s[i][1]
            #             break

            # max_loc = np.argmax(arr[:, 1])
            # self.cell_core_r = arr[max_loc][0]
            # print("cell_core_r: ", self.cell_core_r, arr[max_loc][1])

            # print(arr[max_loc])

            # cond = np.count_nonzero(cell_r_s[:, 0] < 1)
            # print("radius < 1: ", cond, len(cell_r_s[:, 0]))

            # plt.figure(1, figsize=(7, 6 * 2))
            # plt.rcParams.update({'font.size': 16})
            # plt.subplot(121)
            # plt.imshow(gray, cmap="gray")
            # plt.xticks([]), plt.yticks([])

            #****** hist of num ********#

            hist_data = np.histogram(cell_r_s[:, 0], bins, [0, max_r])

            x_array = hist_data[1][0:-1]
            x_array = x_array + 0.05
            y_array_2gauss = hist_data[0]


            ###**** find 2 peaks ****#
            tmp = np.insert(hist_data[0], 0, 0)
            # tmp = hist_data[0]
            peaks_idx, _ = find_peaks(tmp, distance=15)
            peaks_idx = peaks_idx - 1
            peaks = np.array([hist_data[0][peaks_idx], hist_data[1][peaks_idx]])
            peaks[1] += 0.05
            # print(peaks_idx, peaks)
            max_loc = np.argmax(peaks[0])
            tmp = np.delete(peaks[0], max_loc, 0)
            max_2nd_loc = np.argmax(tmp)
            # if (max_2nd_loc >= max_loc):
            #     max_2nd_loc += 1
            #     p0_guess = [peaks[0][0], peaks[1][0], 0.5, peaks[0][max_2nd_loc], peaks[1][max_2nd_loc], 0.5]
            # else:
            #     p0_guess = [peaks[0][0], peaks[1][0], 0.5, peaks[0][max_loc], peaks[1][max_loc], 0.5]
            if (max_2nd_loc >= max_loc):
                max_2nd_loc += 1
                p0_guess = [peaks[0][max_loc], peaks[1][max_loc], 0.5, peaks[0][max_2nd_loc], peaks[1][max_2nd_loc], 0.5]
            else:
                p0_guess = [peaks[0][max_2nd_loc], peaks[1][max_2nd_loc], 0.5, peaks[0][max_loc], peaks[1][max_loc], 0.5]


            gg_init = models.Gaussian1D(p0_guess[0], p0_guess[1], p0_guess[2]) + models.Gaussian1D(p0_guess[3], p0_guess[4], p0_guess[5])
            # plt.plot(x_array, gg_init(x_array))
            fitter = fitting.LevMarLSQFitter()
            gg_fit = fitter(gg_init, x_array, y_array_2gauss)


            g0 = models.Gaussian1D(*(gg_fit.parameters[0:3]))
            g1 = models.Gaussian1D(*(gg_fit.parameters[3:6]))

            g0_tmp = g0(x_array)
            overlap_0 = [min(y_array_2gauss[i], g0_tmp[i]) for i in range(len(x_array)) if x_array[i] >= 0.5]
            overlap_0_sum = sum(overlap_0)
            # print(x_array, y_array_2gauss, g0_tmp, overlap_0, overlap_0_sum)

            g1_tmp = g1(x_array)
            overlap_1 = [min(y_array_2gauss[i], g1_tmp[i]) for i in range(len(x_array)) if x_array[i] >= 0.5]
            overlap_1_sum = sum(overlap_1)
            # print(y_array_2gauss, g1_tmp, overlap_1, overlap_1_sum)


            # # plt.subplot(122)
            # # hist_data = plt.hist(cell_r_s[:, 0], bins, [0, max_r], alpha=0.5)
            # plt.plot(x_array, y_array_2gauss, label = "Radii Histogram")
            # # plt.plot(x_array, gg_fit(x_array), label = ["{:.2f}".format(a) for a in gg_fit.parameters])
            # # g0 = models.Gaussian1D(*(gg_fit.parameters[0:3]))
            # # g1 = models.Gaussian1D(*(gg_fit.parameters[3:6]))
            # x_plot = np.arange(0, 10, 0.01)
            # # plt.plot(x_plot, g0(x_plot), label = "Gaussian Estimation 1" + str((gg_fit.parameters[0:3])))
            # # plt.plot(x_plot, g1(x_plot), label = "Gaussian Estimation 2" + str((gg_fit.parameters[3:6])))
            # # plt.plot(x_plot, g0(x_plot), label = str((gg_fit.parameters[0:3])))
            # # plt.plot(x_plot, g1(x_plot), label = str((gg_fit.parameters[3:6])))
            # #
            # # plt.plot(x_plot, g0(x_plot), label = "Gaussian Estimation 1($\mu_r:$" + "{:.2f}".format(gg_fit.parameters[1]) + ", " +  "$\sigma_r:${:.2f}".format(gg_fit.parameters[2]) + ")")
            # # plt.plot(x_plot, g1(x_plot), label = "Gaussian Estimation 2($\mu_r:$" + "{:.2f}".format(gg_fit.parameters[4]) + ", " +  "$\sigma_r:${:.2f}".format(gg_fit.parameters[5]) + ")")
            # plt.plot(x_plot, g0(x_plot), label = "Fitted Gaussian Distribution 1")
            # plt.plot(x_plot, g1(x_plot), label = "Fitted Gaussian Distribution 2")
            #
            #
            # # $\mu:$" + "{:.2f}".format(self.bg_gau_mean) + ", " +  "$\sigma:${:.2f}".format(self.bg_gau_std) + ")"
            #
            # # plt.plot(x_array, gg_init(x_array))
            #
            # # plt.text(0.0, 0.0, ["{:.2f}".format(a) for a in gg_fit.parameters], size = 10)
            # plt.xlim(0,7)
            # # plt.ylim(0,600)
            # plt.ylim(0,)
            # plt.xlabel("Radiuses of White Points")
            # plt.ylabel("Num. of White Points")
            # plt.legend(loc='best', prop={'size': 16})#fontsize=20
            # # plt.savefig(out_path + "fit_2_peaks.png")
            # plt.show()
            # # # # exit()


            if(np.count_nonzero(gg_fit.parameters[3:6] > 0) == 3 and overlap_1_sum > overlap_0_sum):
                self.cell_core_r = gg_fit.parameters[4]
                self.cell_core_r_std = gg_fit.parameters[5]
                self.radius_thr = [self.cell_core_r - 3 * self.cell_core_r_std,
                                   self.cell_core_r + 3 * self.cell_core_r_std]
            elif(np.count_nonzero(gg_fit.parameters[:3] > 0) == 3 and overlap_0_sum > overlap_1_sum):
                self.cell_core_r = gg_fit.parameters[1]
                self.cell_core_r_std = gg_fit.parameters[2]
                self.radius_thr = [self.cell_core_r - 3 * self.cell_core_r_std,
                                   self.cell_core_r + 3 * self.cell_core_r_std]
            else:
                print("Error, failed to calculate Gaussian Estimation of the Histogram of Cell radii: ", path)
                self.radius_thr = [1, sys.maxsize]
                pass

            self.radius_thr = [self.cell_core_r - 3 * self.cell_core_r_std,
                               self.cell_core_r + 3 * self.cell_core_r_std]

        remove_noise = []
        cells = []
        for i in range(len(contours)):
            try:
                (x, y), radius = cv2.minEnclosingCircle(contours[i])
                if (max(1, self.radius_thr[0]) * scale < radius < self.radius_thr[1] * scale):
                    remove_noise.append(contours[i])
                    # tmp_scale = scale
                    # scale = 8
                    # centeroid = (int(x * scale), int(y * scale))
                    # cv2.circle(frame_draw, centeroid, 5 * scale, (255, 255, 0), ((1 * scale) >> 2))

                    c = cell_d()
                    c.horizontal_x = x
                    c.vertical_y = y
                    c.area = 0
                    c.contour = contours[i]
                    c.radius = radius
                    c.max_pixel = 0
                    c.frame_idx = frame_index
                    # c.feature = features[s_i]
                    c.tlwh = np.array([c.vertical_y - 9, c.horizontal_x - 9, 18, 18])
                    # c.score = det_score[s_i][1]
                    # c.score = det_score_2[s_i]

                    cells.append(c)

                    # scale = tmp_scale
            except ZeroDivisionError:
                pass

            th5 = np.zeros_like(masked)
            cv2.drawContours(th5, remove_noise, -1, (255, 255, 255), -1)

            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 900, 900)
            cv2.imshow('frame', frame)
            # cv2.imwrite(out_path + 'frame.png', frame)
            # cv2.waitKey()

            cv2.namedWindow('remove_noise', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('remove_noise', 900, 900)
            cv2.imshow('remove_noise', th5)
            # cv2.imwrite(out_path + 'remove_noise.png', th5)
            # cv2.waitKey()

            cv2.namedWindow('frame_draw', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame_draw', 900, 900)
            cv2.imshow('frame_draw', frame_draw)
            # cv2.imwrite(out_path + 'frame_draw.png', frame_draw)
            cv2.waitKey()

        sum_r = 0
        amount_r = 0
        for i in range(len(cell_r_s)):
            if(cell_r_s[i][0] > 1):
                sum_r += cell_r_s[i][0]
                amount_r += 1

        self.cell_core_r_mean = sum_r / amount_r
        # print("self.cell_core_r: ", self.cell_core_r)
        # print("self.cell_core_r_mean: ", self.cell_core_r_mean)


        cv2.putText(frame_draw, str(frame_index), (5*draw_scale, 10*draw_scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * draw_scale, (0, 255, 255), round(0.3 * draw_scale))

        debug = 1
        if (debug == 1):
            cv2.namedWindow('frame_draw', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame_draw', 900, 900)
            cv2.imshow('frame_draw', frame_draw)
            cv2.waitKey()


        return frame_draw, centers


    def prepro_frames_2(self, path, prepro_images_path, fluo = False, out_path = None, Beacon = None):
        # print(path, prepro_images_path)
        #[2430:(2430 + (8525 - 7971)), 7971:8525]
        debug = 0
        scale = 8
        frame_count = 0
        image_a = None
        image_b = None

        last_vec = None
        motion_vectors = []

        # out_path = None
        # if(debug == 1):
        #     out_path = prepro_images_path + "debug/"
        #     if (not os.path.exists(out_path)):
        #         os.makedirs(out_path)

        # path = "/home/qibing/disk_t/Pt204/RawData/Beacon-2/"

        files = os.listdir(path)
        files = [x for x in files if
                 ("PNG" in x) or ("TIF" in x) or ("TIFF" in x) or ("JPG" in x) or ("JPEG" in x) or (
                             "tif" in x) or ("tiff" in x) or ("jpg" in x) or ("jpeg" in x)] #  or ("png" in x)

        if(fluo == True):
            files = [x for x in files if ("d1" in x)]# for pt935 with fluorescence images

        if (len(files) == 0):
            print("No images can be found!")
            return

        len_s = [len(x) for x in files]
        len_s = list(dict.fromkeys(len_s))
        len_s = np.array(len_s)
        len_s = np.sort(len_s)
        files_l = []
        for i in range(len(len_s)):
            a = [x for x in files if len(x) == len_s[i]]
            a.sort()
            files_l.append(a)
        files = [x for y in files_l for x in y]
        # print(files)

        # for frame_count in range(len(files)):
        #     frame = cv2.imread(path + files[frame_count])#, cv2.IMREAD_GRAYSCALE
        #     frame = frame[0:1024, 305:1041, :]
        #     out_image_path = "/home/qibing/Work/ground_truth/preprocess/" + "t" + "{0:0=3d}".format(frame_count) + ".tif"
        #     cv2.imwrite(out_image_path, frame)
        #
        # print("preprocess done")
        # exit()


        self.image_amount = min(amount_limit, len(files))
        image_amount_str = str(self.image_amount)
        print("adjust luminance")


        kernel2 = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])

        # Applying the filter2D() function


        for frame_count in range(self.image_amount):

            img_p = path + files[frame_count]
            frame = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)

            # # this part is for Pt935 Beacon 337-341. They are blurred ##################
            # frame = cv2.medianBlur(frame, 3)
            # frame = cv2.GaussianBlur(frame, [3, 3], 0)
            # frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel2)
            #
            # # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # # frame = clahe.apply(frame)
            # #
            # frame = new_hist[frame]
            # # this part is for Pt935 Beacon 337-341. They are blurred
            # #####################################################################33

            if(frame is None):
                frame = cv2.imread(path + files[frame_count - 1], cv2.IMREAD_GRAYSCALE)

            frame_org = frame.copy()

            tmp_img = cv2.resize(frame_org, (frame_org.shape[1] >> 2 , frame_org.shape[0] >> 2), interpolation=cv2.INTER_CUBIC)
            # for i in range(243, 305, 6):#may be this is for patient 174
            for i in range(21, 305, 6):
                # print("qibing: ", i)
                tmp_image_pad = cv2.copyMakeBorder(tmp_img, i, i, i, i, cv2.BORDER_REFLECT)
                bg = cv2.medianBlur(tmp_image_pad, i)  # There is an unexpected effect when ksize is 81, applied to 8 times scaled image.

                # cv2.namedWindow('bg', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('bg', 900, 900)
                # cv2.imshow('bg', bg)
                # # cv2.imwrite(out_path + 'black.png', black)
                # cv2.waitKey()


                tmp = np.where(bg < 15)
                if(len(tmp) > 0 and len(tmp[0]) < 10):
                    # print(np.where(bg < 10), len(np.where(bg < 10)), bg.flatten())
                    bg = bg[i:tmp_img.shape[0] + i, i:tmp_img.shape[1] + i]
                    break


            # i = 81
            # tmp_img = cv2.resize(frame_org, (frame_org.shape[1] , frame_org.shape[0]), interpolation=cv2.INTER_CUBIC)
            # tmp_image_pad = cv2.copyMakeBorder(tmp_img, i, i, i, i, cv2.BORDER_REFLECT)
            # bg = cv2.medianBlur(tmp_image_pad, i)  # There is an unexpected effect when ksize is 81, applied to 8 times scaled image.
            # bg = bg[i:tmp_img.shape[0] + i, i:tmp_img.shape[1] + i]

            bg = cv2.resize(bg, (frame_org.shape[1], frame_org.shape[0]), interpolation=cv2.INTER_CUBIC)

            # cv2.namedWindow('frame_org', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('frame_org', 900, 900)
            # cv2.imshow('frame_org', frame_org)
            # # cv2.imwrite(out_path + 'black.png', black)
            # cv2.waitKey()

            frame = (frame_org.astype(float) / bg.astype(float)) * (100.0)
            # frame = frame_org.astype(float) - bg.astype(float) + 100

            frame += 0.5 # rounding
            np.clip(frame, 0, 255, out=frame)
            frame = frame.astype(np.uint8)

            # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('frame', 900, 900)
            # cv2.imshow('frame', frame)
            # # cv2.imwrite(out_path + 'black.png', black)
            # cv2.waitKey()


            out_image_path = prepro_images_path + "{0:0=6d}".format(frame_count) + ".jpg"
            cv2.imwrite(out_image_path, frame)

            # calculate camera movement for all frames.
            if(frame_count == 0):
                image_b = frame
                last_vec = [0, 0]
                pass
            #
            #
            #     self.background_pixel_mean = frame.mean()
            #     self.background_pixel_std = frame.std()
            #
            #     # hi = plt.hist(frame.flatten(), 256, [0, 256], histtype='step', linewidth=2)
            #     # hi = hi[0]
            #     hi = cv2.calcHist([frame], [0], None, [256], (0, 256), accumulate=False)
            #     hi = hi.flatten()
            #
            #     self.background_pixel = np.argmax(hi)
            #
            #     # self.background_pixel_peak = np.argmax(hi[0])
            #     # self.background_pixel = self.background_pixel_mean
            #
            #     x_array = np.arange(256)
            #     y_array_2gauss = hi
            #     x_array = x_array.astype(int)
            #     y_array_2gauss = y_array_2gauss.astype(int)
            #     p0_guess = [hi[self.background_pixel], self.background_pixel, 4]
            #     popt_2gauss, pcov_2gauss = scipy.optimize.curve_fit(_1gaussian, x_array, y_array_2gauss, p0=p0_guess, maxfev = 5000)
            #     self.bg_gau_mean = popt_2gauss[1]
            #     self.bg_gau_std = popt_2gauss[2]
            #
            #     self.edge_thr = self.background_pixel_mean
            #     self.core_thr = self.bg_gau_mean + 3.0 * self.bg_gau_std
            #
            #     # # plt.figure(0, figsize=(7, 6))
            #     # # plt.xticks([]), plt.yticks([])
            #     # # plt.imshow(frame, cmap="gray")
            #     #
            #     # # plt.figure(1, figsize=(7, 6))
            #     # # plt.subplots_adjust(left=0.2, bottom=0.1, right=0.95, top=0.95)
            #     # # plt.tight_layout()
            #     # plt.rcParams.update({'font.size': 16})
            #     # # plt.title(str(pt) + "_" + str(Beacon))
            #     #
            #     # plt.plot(hi, label = "Image Histogram")
            #     # # plt.plot(x_array, _1gaussian(x_array, *popt_2gauss), '--', label = "Gaussian Estimation($\mu:$" + "{:.2f}".format(self.bg_gau_mean) + ", " +  "$\sigma:${:.2f}".format(self.bg_gau_std) + ")", linewidth=2, color = (0.7, 0.3, 0.7))
            #     # plt.plot(x_array, _1gaussian(x_array, *popt_2gauss), '--', label = "Fitted Gaussian Distribution", linewidth=2, color = (0.7, 0.3, 0.7))
            #     # # plt.legend(loc = "best")
            #     # plt.legend(loc='best', prop={'size': 16})
            #     # # plt.xlim(0, 255)
            #     # plt.ylim(0, 160000)
            #     # # plt.title(r'$\alpha > \beta$')
            #     # # plt.ylim(0, max(200000, hi[0][self.background_pixel]))
            #     #
            #     # # plt.plot(self.background_pixel, hi[self.background_pixel] + 0.05, 'o')
            #     # # plt.text(self.background_pixel, hi[self.background_pixel] + 0.05, "Peak Pixel")
            #     # # plt.plot(self.background_pixel, hi[self.background_pixel], 'o')
            #     # # plt.text(self.background_pixel + 0.05, hi[self.background_pixel], "Peak_Pixel = " + str(self.background_pixel))
            #     #
            #     # # print("background_pixel: ", hi[1][self.background_pixel] + 0.05, hi[0][self.background_pixel], self.background_pixel_mean, self.background_pixel_std)
            #     # # print("bg_gau_mean, bg_gau_std", self.bg_gau_mean, self.bg_gau_std)
            #     #
            #     # plt.xlabel("Pixel Value")
            #     # plt.ylabel("Num. of Pixels")
            #     # plt.tight_layout()
            #     # plt.show()
            #     # # plt.savefig(out_path + "pixel_hist.png")
            #
            #
            # # if(frame_index == 0):
            #     frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
            #     frame_draw = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            #     gray = frame.copy()
            #     ret, black = cv2.threshold(gray, self.background_pixel_mean, 255, cv2.THRESH_BINARY_INV)
            #     black_white = black
            #     contours, hierarchy = cv2.findContours(black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #     frame_0 = np.zeros_like(gray)
            #     black_contour = []
            #     for i in range(len(contours)):
            #         try:
            #             if (hierarchy[0][i][3] == -1 and hierarchy[0][i][2] != -1):
            #                 black_contour.append(contours[i])
            #         except ZeroDivisionError:
            #             pass
            #
            #     cv2.drawContours(frame_0, black_contour, -1, (255, 255, 255), -1)
            #     if (debug == 1):
            #         cv2.namedWindow('black', cv2.WINDOW_NORMAL)
            #         cv2.resizeWindow('black', 900, 900)
            #         cv2.imshow('black', black)
            #         # cv2.imwrite(out_path + 'black.png', black)
            #         cv2.waitKey()
            #
            #         cv2.namedWindow('frame_0', cv2.WINDOW_NORMAL)
            #         cv2.resizeWindow('frame_0', 900, 900)
            #         cv2.imshow('frame_0', frame_0)
            #         # cv2.imwrite(out_path + 'frame_0.png', frame_0)
            #         cv2.waitKey()
            #
            #     thresh = self.bg_gau_mean + 3.0 * self.bg_gau_std
            #     # print("cell thresh: ", thresh)
            #     ret, th4 = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            #     if (debug == 1):
            #         cv2.namedWindow('th4', cv2.WINDOW_NORMAL)
            #         cv2.resizeWindow('th4', 900, 900)
            #         cv2.imshow('th4', th4)
            #         # cv2.imwrite(out_path + 'th4.png', th4)
            #         cv2.waitKey()
            #
            #     contours, hierarchy = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #     cell_r_s = []
            #     select_contour = []
            #
            #     for i in range(len(contours)):
            #         try:
            #             # (x, y), radius = cv2.minEnclosingCircle(contours[i])
            #             # area = cv2.contourArea(contours[i])
            #             # cell_r_s.append([radius / 8, area / 64])
            #             ok = True
            #             for m in range(3):
            #                 for n in range(3):
            #                     idx_0 = int(contours[i][0][0][1] - 1 + m)
            #                     idx_1 = int(contours[i][0][0][0] - 1 + n)
            #                     # if(frame_0[idx_0][idx_1] == 255):
            #                     ret = cv2.pointPolygonTest(contours[i], (int(idx_1), int(idx_0)), False)
            #                     if (ret > 0 and (gray[idx_0][idx_1] <= self.background_pixel_mean or frame_0[idx_0][idx_1] == 0)):
            #                         # print(i, m, n, idx_0, idx_1, ret, gray[idx_0][idx_1], ok)
            #                         ok = False
            #                         break
            #
            #                 if (ok == False):
            #                     break
            #             if (ok == True):
            #                 select_contour.append(contours[i])
            #                 points = contours[i]
            #                 pixels = gray[points[:, 0, 1], points[:, 0, 0]]
            #                 (x, y), radius = cv2.minEnclosingCircle(contours[i])
            #                 area = cv2.contourArea(contours[i])
            #                 # if(radius >= 1 * scale):
            #                 cell_r_s.append([radius / scale, area / (scale * scale), np.mean(pixels)])
            #             else:
            #                 # print("point: ", contours[i][0][0][0], contours[i][0][0][1], gray[contours[i][0][0][1]][contours[i][0][0][0]])
            #                 pass
            #
            #         except ZeroDivisionError:
            #             pass
            #
            #     frame_0 = np.zeros_like(gray)
            #     cv2.drawContours(frame_0, select_contour, -1, (255, 255, 255), -1)
            #
            #     if (debug == 1):
            #         cv2.namedWindow('select_contour', cv2.WINDOW_NORMAL)
            #         cv2.resizeWindow('select_contour', 900, 900)
            #         cv2.imshow('select_contour', frame_0)
            #         # cv2.imwrite(out_path + 'select_contour.png', frame_0)
            #         cv2.waitKey()
            #
            #     cell_r_s = np.array(cell_r_s)
            #     max_r = max(cell_r_s[:, 0])
            #     bins = int(max_r/0.1)
            #     # arr = np.zeros((bins, 2))
            #
            #     # for i in range(bins):
            #     #     arr[i][0] = max_r / bins * i
            #
            #     # for i in range(len(cell_r_s)):
            #     #     # arr[int(cell_r_s[i][0])] += cell_r_s[i][1]
            #     #     for j in range(bins):
            #     #         if cell_r_s[i][0] <= arr[j][0]:
            #     #             arr[j][1] += cell_r_s[i][1]
            #     #             break
            #
            #     # max_loc = np.argmax(arr[:, 1])
            #     # self.cell_core_r = arr[max_loc][0]
            #     # print("cell_core_r: ", self.cell_core_r, arr[max_loc][1])
            #
            #     # print(arr[max_loc])
            #
            #     # cond = np.count_nonzero(cell_r_s[:, 0] < 1)
            #     # print("radius < 1: ", cond, len(cell_r_s[:, 0]))
            #
            #     # plt.figure(1, figsize=(7, 6 * 2))
            #     # plt.rcParams.update({'font.size': 16})
            #     # plt.subplot(121)
            #     # plt.imshow(gray, cmap="gray")
            #     # plt.xticks([]), plt.yticks([])
            #
            #     #****** hist of num ********#
            #
            #     hist_data = np.histogram(cell_r_s[:, 0], bins, [0, max_r])
            #
            #     x_array = hist_data[1][0:-1]
            #     x_array = x_array + 0.05
            #     y_array_2gauss = hist_data[0]
            #
            #
            #     ###**** find 2 peaks ****#
            #     tmp = np.insert(hist_data[0], 0, 0)
            #     # tmp = hist_data[0]
            #     peaks_idx, _ = find_peaks(tmp, distance=15)
            #     peaks_idx = peaks_idx - 1
            #     peaks = np.array([hist_data[0][peaks_idx], hist_data[1][peaks_idx]])
            #     peaks[1] += 0.05
            #     # print(peaks_idx, peaks)
            #     max_loc = np.argmax(peaks[0])
            #     tmp = np.delete(peaks[0], max_loc, 0)
            #     max_2nd_loc = np.argmax(tmp)
            #     # if (max_2nd_loc >= max_loc):
            #     #     max_2nd_loc += 1
            #     #     p0_guess = [peaks[0][0], peaks[1][0], 0.5, peaks[0][max_2nd_loc], peaks[1][max_2nd_loc], 0.5]
            #     # else:
            #     #     p0_guess = [peaks[0][0], peaks[1][0], 0.5, peaks[0][max_loc], peaks[1][max_loc], 0.5]
            #     if (max_2nd_loc >= max_loc):
            #         max_2nd_loc += 1
            #         p0_guess = [peaks[0][max_loc], peaks[1][max_loc], 0.5, peaks[0][max_2nd_loc], peaks[1][max_2nd_loc], 0.5]
            #     else:
            #         p0_guess = [peaks[0][max_2nd_loc], peaks[1][max_2nd_loc], 0.5, peaks[0][max_loc], peaks[1][max_loc], 0.5]
            #
            #
            #     gg_init = models.Gaussian1D(p0_guess[0], p0_guess[1], p0_guess[2]) + models.Gaussian1D(p0_guess[3], p0_guess[4], p0_guess[5])
            #     # plt.plot(x_array, gg_init(x_array))
            #     fitter = fitting.LevMarLSQFitter()
            #     gg_fit = fitter(gg_init, x_array, y_array_2gauss)
            #
            #
            #     g0 = models.Gaussian1D(*(gg_fit.parameters[0:3]))
            #     g1 = models.Gaussian1D(*(gg_fit.parameters[3:6]))
            #
            #     g0_tmp = g0(x_array)
            #     overlap_0 = [min(y_array_2gauss[i], g0_tmp[i]) for i in range(len(x_array)) if x_array[i] >= 0.5]
            #     overlap_0_sum = sum(overlap_0)
            #     # print(x_array, y_array_2gauss, g0_tmp, overlap_0, overlap_0_sum)
            #
            #     g1_tmp = g1(x_array)
            #     overlap_1 = [min(y_array_2gauss[i], g1_tmp[i]) for i in range(len(x_array)) if x_array[i] >= 0.5]
            #     overlap_1_sum = sum(overlap_1)
            #     # print(y_array_2gauss, g1_tmp, overlap_1, overlap_1_sum)
            #
            #
            #     # # plt.subplot(122)
            #     # # hist_data = plt.hist(cell_r_s[:, 0], bins, [0, max_r], alpha=0.5)
            #     # plt.plot(x_array, y_array_2gauss, label = "White Points Radii Histogram")
            #     # # plt.plot(x_array, gg_fit(x_array), label = ["{:.2f}".format(a) for a in gg_fit.parameters])
            #     # # g0 = models.Gaussian1D(*(gg_fit.parameters[0:3]))
            #     # # g1 = models.Gaussian1D(*(gg_fit.parameters[3:6]))
            #     # x_plot = np.arange(0, 10, 0.01)
            #     # # plt.plot(x_plot, g0(x_plot), label = "Gaussian Estimation 1" + str((gg_fit.parameters[0:3])))
            #     # # plt.plot(x_plot, g1(x_plot), label = "Gaussian Estimation 2" + str((gg_fit.parameters[3:6])))
            #     # # plt.plot(x_plot, g0(x_plot), label = str((gg_fit.parameters[0:3])))
            #     # # plt.plot(x_plot, g1(x_plot), label = str((gg_fit.parameters[3:6])))
            #     # #
            #     # # plt.plot(x_plot, g0(x_plot), label = "Gaussian Estimation 1($\mu_r:$" + "{:.2f}".format(gg_fit.parameters[1]) + ", " +  "$\sigma_r:${:.2f}".format(gg_fit.parameters[2]) + ")")
            #     # # plt.plot(x_plot, g1(x_plot), label = "Gaussian Estimation 2($\mu_r:$" + "{:.2f}".format(gg_fit.parameters[4]) + ", " +  "$\sigma_r:${:.2f}".format(gg_fit.parameters[5]) + ")")
            #     # plt.plot(x_plot, g0(x_plot), label = "Fitted Gaussian Distribution 1")
            #     # plt.plot(x_plot, g1(x_plot), label = "Fitted Gaussian Distribution 2")
            #     #
            #     #
            #     # # $\mu:$" + "{:.2f}".format(self.bg_gau_mean) + ", " +  "$\sigma:${:.2f}".format(self.bg_gau_std) + ")"
            #     #
            #     # # plt.plot(x_array, gg_init(x_array))
            #     #
            #     # # plt.text(0.0, 0.0, ["{:.2f}".format(a) for a in gg_fit.parameters], size = 10)
            #     # plt.xlim(0,7)
            #     # # plt.ylim(0,600)
            #     # plt.ylim(0,)
            #     # plt.xlabel("Radiuses of White Points")
            #     # plt.ylabel("Num. of White Points")
            #     # plt.legend(loc='best', prop={'size': 16})#fontsize=20
            #     # plt.savefig(out_path + "fit_2_peaks.png")
            #     # plt.show()
            #     # # # exit()
            #
            #
            #     if(np.count_nonzero(gg_fit.parameters[3:6] > 0) == 3 and overlap_1_sum > overlap_0_sum):
            #         self.cell_core_r = gg_fit.parameters[4]
            #         self.cell_core_r_std = gg_fit.parameters[5]
            #         self.radius_thr = [self.cell_core_r - 3 * self.cell_core_r_std,
            #                            self.cell_core_r + 3 * self.cell_core_r_std]
            #     elif(np.count_nonzero(gg_fit.parameters[:3] > 0) == 3 and overlap_0_sum > overlap_1_sum):
            #         self.cell_core_r = gg_fit.parameters[1]
            #         self.cell_core_r_std = gg_fit.parameters[2]
            #         self.radius_thr = [self.cell_core_r - 3 * self.cell_core_r_std,
            #                            self.cell_core_r + 3 * self.cell_core_r_std]
            #     else:
            #         print("Error, failed to calculate Gaussian Estimation of the Histogram of Cell radii: ", path)
            #         self.radius_thr = [1, sys.maxsize]
            #         # self.cell_core_r = 0
            #         # self.cell_core_r_std = 0
            #         pass
            #
            #     # print(self.cell_core_r, self.cell_core_r_std, self.radius_thr)
            #     # print("radius, std: ", self.cell_core_r, self.cell_core_r_std)
            #     # self.noise_radius_thresh = (self.cell_core_r - 3 * self.cell_core_r_std)
            #     # print("noise radius thresh: ", self.noise_radius_thresh)
            #
            #     if (debug == 1):
            #         self.radius_thr = [self.cell_core_r - 3 * self.cell_core_r_std,
            #                            self.cell_core_r + 3 * self.cell_core_r_std]
            #         remove_noise = []
            #         for i in range(len(select_contour)):
            #             try:
            #                 (x, y), radius = cv2.minEnclosingCircle(select_contour[i])
            #
            #                 if (max(1, self.radius_thr[0]) * scale < radius < self.radius_thr[1] * scale):
            #                     remove_noise.append(select_contour[i])
            #                     centeroid = (int(x), int(y))
            #                     cv2.circle(frame_draw, centeroid, 5 * scale, (255, 255, 0), ((1 * scale) >> 2))
            #             except ZeroDivisionError:
            #                 pass
            #
            #         th5 = np.zeros_like(th4)
            #         cv2.drawContours(th5, remove_noise, -1, (255, 255, 255), -1)
            #
            #         cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            #         cv2.resizeWindow('frame', 900, 900)
            #         cv2.imshow('frame', frame)
            #         # cv2.imwrite(out_path + 'frame.png', frame)
            #         cv2.waitKey()
            #
            #         cv2.namedWindow('remove_noise', cv2.WINDOW_NORMAL)
            #         cv2.resizeWindow('remove_noise', 900, 900)
            #         cv2.imshow('remove_noise', th5)
            #         # cv2.imwrite(out_path + 'remove_noise.png', th5)
            #         cv2.waitKey()
            #
            #         cv2.namedWindow('frame_draw', cv2.WINDOW_NORMAL)
            #         cv2.resizeWindow('frame_draw', 900, 900)
            #         cv2.imshow('frame_draw', frame_draw)
            #         cv2.imwrite(out_path + 'frame_draw.png', frame_draw)
            #         cv2.waitKey()
            #
            #     sum_r = 0
            #     amount_r = 0
            #     for i in range(len(cell_r_s)):
            #         if(cell_r_s[i][0] > 1):
            #             sum_r += cell_r_s[i][0]
            #             amount_r += 1
            #
            #     self.cell_core_r_mean = sum_r / amount_r
            #     # print("self.cell_core_r: ", self.cell_core_r)
            #     # print("self.cell_core_r_mean: ", self.cell_core_r_mean)
            #

            else:
                print("\r", frame_count, end = "/" + image_amount_str, flush=True)
                image_a = image_b
                image_b = frame

                d0 = image_a.shape[0] >> 2
                d1 = image_a.shape[1] >> 2
                template = image_a[d0:3 * d0, d1:3 * d1]
                ret = cv2.matchTemplate(image_b, template, cv2.TM_SQDIFF)
                resu = cv2.minMaxLoc(ret)

                if(frame_count == 1):
                    last_vec = [resu[2][1] - d0, resu[2][0] - d1]
                else:
                    # last_vec = last_vec + [resu[2][1] - d0, resu[2][0] - d1]
                    last_vec = list(map(add, last_vec, [resu[2][1] - d0, resu[2][0] - d1]))

            motion_vectors.append(last_vec)
                # print(last_vec, end = " ")
            #
        print()

        # print("motion_vectors", motion_vectors)
        motion_vectors_arr = np.asarray(motion_vectors)
        average = [mean(motion_vectors_arr[:,0]), mean(motion_vectors_arr[:,1])]
        # print("average", average)
        motion_vectors_arr = motion_vectors_arr - average
        # print("motion_vectors_arr", motion_vectors_arr)

        # np.savetxt(prepro_images_path + "motion_vectors.txt", motion_vectors_arr, fmt='%d')

        ret = cv2.minMaxLoc(motion_vectors_arr)
        pad_wid = int(max(abs(ret[0]), abs(ret[1])))

        print("stable images")
        f_fluo = open(out_path + "Misc/info_ucf/fluo_Beacon-" + str(Beacon) + ".txt", "w")

        for i in range(self.image_amount):
            print("\r", i, end="/" + image_amount_str, flush=True)
            # image_path = prepro_images_path + "t" + "{0:0=3d}".format(i) + ".tif" #"/img1/{0:0=6d}".format(frame_index) + ".jpg"
            image_path = prepro_images_path + "{0:0=6d}".format(i) + ".jpg"
            frame = cv2.imread(image_path)

            # if(False and fluo == True): # Do not add the gray image and the fluo image together.
            #
            #     img_p = path + files[i]
            #     fluo_p = re.sub(r'd1', 'd0', img_p)
            #     fluo_img = cv2.imread(fluo_p, cv2.IMREAD_GRAYSCALE)
            #     fluo_img = np.where(fluo_img < 25, 0, fluo_img)
            #     fluo_img = fluo_img * 10
            #     frame = frame.astype(np.float16)
            #     frame[:, :, 0] += fluo_img
            #     frame = np.clip(frame, 0, 255)
            #     frame = frame.astype(np.uint8)

            frame_pad = cv2.copyMakeBorder(frame, pad_wid, pad_wid, pad_wid, pad_wid, cv2.BORDER_CONSTANT, value=(100, 100, 100))
            # new_frame = np.zeros((frame.shape[0] + motion_vectors_arr[i][0], frame.shape[1] + motion_vectors_arr[i][1]), np.uint8)
            new_frame = frame_pad[pad_wid + motion_vectors_arr[i][0]:pad_wid + motion_vectors_arr[i][0] + frame.shape[0], pad_wid + motion_vectors_arr[i][1]:pad_wid + motion_vectors_arr[i][1] + frame.shape[1]]

            cv2.imwrite(image_path, new_frame)

            if(fluo == True):
                img_p = path + files[i]
                fluo_p = re.sub(r'd1', 'd0', img_p)
                fluo_img = cv2.imread(fluo_p, cv2.IMREAD_GRAYSCALE)

                new_fluo_p = prepro_images_path + "fluo{0:0=6d}".format(i) + ".jpg"

                fluo_pad = cv2.copyMakeBorder(fluo_img, pad_wid, pad_wid, pad_wid, pad_wid, cv2.BORDER_CONSTANT, value=(100, 100, 100))
                # new_frame = np.zeros((frame.shape[0] + motion_vectors_arr[i][0], frame.shape[1] + motion_vectors_arr[i][1]), np.uint8)
                new_fluo = fluo_pad[pad_wid + motion_vectors_arr[i][0]:pad_wid + motion_vectors_arr[i][0] + frame.shape[0], pad_wid + motion_vectors_arr[i][1]:pad_wid + motion_vectors_arr[i][1] + frame.shape[1]]

                ###################################################################################333
                fluo_only = np.where(new_fluo < 25, 0, new_fluo)
                fluo_cnt = np.count_nonzero(fluo_only)
                contours, hierarchy = cv2.findContours(fluo_only, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for ci in range(len(contours)):
                    area = cv2.contourArea(contours[ci])
                    if (area > 300):
                        cv2.drawContours(fluo_only, [contours[ci]], -1, (0, 0, 0), -1)

                f_fluo.write(str(fluo_cnt) + "\n")
                ###################################################################################333

                # ret, black = cv2.threshold(gray, min(self.edge_thr, 99), 255, cv2.THRESH_BINARY_INV)
                # # ret, black = cv2.threshold(gray, 0.95 * self.background_pixel, 255, cv2.THRESH_BINARY_INV)
                # # print("black edge thresh: ", 0.95 * self.background_pixel)
                #
                # black_white = black
                #
                # if (debug == 1):
                #     cv2.namedWindow('black edge', cv2.WINDOW_NORMAL)
                #     cv2.resizeWindow('black edge', 900, 900)
                #     cv2.imshow('black edge', black)
                #     # cv2.waitKey()
                #     pass
                #
                # # t1 = time.time()
                # contours, hierarchy = cv2.findContours(black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                cv2.imwrite(new_fluo_p, fluo_only)



        # return True, frame
        print()


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

