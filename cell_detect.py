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
from astropy.modeling import models, fitting
import sys
# amount_limit = 790
amount_limit = 10
#
line_thick = 8
draw = True
colors = [(255, 255, 0), (255, 0, 255)]

# x3 = self.tracks[i].trace[-1][0]
# y3 = self.tracks[i].trace[-1][1]
# ratio = self.tracks[i].trace[-1][2]
# area = self.tracks[i].trace[-1][3]
# le = int(self.tracks[i].trace[-1][4])
# loc_var = self.tracks[i].trace[-1][5]

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

    def detect_by_contour(self, frame, scale):

        if (len(frame.shape) > 2):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # debug = 0
        debug = 1

        if (debug == 1):
            cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('gray', 900, 900)
            cv2.imshow('gray', gray)
            # cv2.waitKey()

        if (debug == 2):
            np.save("gray", gray)

        ret, black = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
        black_white = black

        if (debug == 1):
            cv2.namedWindow('black', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black', 900, 900)
            cv2.imshow('black', black)
            pass

        if (debug == 2):
            np.save("black", black)

        t1 = time.time()
        contours, hierarchy = cv2.findContours(black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        t2 = time.time()

        if (debug == 1):
            contours_image = np.zeros_like(black_white)

            cv2.drawContours(contours_image, contours, -1, (255, 255, 255), line_thick)

            cv2.namedWindow('contours_image2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('contours_image2', 900, 900)
            cv2.imshow('contours_image2', contours_image)

        centers = []  # vector of object centroids in a frame

        fake_cell = 0

        count = 0
        t3 = time.time()
        maximum = 0
        for i in range(len(contours)):
            try:
                if (hierarchy[0][i][3] > -1 and hierarchy[0][hierarchy[0][i][3]][3] < 0 and len(
                        contours[i]) < 200):  # and len(contours[i]) < (10 * scale) * (10 * scale)
                    (x, y), radius = cv2.minEnclosingCircle(contours[i])
                    centeroid = (int(x), int(y))
                    radius = int(radius)
                    if (radius < 90 and radius > 7 and gray[int(y)][int(x)] > 100):

                        retval = cv2.minAreaRect(contours[i])
                        ratio = min(retval[1][0], retval[1][1]) / max(retval[1][0], retval[1][1])

                        box = cv2.boxPoints(retval)
                        box = np.int0(box)
                        cv2.drawContours(black_white, [box], 0, 0, 1)
                        # cv2.putText(black_white, str(float("{0:.2f}".format(ratio))), (int(retval[0][0] - 20), int(retval[0][1] + 65)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                        b = np.array([[x / scale], [y / scale], [ratio]])
                        centers.append(b)

                        if (draw == True):
                            cv2.circle(frame, centeroid, 10 * scale, (0, 255, 255), 2)

                    else:
                        fake_cell = fake_cell + 1
            except ZeroDivisionError:
                pass

        debug = 0

        return centers

# I don't have to use annulus to detect cells. So I thought maybe I just use 2 overlap contours to detect cells. Actually the performance is not good.
    def detect_by_contour_ex(self, frame, scale):

        debug = 0

        if (len(frame.shape) > 2):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        gray_org = gray.copy()

        if (debug == 1):
            cv2.namedWindow('gray_org', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('gray_org', 900, 900)
            cv2.imshow('gray_org', gray_org)

        centers = []  # vector of object centroids in a frame
        very_white_cell = []
        ###********** detect very white core start ********#########
        if (debug == 1):
            print(time.ctime(time.time()))
        ret, th4 = cv2.threshold(gray, 175, 255,
                                 cv2.THRESH_BINARY)  # if you want to extend the white core, you can use 120

        if (debug == 1):
            cv2.namedWindow('very white_core', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('very white_core', 900, 900)
            cv2.imshow('very white_core', th4)
            # cv2.waitKey()

        contours, hierarchy = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours_image = np.zeros_like(gray)
        # cv2.drawContours(contours_image, contours, -1, (255, 255, 255), 1)

        if (len(frame.shape) == 2):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        fake_cell = 0

        for cnt in contours:
            try:
                retval = cv2.minAreaRect(cnt)
                ratio = min(retval[1][0], retval[1][1]) / max(retval[1][0], retval[1][1])

                (x, y), radius = cv2.minEnclosingCircle(cnt)
                x = int(x)
                y = int(y)
                centeroid = (int(x), int(y))

                if (x > 5 * scale and x < (frame.shape[1] - 5 * scale) and y > 5 * scale and y < (
                        frame.shape[0] - 5 * scale) and radius < 6 * scale and radius > 1.2 * scale and gray_org[y][
                    x] > 135):
                    b = np.array([x / scale, y / scale, ratio, cv2.contourArea(cnt), 0])
                    very_white_cell.append(b)
                    if (draw == True):
                        cv2.circle(frame, centeroid, 5 * scale, (255, 255, 0), int(0.5 * scale))

                    hull = cv2.convexHull(cnt)
                    cv2.drawContours(gray, [hull], -1, (0, 0, 0), -1)

                else:
                    if debug == 1:
                        # print('fake object number in the frame:')
                        pass
                    fake_cell = fake_cell + 1
            except ZeroDivisionError:
                pass

        if (debug == 1):
            # cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('gray', 900, 900)
            # cv2.imshow('gray', gray)

            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 900, 900)
            cv2.imshow('frame', frame)
            # cv2.waitKey()
        if (debug == 1):
            print(time.ctime(time.time()))

        ##########***** detect very white core end *****######################

        ##########***** detect black edge *****######################
        if (debug == 1):
            print(time.ctime(time.time()))

        ret, black = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY_INV)
        black_white = black

        if (debug == 1):
            cv2.namedWindow('black edge', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black edge', 900, 900)
            cv2.imshow('black edge', black)
            pass

        t1 = time.time()
        black_contours, black_hierarchy = cv2.findContours(black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        t2 = time.time()

        if (debug == 1):
            contours_image = np.zeros_like(gray)
            cv2.drawContours(contours_image, black_contours, -1, (255, 255, 255), 1)
            cv2.namedWindow('black_edge contours_image2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black_edge contours_image2', 900, 900)
            cv2.imshow('black_edge contours_image2', contours_image)

        frame_0 = np.zeros_like(gray)

        # frame_0
        if (debug == 1):
            print(time.ctime(time.time()))
        # print(len(contours))
        # print(hierarchy)
        for i in range(len(black_contours)):
            try:
                if (black_hierarchy[0][i][3] == -1 and black_hierarchy[0][i][2] != -1):
                    (x, y), radius = cv2.minEnclosingCircle(black_contours[i])
                    if (radius > 3 * scale):
                        cv2.drawContours(frame_0, black_contours, i, (255, 255, 255), -1)

                    # cv2.namedWindow('black frame', cv2.WINDOW_NORMAL)
                    # cv2.resizeWindow('black frame', 900, 900)
                    # cv2.imshow('black frame', frame_0)
                    # cv2.waitKey()
            except ZeroDivisionError:
                pass
        if (debug == 1):
            print(time.ctime(time.time()))

        # cv2.drawContours(frame_0, contours, -1, (255, 255, 255), -1)

        if (debug == 1):
            cv2.namedWindow('black frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black frame', 900, 900)
            cv2.imshow('black frame', frame_0)
            # cv2.waitKey()

        if (debug == 1):
            print(time.ctime(time.time()))

        ##########***** detect black edge end *****######################

        ##########***** detect strauma start *****######################
        if (debug == 1):
            print(time.ctime(time.time()))

        ret, th4 = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

        if (debug == 1):
            cv2.namedWindow('stroma 0', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('stroma 0', 900, 900)
            cv2.imshow('stroma 0', th4)
            # cv2.waitKey()

        contours, hierarchy = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            try:
                (x, y), radius = cv2.minEnclosingCircle(contours[i])
                # 9.2
                if (radius > 9.2 * scale):
                    cv2.drawContours(gray, contours, i, (0, 0, 0), -1)
            except ZeroDivisionError:
                pass

        if (debug == 1):
            cv2.namedWindow('stroma gray', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('stroma gray', 900, 900)
            cv2.imshow('stroma gray', gray)
            # cv2.waitKey()
        if (debug == 1):
            print(time.ctime(time.time()))

        # detect strauma end

        # cell with a closed black edge
        cell_with_edge = []

        for i in range(len(black_contours)):
            try:
                if (black_hierarchy[0][i][3] > -1 and black_hierarchy[0][black_hierarchy[0][i][3]][3] < 0):  # and len(black_contours[i]) < 200 and len(contours[i]) < (10 * scale) * (10 * scale)
                    (x, y), radius = cv2.minEnclosingCircle(black_contours[i])
                    centeroid = (int(x), int(y))
                    radius = int(radius)
                    if (radius < 90 and radius > 7 and gray[int(y)][int(x)] > 100):

                        retval = cv2.minAreaRect(black_contours[i])
                        ratio = min(retval[1][0], retval[1][1]) / max(retval[1][0], retval[1][1])

                        # box = cv2.boxPoints(retval)
                        # box = np.int0(box)
                        # cv2.drawContours(black_white, [box], 0, 0, 1)
                        # cv2.putText(black_white, str(float("{0:.2f}".format(ratio))), (int(retval[0][0] - 20), int(retval[0][1] + 65)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                        b = np.array([x / scale, y / scale, ratio, cv2.contourArea(black_contours[i]), 0])
                        cell_with_edge.append(b)

                        if (draw == True):
                            cv2.circle(frame, centeroid, 5 * scale, (255, 0, 255), 2)

                    else:
                        fake_cell = fake_cell + 1
            except ZeroDivisionError:
                pass


        if (debug == 1):
            cv2.namedWindow('result', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('result', 900, 900)
            cv2.imshow('result', frame)
            cv2.waitKey()

        if (debug == 1):
            print(time.ctime(time.time()))

        ##########**********######################
        # print("cells: ", len(very_white_cell), len(cell_with_edge), len(cell_with_half_edge), len(centers))
        # print(very_white_cell, cell_with_edge, cell_with_half_edge, centers)

        very_white_cell = np.array(very_white_cell)
        cell_with_edge = np.array(cell_with_edge)
        # cell_with_half_edge = np.array(cell_with_half_edge)
        centers.append(very_white_cell)
        centers.append(cell_with_edge)
        # centers.append(cell_with_half_edge)

        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('result', 900, 900)
        cv2.imshow('result', frame)
        cv2.waitKey()

        return centers

    def detect_by_white_core_iowa(self, frame, scale, frame_index):
        debug = 0

        if (len(frame.shape) > 2):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_draw = frame.copy()
            frame_draw = frame_draw.astype(np.uint8)
        else:
            gray = frame.copy()
            frame_draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if (debug == 1):
            cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('gray', 900, 900)
            cv2.imshow('gray', gray)
            # cv2.waitKey()
        # debug = 0

        # ret, th4 = cv2.threshold(gray, 30000/256, 255, cv2.THRESH_BINARY)# iowa channel 1
        ret, th4 = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)# iowa channel 2

        th4 = th4.astype(np.uint8)

        if (debug == 1):
            cv2.namedWindow('th4', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('th4', 900, 900)
            cv2.imshow('th4', th4)
            # cv2.waitKey()

        contours, hierarchy = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_img = np.zeros_like(th4)
        cv2.drawContours(contours_img, contours, -1, (255, 255, 0), 2)

        centers = []

        count = 0
        for cnt in contours:
            try:
                retval = cv2.minAreaRect(cnt)
                ratio = min(retval[1][0], retval[1][1]) / max(retval[1][0], retval[1][1])
                # ret = (retval[0], (retval[1][0] * 1.7, retval[1][1] * 1.7), retval[2])
                ret = (retval[0], (retval[1][0], retval[1][1]), retval[2])
                box = cv2.boxPoints(ret)
                box = np.int0(box)

                (x, y), radius = cv2.minEnclosingCircle(cnt)
                # cv2.circle(frame_draw, centeroid, int(radius), (255, 255, 0), 2)

                # if (retval[1][0] > 10 and retval[1][1] > 10):
                # if(radius > 15 and np.all(box[:, 0] > 0) and np.all(box[:, 0] < gray.shape[1]) and np.all(box[:, 1] > 0) and np.all(box[:, 1] < gray.shape[0])): # iowa channel 1
                if (radius > 8 and np.all(box[:, 0] > 0) and np.all(box[:, 0] < gray.shape[1]) and np.all(box[:, 1] > 0) and np.all(box[:, 1] < gray.shape[0])): # iowa channel 2
                    cv2.drawContours(frame_draw, [box], 0, (255, 255, 0), 2)
                    cv2.putText(frame_draw, str(count), (int(x + 50), int(y + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    b = np.array([x / scale, y / scale, ratio, cv2.contourArea(cnt), 0, 0, cnt, box, radius, frame_index])
                    centers.append(b)
                    count += 1
                else:
                    if debug == 1:
                        print('fake object number in the frame:')
            except ZeroDivisionError:
                pass

        if (debug == 1):
            cv2.namedWindow('result', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('result', 900, 900)
            cv2.imshow('result', frame_draw)
            cv2.waitKey()

        return frame_draw, [centers]


    def detect_hybrid(self, frame, scale):

        # debug = 0
        debug = 1

        if (len(frame.shape) > 2):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        gray_org = gray.copy()

        if (debug == 1):
            cv2.namedWindow('gray_org', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('gray_org', 900, 900)
            cv2.imshow('gray_org', gray_org)

        very_white_cell = []
        white_cell = []
        centers = []  # vector of object centroids in a frame

        ###********** detect very white core start ********#########

        ret, th4 = cv2.threshold(gray, 170, 255,
                                 cv2.THRESH_BINARY)  # if you want to extend the white core, you can use 120

        if (debug == 1):
            cv2.namedWindow('very white_core', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('very white_core', 900, 900)
            cv2.imshow('very white_core', th4)
            # cv2.waitKey()

        contours, hierarchy = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_image = np.zeros_like(gray)
        cv2.drawContours(contours_image, contours, -1, (255, 255, 255), 1)

        # cv2.drawContours(gray, contours, -1, (0, 0, 0), -1)
        # for i in range(len(contours)):
        #     try:
        #         if (hierarchy[0][i][3] == -1 and cv2.isContourConvex(contours[i]) == False):
        #             hull = cv2.convexHull(contours[i])
        #             cv2.drawContours(gray, [hull], -1, (0, 0, 0), -1)
        #             pass
        #     except ZeroDivisionError:
        #         pass

        if (debug == 1):
            cv2.namedWindow('very contours_image2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('very contours_image2', 900, 900)
            cv2.imshow('very contours_image2', contours_image)

        fake_cell = 0
        for cnt in contours:
            try:
                retval = cv2.minAreaRect(cnt)  # xie le mei ce
                ratio = min(retval[1][0], retval[1][1]) / max(retval[1][0], retval[1][1])

                (x, y), radius = cv2.minEnclosingCircle(cnt)
                x = int(x)
                y = int(y)
                centeroid = (int(x), int(y))

                if (radius < 6 * scale and radius > 1.2 * scale and gray_org[y][x] > 135):
                    b = np.array([[int(x / scale)], [int(y / scale)], [ratio], [cv2.contourArea(cnt)]])
                    if (draw == True):
                        cv2.circle(frame, centeroid, 5 * scale, (255, 255, 0), int(0.5 * scale))
                    very_white_cell.append(b)

                    hull = cv2.convexHull(cnt)
                    cv2.drawContours(gray, [hull], -1, (0, 0, 0), -1)

                else:
                    if debug == 1:
                        print('fake object number in the frame:')
                    fake_cell = fake_cell + 1
            except ZeroDivisionError:
                pass

        if (debug == 1):
            cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('gray', 900, 900)
            cv2.imshow('gray', gray)

        ##########***** detect very white core end *****######################

        ##########***** detect black edge *****######################

        ret, black = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY_INV)
        black_white = black

        if (debug == 1):
            cv2.namedWindow('black', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black', 900, 900)
            cv2.imshow('black', black)
            pass

        if (debug == 2):
            np.save("black", black)

        t1 = time.time()
        contours, hierarchy = cv2.findContours(black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        t2 = time.time()

        # print(hierarchy)

        if (debug == 1):
            contours_image = np.zeros_like(black_white)
            cv2.drawContours(contours_image, contours, -1, (255, 255, 255), line_thick)
            cv2.namedWindow('contours_image2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('contours_image2', 900, 900)
            cv2.imshow('contours_image2', contours_image)

        frame_0 = np.zeros_like(gray)

        # frame_0
        for i in range(len(contours)):
            try:
                if (hierarchy[0][i][3] == -1 and hierarchy[0][i][2] != -1):
                    (x, y), radius = cv2.minEnclosingCircle(contours[i])
                    if (radius > 3 * scale):
                        hull = cv2.convexHull(contours[i])
                        cv2.drawContours(frame_0, [hull], -1, (255, 255, 255), -1)
            except ZeroDivisionError:
                pass

        # approxCurve, approxPolyDP
        # for i in range(len(contours)):
        #     try:
        #         if (hierarchy[0][i][3] == -1):
        #             if(cv2.isContourConvex(contours[i]) == True):
        #                 cv2.drawContours(frame_0, [contours[i]], -1, (255, 255, 255), -1)
        #                 pass
        #             else:
        #                 approxCurve = cv2.approxPolyDP(contours[i], 20, True)
        #                 cv2.drawContours(frame_0, [approxCurve], -1, (255, 255, 255), -1)
        #                 pass
        #     except ZeroDivisionError:
        #         pass

        # for i in range(len(contours)):
        #     try:
        #         if (hierarchy[0][i][3] > -1 and hierarchy[0][hierarchy[0][i][3]][3] < 0 and len(contours[i]) < 200):  # and len(contours[i]) < (10 * scale) * (10 * scale)
        #             (x, y), radius = cv2.minEnclosingCircle(contours[i])
        #             centeroid = (int(x), int(y))
        #             radius = int(radius)
        #             if (radius < 90 and radius > 7 and gray[int(y)][int(x)] > 100):
        #
        #                 retval = cv2.minAreaRect(contours[i])
        #
        #                 ratio = min(retval[1][0], retval[1][1]) / max(retval[1][0], retval[1][1])
        #
        #                 box = cv2.boxPoints(retval)
        #                 box = np.int0(box)
        #                 cv2.drawContours(black_white, [box], 0, 0, 1)
        #                 cv2.putText(black_white, str(float("{0:.2f}".format(ratio))),
        #                             (int(retval[0][0] - 20), int(retval[0][1] + 65)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                             (0, 0, 0))
        #
        #                 b = np.array([[int(x/scale)], [int(y/scale)], [ratio], [cv2.contourArea(cnt)]])
        #                 centers.append(b)
        #                 cv2.circle(frame, centeroid, 10 * scale, (0, 255, 255), 2)
        #             else:
        #                 fake_cell = fake_cell + 1
        #         elif(hierarchy[0][i][3] == -1 and cv2.isContourConvex(contours[i]) == False):
        #             hull = cv2.convexHull(contours[i])
        #             cv2.drawContours(frame_0, [hull], -1, (255, 255, 255), -1)
        #
        #         else:
        #             pass
        #     except ZeroDivisionError:
        #         pass

        if (debug == 1):
            cv2.namedWindow('black frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black frame', 900, 900)
            cv2.imshow('black frame', frame_0)
            # cv2.waitKey()
        ##########***** detect black edge end *****######################

        # detect strauma start

        ret, th4 = cv2.threshold(gray, 103, 255, cv2.THRESH_BINARY)

        if (debug == 1):
            cv2.namedWindow('stroma 0', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('stroma 0', 900, 900)
            cv2.imshow('stroma 0', th4)
            # cv2.waitKey()

        contours, hierarchy = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_image = np.zeros_like(gray)
        cv2.drawContours(contours_image, contours, -1, (255, 255, 255), 1)

        if (debug == 1):
            cv2.namedWindow('stroma contours_image2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('stroma contours_image2', 900, 900)
            cv2.imshow('stroma contours_image2', contours_image)

        for i in range(len(contours)):
            try:
                (x, y), radius = cv2.minEnclosingCircle(contours[i])
                # 9.2
                if (radius > 8 * scale):
                    cv2.drawContours(gray, contours, i, (0, 0, 0), -1)
            except ZeroDivisionError:
                pass

        if (debug == 1):
            cv2.namedWindow('stroma gray', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('stroma gray', 900, 900)
            cv2.imshow('stroma gray', gray)
            # cv2.waitKey()

        # detect strauma end

        ###********** detect white core start ********#########

        ret, th4 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        if (debug == 1):
            cv2.namedWindow('white_core', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('white_core', 900, 900)
            cv2.imshow('white_core', th4)
            # cv2.waitKey()

        contours, hierarchy = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_image = np.zeros_like(gray)
        cv2.drawContours(contours_image, contours, -1, (255, 255, 255), 1)

        if (debug == 1):
            cv2.namedWindow('contours_image2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('contours_image2', 900, 900)
            cv2.imshow('contours_image2', contours_image)

        fake_cell = 0
        for cnt in contours:
            try:

                retval = cv2.minAreaRect(cnt)
                ratio = min(retval[1][0], retval[1][1]) / max(retval[1][0], retval[1][1])

                (x, y), radius = cv2.minEnclosingCircle(cnt)
                x = int(x)
                y = int(y)
                centeroid = (int(x), int(y))

                # 9.2
                if (radius < 9.2 * scale and radius > 1 * scale and gray[y][x] > 125):
                    b = np.array([[int(x / scale)], [int(y / scale)], [ratio], [cv2.contourArea(cnt)]])
                    if (frame_0[y][x] == 255):
                        if (draw == True):
                            cv2.circle(frame, centeroid, 5 * scale, (255, 0, 255), int(0.5 * scale))
                        white_cell.append(b)
                else:
                    if debug == 1:
                        print('fake object number in the frame:')
                    fake_cell = fake_cell + 1
            except ZeroDivisionError:
                pass

        if (debug == 1):
            cv2.namedWindow('result', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('result', 900, 900)
            cv2.imshow('result', frame)
            cv2.waitKey()

        ##########**********######################

        very_white_cell = np.array(very_white_cell)
        white_cell = np.array(white_cell)
        centers.append(very_white_cell)
        centers.append(white_cell)

        return centers

    def detect_edge_test(self, frame, frame_index, scale):

        # debug = 0
        debug = 0
        # draw = false

        if (len(frame.shape) > 2):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        gray_org = gray.copy()

        if (debug == 1):
            cv2.namedWindow('gray_org', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('gray_org', 900, 900)
            cv2.imshow('gray_org', gray_org)

        centers = []  # vector of object centroids in a frame
        very_white_cell = []
        ###********** detect very white core start ********#########
        if(debug == 2):
            temp_t = time.time()

        ret, th4 = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)  # if you want to extend the white core, you can use 120

        contours, hierarchy = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours_image = np.zeros_like(gray)
        # cv2.drawContours(contours_image, contours, -1, (255, 255, 255), 1)

        if (len(frame.shape) == 2):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        fake_cell = 0

        for cnt in contours:
            try:
                retval = cv2.minAreaRect(cnt)
                ratio = min(retval[1][0], retval[1][1]) / max(retval[1][0], retval[1][1])

                # box = cv2.boxPoints(retval)
                # box = np.int0(box)
                # cv2.drawContours(th4, [box], 0, 255, 1)

                (x, y), radius = cv2.minEnclosingCircle(cnt)
                x = int(x)
                y = int(y)
                centeroid = (int(x), int(y))

                # cv2.putText(frame, str(float("{0:.2f}".format(ratio))), (int(x + 50), int(y + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.1 * scale, (255, 255, 255), int(0.2 * scale))

                if (frame.shape[1] - 5 * scale) > x > 5 * scale < y < (frame.shape[0] - 5 * scale) and 6 * scale > radius > 1.2 * scale and gray_org[y][x] > 135:

                    # cell = gray_org[(y - 5 * scale):(y + 5 * scale + 1), (x - 5 * scale):(x + 5 * scale + 1)]
                    # my_mask = np.full(((10 * scale + 1), (10 * scale + 1)), 1, dtype = np.uint8)
                    # cv2.circle(my_mask, (5 * scale, 5 * scale), 5 * scale, (0, 0, 0), -1)
                    # mask_arr = ma.masked_array(cell, mask = my_mask)
                    # out_arr = ma.var(mask_arr)

                    # cell = np.where(cell > 120, 1, 0)
                    # area = np.count_nonzero(cell)

                    b = np.array([x / scale, y / scale, ratio, int(cv2.contourArea(cnt)), 0, 0])
                    very_white_cell.append(b)

                    if (draw == True):
                        cv2.circle(frame, centeroid, 5 * scale, (255, 255, 0), int(0.5 * scale))
                        # cv2.putText(frame, str(float("{0:.2f}".format(ratio))), (int(x + 50), int(y + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (255, 255, 0), int(0.5 * scale))
                        cv2.putText(frame, str(int(cv2.contourArea(cnt))), (int(x + 50), int(y + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (255, 255, 0), int(0.5 * scale))


                    hull = cv2.convexHull(cnt)
                    cv2.drawContours(gray, [hull], -1, (0, 0, 0), -1)

                else:
                    if debug == 1:
                        # print('fake object number in the frame:')
                        pass
                    fake_cell = fake_cell + 1
            except ZeroDivisionError:
                pass

        if (debug == 1):
            cv2.namedWindow('very white_core', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('very white_core', 900, 900)
            cv2.imshow('very white_core', th4)
            # cv2.waitKey()

            # cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('gray', 900, 900)
            # cv2.imshow('gray', gray)

            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 900, 900)
            cv2.imshow('frame', frame)
            # cv2.waitKey()
        if (debug == 2):
            print("detect very white core t:", time.time() - temp_t)

        ##########***** detect very white core end *****######################

        ##########***** detect black edge *****######################
        if (debug == 2):
            temp_t = time.time()

        ret, black = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY_INV)
        black_white = black

        if (debug == 1):
            cv2.namedWindow('black edge', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black edge', 900, 900)
            cv2.imshow('black edge', black)
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

        frame_0 = np.zeros_like(gray)

        # frame_0

        t1 = time.time()
        black_contour = []
        for i in range(len(contours)):
            try:
                if (hierarchy[0][i][3] == -1 and hierarchy[0][i][2] != -1):
                    # (x, y), radius = cv2.minEnclosingCircle(contours[i])
                    # if (radius > 3 * scale):
                    # cv2.drawContours(frame_0, contours, i, (255, 255, 255), -1)
                    black_contour.append(contours[i])
                    # cv2.namedWindow('black frame', cv2.WINDOW_NORMAL)
                    # cv2.resizeWindow('black frame', 900, 900)
                    # cv2.imshow('black frame', frame_0)
                    # cv2.waitKey()
            except ZeroDivisionError:
                pass

        if(debug == 2):
            print(time.time() - t1)

        t1 = time.time()
        cv2.drawContours(frame_0, black_contour, -1, (255, 255, 255), -1)

        if (debug == 2):
            print(time.time() - t1)

        # t1 = time.time()
        # cv2.drawContours(frame_0, contours, -1, (255, 255, 255), -1)
        # print(time.time() - t1)

        if (debug == 1):
            cv2.namedWindow('black frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black frame', 900, 900)
            cv2.imshow('black frame', frame_0)
            # cv2.waitKey()

        if (debug == 2):
            print("detect black edge", time.time() - temp_t)
        ##########***** detect black edge end *****######################

        ##########***** detect strauma start *****######################
        if (debug == 2):
            temp_t = time.time()

        ret, th4 = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

        if (debug == 1):
            cv2.namedWindow('stroma 0', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('stroma 0', 900, 900)
            cv2.imshow('stroma 0', th4)
            # cv2.waitKey()

        contours, hierarchy = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # if (debug == 1):
        #     contours_image = np.zeros_like(gray)
        #     cv2.drawContours(contours_image, contours, -1, (255, 255, 255), 1)
        #     cv2.namedWindow('stroma contours_image2', cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow('stroma contours_image2', 900, 900)
        #     cv2.imshow('stroma contours_image2', contours_image)

        for i in range(len(contours)):
            try:
                (x, y), radius = cv2.minEnclosingCircle(contours[i])
                # 9.2
                if (radius > 9.2 * scale):
                    cv2.drawContours(gray, contours, i, (0, 0, 0), -1)
            except ZeroDivisionError:
                pass

        if (debug == 1):
            cv2.namedWindow('stroma gray', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('stroma gray', 900, 900)
            cv2.imshow('stroma gray', gray)
            # cv2.waitKey()

        if (debug == 2):
            print("detect strauma time: ", time.time() - temp_t)

        # detect strauma end

        ###********** detect white core start ********#########
        if (debug == 2):
            temp_t = time.time()

        cell_with_edge = []
        cell_with_half_edge = []
        ret, th4 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

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

        fake_cell = 0

        for i in range(len(contours)):
            try:
                retval = cv2.minAreaRect(contours[i])
                ratio = min(retval[1][0], retval[1][1]) / max(retval[1][0], retval[1][1])

                (x, y), radius = cv2.minEnclosingCircle(contours[i])
                x = int(x)
                y = int(y)
                centeroid = (int(x), int(y))

                # 9.2

                if (x > 5 * scale and x < (frame.shape[1] - 5 * scale) and y > 5 * scale and y < (frame.shape[0] - 5 * scale)):
                    pass
                else:
                    continue

                if (radius > 1 * scale and gray[y][x] > 125 and frame_0[y][x] == 255):  # radius < 2 * scale and gray[y][x] > 125 and
                    cell = gray_org[(y - 5 * scale):(y + 5 * scale + 1), (x - 5 * scale):(x + 5 * scale + 1)]
                    my_mask = np.full(((10 * scale + 1), (10 * scale + 1)), 1, dtype=np.uint8)
                    cv2.circle(my_mask, (5 * scale, 5 * scale), 5 * scale, (0, 0, 0), -1)
                    mask_arr = ma.masked_array(cell, mask=my_mask)
                    out_arr = ma.var(mask_arr)

                    b = np.array([x / scale, y / scale, ratio, cv2.contourArea(contours[i]), 1, out_arr])
                    cell_with_edge.append(b)
                    if (draw == True):

                        cv2.circle(frame, centeroid, 5 * scale, (255, 0, 255), int(0.5 * scale))
                        # cv2.putText(frame, str(float("{0:.2f}".format(ratio))), (int(x + 50), int(y + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (255, 0, 255), int(0.5 * scale))
                        cv2.putText(frame, str(int(out_arr)), (int(x + 50), int(y + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (255, 0, 255), int(0.5 * scale))

                        # cv2.putText(frame, str(i), (int(x + 65), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 0), int(0.5 * scale))
                ##### annulus test
                # elif(radius < 3 * scale and radius > 1 * scale and gray[y][x] > 120): # and gray[y][x] > 120
                #
                #     r1 = radius * 1.5
                #     r2 = radius * 2.5
                #     r3 = (r1 + r2) * 0.5
                #
                #     start_y = int(max(y - r2, 0))
                #     end_y = int(min(y + r2, frame.shape[0]))
                #
                #     start_x = int(max(x - r2, 0))
                #     end_x = int(min(x + r2, frame.shape[1]))
                #     cell = gray[start_y:end_y, start_x:end_x]
                #
                #     white_core_bin = th4[start_y:end_y, start_x:end_x]
                #     white_core_mask = np.logical_not(white_core_bin)
                #     white_core_gray = ma.array(cell, mask=white_core_mask)
                #     cell_max = white_core_gray.max()
                #
                #     whole_cell_mask = np.zeros((cell.shape[0], cell.shape[1]), dtype=np.uint8)
                #     whole_cell_mask[:] = 0
                #
                #     if(x > r2):
                #         center_x = int(r2)
                #     else:
                #         center_x = int(x)
                #
                #     if(y > r2):
                #         center_y = int(r2)
                #     else:
                #         center_y = int(y)
                #
                #     if(draw == True):
                #         cv2.circle(whole_cell_mask, (center_x, center_y), int(r2), (255, 255, 255), -1)
                #
                #     if(cell_max > 125):
                #         line = r2 - r1
                #         my_mask = np.zeros((cell.shape[0], cell.shape[1]), dtype=np.uint8)
                #         my_mask[:] = 0
                #         if(draw == True):
                #             cv2.circle(my_mask, ((center_x, center_y)), int(r3), (255, 255, 255), int(line))
                #
                #         my_mask = np.logical_not(my_mask)
                #
                #         annulus = ma.array(cell, mask=my_mask)
                #
                #         ann_mean = annulus.mean()
                #         ann_var = annulus.var()
                #
                #         if(ann_mean < 97 and ann_var < 290):
                #             b = np.array([x / scale, y / scale, ratio, cv2.contourArea(contours[i]), 2])
                #             cell_with_half_edge.append(b)
                #
                #             if(draw == True):
                #                 cv2.circle(frame, centeroid, int(r1), (0, 255, 0), int(0.3 * scale))
                #                 cv2.circle(frame, centeroid, int(r2), (0, 255, 0), int(0.3 * scale))
                #         else:
                #             pass

                else:
                    if debug == 1:
                        # print('fake object number in the frame:')
                        pass
                    fake_cell = fake_cell + 1
            except ZeroDivisionError:
                pass

        if (debug == 1):
            cv2.namedWindow('result', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('result', 900, 900)
            cv2.imshow('result', frame)
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

        cv2.putText(frame, str(frame_index), (5*scale, 10*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (0, 255, 255), int(0.3 * scale))

        # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('result', 900, 900)
        # cv2.imshow('result', frame)
        # cv2.waitKey()

        return frame, centers

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

    def detect_by_edge_core_and_level_RFP(self, path, frame, frame_index, scale):

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
            cv2.waitKey()


        centers = []  # vector of object centroids in a frame
        very_white_cell = []

        ##########***** detect black edge *****######################
        if (debug == 2):
            temp_t = time.time()

        # print("black edge thresh: ", self.edge_thr)
        ret, black = cv2.threshold(gray, self.edge_thr, 255, cv2.THRESH_BINARY_INV)
        # ret, black = cv2.threshold(gray, 0.95 * self.background_pixel, 255, cv2.THRESH_BINARY_INV)
        # print("black edge thresh: ", 0.95 * self.background_pixel)

        black_white = black

        if (debug == 1):
            cv2.namedWindow('black edge', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black edge', 900, 900)
            cv2.imshow('black edge', black)
            cv2.waitKey()
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
            cv2.waitKey()

        frame_0 = np.zeros_like(gray)

        # frame_0

        # t1 = time.time()
        black_contour = []
        for i in range(len(contours)):
            try:
                if (hierarchy[0][i][3] == -1 and hierarchy[0][i][2] != -1):
                    # (x, y), radius = cv2.minEnclosingCircle(contours[i])
                    # if (radius > 3 * scale):
                    # cv2.drawContours(frame_0, contours, i, (255, 255, 255), -1)
                    black_contour.append(contours[i])
                    # cv2.namedWindow('black frame', cv2.WINDOW_NORMAL)
                    # cv2.resizeWindow('black frame', 900, 900)
                    # cv2.imshow('black frame', frame_0)
                    # cv2.waitKey()
            except ZeroDivisionError:
                pass



        # for i in range(len(contours)):
        #     try:
        #         if (hierarchy[0][i][3] == -1):
        #             # new_cont = cv2.convexHull(contours[i])
        #             # new_cont = cv2.approxPolyDP(contours[i], 10*scale, True)
        #             black_contour.append(contours[i])
        #             cv2.fillConvexPoly(frame_0, contours[i], (255, 255, 255))
        #     except ZeroDivisionError:
        #         pass

        # t1 = time.time()
        cv2.drawContours(frame_0, black_contour, -1, (255, 255, 255), -1)
        # cv2.fillPoly(frame_0, contours, (255, 255, 255))

        # kernel = np.ones((3 * scale, 3 * scale), np.uint8)
        # frame_0 = cv2.dilate(frame_0, kernel, iterations=1)


        if (debug == 1):
            cv2.namedWindow('black frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('black frame', 900, 900)
            cv2.imshow('black frame', frame_0)
            cv2.waitKey()

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
            cv2.waitKey()

        contours, hierarchy = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if (debug == 1):
            contours_image = np.zeros_like(gray)
            cv2.drawContours(contours_image, contours, -1, (255, 255, 255), 1)
            cv2.namedWindow('white point contour', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('white point contour', 900, 900)
            cv2.imshow('white point contour', contours_image)
            cv2.waitKey()

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




                bRect_x, bRect_y, w, h = cv2.boundingRect(contours[i])
                rect = gray_org[bRect_y:bRect_y + h, bRect_x:bRect_x + w]
                local_cnt = contours[i]
                local_cnt = np.array(local_cnt)
                local_cnt[:, :, 0] = local_cnt[:, :, 0] - bRect_x
                local_cnt[:, :, 1] = local_cnt[:, :, 1] - bRect_y
                my_mask = np.zeros_like(rect)
                my_mask[:, :] = 255
                cv2.drawContours(my_mask, [local_cnt], -1, (0, 0, 0), -1)

                mask_arr = ma.masked_array(rect, mask=my_mask)
                brightest = ma.max(mask_arr)
                bright_mean = ma.mean(mask_arr)

                # if(brightest < 200):#self.max_pixel
                #     continue

                # if(brightest < 130):#self.max_pixel
                #     continue



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
                # if (max(1, self.radius_thr[0]) * scale < radius < 20 * scale and frame_0[y][x] == 255): # < self.radius_thr[1] * scale
                if (max(1, self.radius_thr[0]) * scale < radius and frame_0[y][x] == 255): # < self.radius_thr[1] * scale
                    cell = gray_org[(y - cell_size * scale):(y + cell_size * scale + 1), (x - cell_size * scale):(x + cell_size * scale + 1)]
                    black_new[(y - cell_size * scale):(y + cell_size * scale + 1), (x - cell_size * scale):(x + cell_size * scale + 1)] = gray_org[(y - cell_size * scale):(y + cell_size * scale + 1), (x - cell_size * scale):(x + cell_size * scale + 1)]

                    # x, y, w, h = cv2.boundingRect(contours[i])
                    # rect = gray_org[y:y + h, x:x + w]
                    # local_cnt = contours[i]
                    # local_cnt = np.array(local_cnt)
                    # local_cnt[:, :, 0] = local_cnt[:, :, 0] - x
                    # local_cnt[:, :, 1] = local_cnt[:, :, 1] - y
                    # my_mask = np.zeros_like(rect)
                    # my_mask[:, :] = 255
                    # cv2.drawContours(my_mask, [local_cnt], -1, (0, 0, 0), -1)
                    #
                    # mask_arr = ma.masked_array(rect, mask=my_mask)
                    # brightest = ma.max(mask_arr)
                    # bright_mean = ma.mean(mask_arr)

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


    def detect_by_white_core_and_level(self, frame, frame_index, scale):

        # print("enter detect_and_level")
        debug = 0
        draw = False
        # debug = 1
        # draw = True
        # if(frame_index == 94):
        #     debug = 1
        #     draw = True

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

        centers = []  # vector of object centroids in a frame
        very_white_cell = []


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

        fake_cell = 0

        black_new = np.zeros_like(gray)

        # print("radius thr: ", self.radius_thr)
        # print("max pixel thr: ", self.max_pixel)
        for i in range(len(contours)):
            try:

                (x, y), radius = cv2.minEnclosingCircle(contours[i])
                x = int(x)
                y = int(y)
                centeroid = (int(x), int(y))



                if (x > cell_size * scale and x < (frame.shape[1] - cell_size * scale) and y > cell_size * scale and y < (frame.shape[0] - cell_size * scale)):
                    pass
                else:
                    continue

                if (x > radius and x < (frame.shape[1] - radius) and y > radius and y < (frame.shape[0] - radius)):
                    pass
                else:
                    continue

                    # if (radius > (np.max(1.0, self.cell_core_r - 3 * self.cell_core_r_std)) * scale and frame_0[y][x] == 255):  # radius < 2 * scale and gray[y][x] > 125 and and gray[y][x] > 125
                # if (radius > self.noise_radius_thresh * scale and frame_0[y][x] == 255):  # radius < 2 * scale and gray[y][x] > 125 and and gray[y][x] > 125
                # if (self.noise_radius_thresh * scale < radius < (self.cell_core_r + 3 * self.cell_core_r_std) * scale and frame_0[y][x] == 255):
                # if (max(1, self.noise_radius_thresh) * scale < radius < (self.cell_core_r + 3 * self.cell_core_r_std) * scale):
                if (max(1, self.radius_thr[0]) * scale < radius):
                    cell = gray_org[(y - cell_size * scale):(y + cell_size * scale + 1), (x - cell_size * scale):(x + cell_size * scale + 1)]

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


                    if(brightest < self.max_pixel):
                        continue
                    # black_new[(y - cell_size * scale):(y + cell_size * scale + 1), (x - cell_size * scale):(x + cell_size * scale + 1)] = gray_org[(y - cell_size * scale):(y + cell_size * scale + 1), (x - cell_size * scale):(x + cell_size * scale + 1)]

                    # if(len(frames_arr) == 0):
                    #     frames_arr = frame.flatten()
                    # else:
                    #     frames_arr = np.concatenate((frames_arr, frame.flatten()))

                    retval = cv2.minAreaRect(contours[i])
                    a = max(retval[1][0], retval[1][1])  # long
                    b = min(retval[1][0], retval[1][1])  # short
                    area = cv2.contourArea(contours[i])
                    ratio = b / a
                    eccentricity = np.sqrt(1 - ratio ** 2)

                    # ret = (retval[0], (retval[1][0] * 1.7, retval[1][1] * 1.7), retval[2])
                    ret = (retval[0], (retval[1][0], retval[1][1]), retval[2])
                    box = cv2.boxPoints(ret)
                    box = np.int0(box)

                    # cv2.drawContours(frame_draw, [box], -1, (0, 0, 255), 1)

                    new_cell = np.where(cell > 175, cell, 0)
                    cell_sum = np.sum(new_cell)

                    # x3 = self.tracks[i].trace[-1][0]
                    # y3 = self.tracks[i].trace[-1][1]
                    # ratio = self.tracks[i].trace[-1][2]
                    # area = self.tracks[i].trace[-1][3]
                    # le = int(self.tracks[i].trace[-1][4])
                    # loc_var = self.tracks[i].trace[-1][5]

                    # print(area, bright_mean, eccentricity)

                    # cell_r = 40
                    # one_cell = gray_org[centeroid[1] - cell_r:centeroid[1] + cell_r + 1, centeroid[0] - cell_r:centeroid[0] + cell_r + 1]
                    #
                    # local_cnt = contours[i]
                    # local_cnt = np.array(local_cnt)
                    # local_cnt[:, :, 0] = local_cnt[:, :, 0] - (centeroid[0] - cell_r)
                    # local_cnt[:, :, 1] = local_cnt[:, :, 1] - (centeroid[1] - cell_r)
                    # one_cell_color = one_cell.copy()
                    # one_cell_color = cv2.cvtColor(one_cell_color, cv2.COLOR_GRAY2BGR)
                    # cv2.drawContours(one_cell_color, [local_cnt], -1, (0, 0, 255), 1)

                    # core = th4[centeroid[1] - cell_r:centeroid[1] + cell_r + 1, centeroid[0] - cell_r:centeroid[0] + cell_r + 1]
                    # cv2.imshow('one_cell', one_cell)
                    # cv2.imshow('core', core)
                    # cv2.imshow('my_mask', my_mask)
                    # cv2.imshow('mask_arr', mask_arr)
                    #
                    # cv2.namedWindow('one_cell_color', cv2.WINDOW_NORMAL)
                    # cv2.resizeWindow('one_cell_color', 900, 900)
                    # cv2.imshow('one_cell_color', one_cell_color)
                    #
                    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                    # cv2.resizeWindow('result', 900, 900)
                    # cv2.imshow('result', frame_draw)
                    #
                    # cv2.waitKey()



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
                        cv2.circle(frame_draw, centeroid, 7 * scale, (0, 255, 0), int(1 * scale))
                        # cv2.putText(frame_draw, str(i), (int(x + 95), int(y + 30)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                        # print(i, radius, self.cell_core_r, self.cell_core_r_std, self.noise_radius_thresh)
                        pass
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

        if (draw == True):
            cv2.putText(frame_draw, str(frame_index), (10*scale, 20*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 255, 255), int(0.6 * scale))

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


        # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('result', 900, 900)
        # cv2.imshow('result', frame)
        # cv2.waitKey()

        # print("detect and level end")
        return frame_draw, centers

    def prepro_frames_2(self, path, prepro_images_path):
        print(path, prepro_images_path)
        #[2430:(2430 + (8525 - 7971)), 7971:8525]
        debug = 0
        scale = 8
        frame_count = 0
        image_a = None
        image_b = None

        last_vec = None
        motion_vectors = []

        out_path = None
        if(debug == 1):
            out_path = prepro_images_path + "debug/"
            if (not os.path.exists(out_path)):
                os.makedirs(out_path)

        # path = "/home/qibing/disk_t/Pt204/RawData/Beacon-2/"

        files = os.listdir(path)
        files = [x for x in files if
                 ("PNG" in x) or ("TIF" in x) or ("TIFF" in x) or ("JPG" in x) or ("JPEG" in x) or (
                             "tif" in x) or ("tiff" in x) or ("jpg" in x) or ("jpeg" in x)] #  or ("png" in x)

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
        for frame_count in range(self.image_amount):

            frame = cv2.imread(path + files[frame_count], cv2.IMREAD_GRAYSCALE)
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


            out_image_path = prepro_images_path + "t" + "{0:0=3d}".format(frame_count) + ".tif"
            cv2.imwrite(out_image_path, frame)

            # calculate camera movement for all frames.
            if(frame_count == 0):
                out_image_path = prepro_images_path + "{0:0=1d}".format(frame_count) + ".tif"
                cv2.imwrite(out_image_path, frame)

                image_b = frame
                last_vec = [0, 0]


                self.background_pixel_mean = frame.mean()
                self.background_pixel_std = frame.std()

                # hi = plt.hist(frame.flatten(), 256, [0, 256], histtype='step', linewidth=2)
                # hi = hi[0]
                hi = cv2.calcHist([frame], [0], None, [256], (0, 256), accumulate=False)
                hi = hi.flatten()

                self.background_pixel = np.argmax(hi)

                # self.background_pixel_peak = np.argmax(hi[0])
                # self.background_pixel = self.background_pixel_mean

                x_array = np.arange(256)
                y_array_2gauss = hi
                x_array = x_array.astype(int)
                y_array_2gauss = y_array_2gauss.astype(int)
                p0_guess = [hi[self.background_pixel], self.background_pixel, 4]
                popt_2gauss, pcov_2gauss = scipy.optimize.curve_fit(_1gaussian, x_array, y_array_2gauss, p0=p0_guess, maxfev = 5000)
                self.bg_gau_mean = popt_2gauss[1]
                self.bg_gau_std = popt_2gauss[2]

                self.edge_thr = self.background_pixel_mean
                self.core_thr = self.bg_gau_mean + 3.0 * self.bg_gau_std

                # # plt.figure(0, figsize=(7, 6))
                # # plt.xticks([]), plt.yticks([])
                # # plt.imshow(frame, cmap="gray")
                #
                # # plt.figure(1, figsize=(7, 6))
                # # plt.subplots_adjust(left=0.2, bottom=0.1, right=0.95, top=0.95)
                # # plt.tight_layout()
                # plt.rcParams.update({'font.size': 16})
                # # plt.title(str(pt) + "_" + str(Beacon))
                #
                # plt.plot(hi, label = "Image Histogram")
                # # plt.plot(x_array, _1gaussian(x_array, *popt_2gauss), '--', label = "Gaussian Estimation($\mu:$" + "{:.2f}".format(self.bg_gau_mean) + ", " +  "$\sigma:${:.2f}".format(self.bg_gau_std) + ")", linewidth=2, color = (0.7, 0.3, 0.7))
                # plt.plot(x_array, _1gaussian(x_array, *popt_2gauss), '--', label = "Fitted Gaussian Distribution", linewidth=2, color = (0.7, 0.3, 0.7))
                # # plt.legend(loc = "best")
                # plt.legend(loc='best', prop={'size': 16})
                # # plt.xlim(0, 255)
                # plt.ylim(0, 160000)
                # # plt.title(r'$\alpha > \beta$')
                # # plt.ylim(0, max(200000, hi[0][self.background_pixel]))
                #
                # # plt.plot(self.background_pixel, hi[self.background_pixel] + 0.05, 'o')
                # # plt.text(self.background_pixel, hi[self.background_pixel] + 0.05, "Peak Pixel")
                # # plt.plot(self.background_pixel, hi[self.background_pixel], 'o')
                # # plt.text(self.background_pixel + 0.05, hi[self.background_pixel], "Peak_Pixel = " + str(self.background_pixel))
                #
                # # print("background_pixel: ", hi[1][self.background_pixel] + 0.05, hi[0][self.background_pixel], self.background_pixel_mean, self.background_pixel_std)
                # # print("bg_gau_mean, bg_gau_std", self.bg_gau_mean, self.bg_gau_std)
                #
                # plt.xlabel("Pixel Value")
                # plt.ylabel("Num. of Pixels")
                # plt.tight_layout()
                # plt.show()
                # # plt.savefig(out_path + "pixel_hist.png")


            # if(frame_index == 0):
                frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
                frame_draw = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                gray = frame.copy()
                ret, black = cv2.threshold(gray, self.background_pixel_mean, 255, cv2.THRESH_BINARY_INV)
                black_white = black
                contours, hierarchy = cv2.findContours(black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                frame_0 = np.zeros_like(gray)
                black_contour = []
                for i in range(len(contours)):
                    try:
                        if (hierarchy[0][i][3] == -1 and hierarchy[0][i][2] != -1):
                            black_contour.append(contours[i])
                    except ZeroDivisionError:
                        pass

                cv2.drawContours(frame_0, black_contour, -1, (255, 255, 255), -1)
                if (debug == 1):
                    cv2.namedWindow('black', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('black', 900, 900)
                    cv2.imshow('black', black)
                    # cv2.imwrite(out_path + 'black.png', black)
                    cv2.waitKey()

                    cv2.namedWindow('frame_0', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('frame_0', 900, 900)
                    cv2.imshow('frame_0', frame_0)
                    # cv2.imwrite(out_path + 'frame_0.png', frame_0)
                    cv2.waitKey()

                thresh = self.bg_gau_mean + 3.0 * self.bg_gau_std
                # print("cell thresh: ", thresh)
                ret, th4 = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
                if (debug == 1):
                    cv2.namedWindow('th4', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('th4', 900, 900)
                    cv2.imshow('th4', th4)
                    # cv2.imwrite(out_path + 'th4.png', th4)
                    cv2.waitKey()

                contours, hierarchy = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cell_r_s = []
                select_contour = []

                for i in range(len(contours)):
                    try:
                        # (x, y), radius = cv2.minEnclosingCircle(contours[i])
                        # area = cv2.contourArea(contours[i])
                        # cell_r_s.append([radius / 8, area / 64])
                        ok = True
                        for m in range(3):
                            for n in range(3):
                                idx_0 = int(contours[i][0][0][1] - 1 + m)
                                idx_1 = int(contours[i][0][0][0] - 1 + n)
                                # if(frame_0[idx_0][idx_1] == 255):
                                ret = cv2.pointPolygonTest(contours[i], (int(idx_1), int(idx_0)), False)
                                if (ret > 0 and (gray[idx_0][idx_1] <= self.background_pixel_mean or frame_0[idx_0][idx_1] == 0)):
                                    # print(i, m, n, idx_0, idx_1, ret, gray[idx_0][idx_1], ok)
                                    ok = False
                                    break

                            if (ok == False):
                                break
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
                # plt.plot(x_array, y_array_2gauss, label = "White Points Radii Histogram")
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
                # plt.savefig(out_path + "fit_2_peaks.png")
                # plt.show()
                # # # exit()


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
                    # self.cell_core_r = 0
                    # self.cell_core_r_std = 0
                    pass

                # print(self.cell_core_r, self.cell_core_r_std, self.radius_thr)
                # print("radius, std: ", self.cell_core_r, self.cell_core_r_std)
                # self.noise_radius_thresh = (self.cell_core_r - 3 * self.cell_core_r_std)
                # print("noise radius thresh: ", self.noise_radius_thresh)

                if (debug == 1):
                    self.radius_thr = [self.cell_core_r - 3 * self.cell_core_r_std,
                                       self.cell_core_r + 3 * self.cell_core_r_std]
                    remove_noise = []
                    for i in range(len(select_contour)):
                        try:
                            (x, y), radius = cv2.minEnclosingCircle(select_contour[i])

                            if (max(1, self.radius_thr[0]) * scale < radius < self.radius_thr[1] * scale):
                                remove_noise.append(select_contour[i])
                                centeroid = (int(x), int(y))
                                cv2.circle(frame_draw, centeroid, 5 * scale, (255, 255, 0), ((1 * scale) >> 2))
                        except ZeroDivisionError:
                            pass

                    th5 = np.zeros_like(th4)
                    cv2.drawContours(th5, remove_noise, -1, (255, 255, 255), -1)

                    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('frame', 900, 900)
                    cv2.imshow('frame', frame)
                    # cv2.imwrite(out_path + 'frame.png', frame)
                    cv2.waitKey()

                    cv2.namedWindow('remove_noise', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('remove_noise', 900, 900)
                    cv2.imshow('remove_noise', th5)
                    # cv2.imwrite(out_path + 'remove_noise.png', th5)
                    cv2.waitKey()

                    cv2.namedWindow('frame_draw', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('frame_draw', 900, 900)
                    cv2.imshow('frame_draw', frame_draw)
                    cv2.imwrite(out_path + 'frame_draw.png', frame_draw)
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

        np.savetxt(prepro_images_path + "motion_vectors.txt", motion_vectors_arr, fmt='%d')

        ret = cv2.minMaxLoc(motion_vectors_arr)
        pad_wid = int(max(abs(ret[0]), abs(ret[1])))

        print("stable images")
        for i in range(self.image_amount):
            print("\r", i, end="/" + image_amount_str, flush=True)
            image_path = prepro_images_path + "t" + "{0:0=3d}".format(i) + ".tif"
            frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            frame_pad = cv2.copyMakeBorder(frame, pad_wid, pad_wid, pad_wid, pad_wid, cv2.BORDER_CONSTANT, value=100)
            # new_frame = np.zeros((frame.shape[0] + motion_vectors_arr[i][0], frame.shape[1] + motion_vectors_arr[i][1]), np.uint8)
            new_frame = frame_pad[pad_wid + motion_vectors_arr[i][0]:pad_wid + motion_vectors_arr[i][0] + frame.shape[0], pad_wid + motion_vectors_arr[i][1]:pad_wid + motion_vectors_arr[i][1] + frame.shape[1]]
            cv2.imwrite(image_path, new_frame)
        # return True, frame
        print()
        # print("qibing 1: ", self.radius_thr)


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

