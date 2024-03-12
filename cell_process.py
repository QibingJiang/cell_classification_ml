import cv2
import copy

# from PIL.ImageOps import scale

from det_ml import CellDetector
from cell_classify import CellClassifier
import os
from matplotlib import pyplot as plt
import numpy as np
# from phagocytosis_detect import PhagocytosisDetector
import multiprocessing
import time
# import imageio
from itertools import chain
from operator import add
from statistics import mean
import sys
import re
from util import read_frame

from deep_sort_2 import tools
import generate_detections
# from generate_detections import *
# generate_detections
import deep_sort_app
import nn_matching
from tracker import Tracker

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# home_dir = os.path.expanduser("~") + "/"

debug = 0

crop_width = 0
crop_height = 0

# crop_width = 1328
# crop_height = 1048

# "/data/qibing/Pt935_MACROPHAGE_plate1_08102022/RawData/Beacon-92/"
# crop_width = 1092
# crop_height = 838

# crop_width = 512
# crop_height = 512

# crop_width = 256
# crop_height = 256
# crop_height = 128

# crop_width = 128
# crop_height = 128

# crop_width = 730
# crop_height = 1024


scale = 1

line_thick = 1
debug = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# configure_path = sys.argv[1]
# path = sys.argv[2]

def main(configure_path = "./configure.txt", path = "Work/ground_truth/preprocess", out_path = "Default", fluo=False):
# def main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt196/RawData/Beacon-77", out_path = "/home/qibing/disk_16t/Pt196/Beacon-77_512_old_preprocess"):

    if(out_path == "Default"):
        out_path = re.sub(r'RawData.*', 'TimeLapseVideos/', path)
    # else:
    #     out_path = home_dir + out_path
    

    ret = re.sub(r'.*Beacon-', '', path)
    Beacon = re.sub(r'/.*', '', ret)

    if(Beacon == ''):
        Beacon = 0
    else:
        Beacon = int(Beacon)

    if(Beacon == 0):
        ret = re.sub(r'.*Beacon_', '', path)
        Beacon = re.sub(r'/.*', '', ret)
        if(Beacon != ''):
            Beacon = int(Beacon)

    paras = []
    with open(configure_path) as f:
        for l in f:
            l = re.sub(r'#.*', '', l)# replace the comments with ''
            l = l.replace(" ", "")
            l = l.replace("\n", "")
            if(len(l) > 0):
                paras.append(l.split("="))

    paras_dict = {p[0]:p[1] for p in paras}

    for key in ["cell_core_radius_range_2", "cell_core_radius_range_3"]:
        radius_interval = paras_dict[key]
        radius_interval = radius_interval.replace("(", "")
        radius_interval = radius_interval.replace(")", "")
        radius_interval = radius_interval.split(",")
        radius_interval = [float(radius_interval[0]), float(radius_interval[1])]
        paras_dict[key] = radius_interval

    for key in ["cell_max_1", "black_edge_2", "white_core_2", "white_core_3", "cell_max_3"]:
        paras_dict[key] = float(paras_dict[key])

    print("Mode: ", paras_dict["Mode"])

    if (path[-1] != '/'):
        path += '/'

    if (out_path[-1] != '/'):
        out_path += '/'

    print(path, out_path)

    os.makedirs(out_path, exist_ok = True)

    for new_folder in ["images_ucf/Beacon_" + str(Beacon) + "/", "videos_ucf/", "Results_ucf/", "Misc/info_ucf/", "ML/", "Misc/Results_ucf/", "Results/"]:
        os.makedirs(out_path + new_folder, exist_ok = True)

    process_one_video_main(path, Beacon, 1, None, out_path, paras_dict, fluo=fluo) #path + "ML/" + "Beacon_" + str(Beacon)
    # generate_detections.main(out_path + "images_ucf/" + "Beacon_" + str(Beacon) + "/", out_path + "ML/")
    # deep_sort_app.my_run(out_path + "images_ucf/" + "Beacon_" + str(Beacon) + "/Beacon_" + str(Beacon) + "/", out_path + "ML/Beacon_" + str(Beacon) + ".npy", out_path)

def process_one_video_main(path, Beacon, data_type, pt, out_path, paras_dict, fluo=False):
    global crop_width, crop_height
    t0 = time.time()
    t0_str = time.ctime(t0)

    detector = CellDetector()
    classifier = CellClassifier(8, 20, 5, 0)
    # ph_detector = PhagocytosisDetector(10, 30, 5, 0)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, 100)
    tracker = Tracker(metric)

    image_path = path

    os.makedirs(out_path + "images_ucf/" + "Beacon_" + str(Beacon) + "/Beacon_" + str(Beacon) + "/det/", exist_ok=True)
    os.makedirs(out_path + "images_ucf/Beacon_" + str(Beacon) + "/Beacon_" + str(Beacon) + "/img1/", exist_ok=True)

    develop = 1
    if(develop == 0):
        detector.prepro_frames_2(image_path, out_path + "images_ucf/Beacon_" + str(Beacon) + "/Beacon_" + str(Beacon) + "/img1/", fluo=fluo, out_path = out_path, Beacon = Beacon)
        classifier.image_amount = tracker.image_amount = detector.image_amount
        # classifier.image_amount = detector.image_amount = 245

        if(paras_dict["Mode"] == '2'):
            detector.edge_thr = paras_dict["black_edge_2"]
            detector.core_thr = paras_dict["white_core_2"]
            detector.radius_thr = paras_dict["cell_core_radius_range_2"]
            pass
        elif (paras_dict["Mode"] == '3'):
            detector.core_thr = paras_dict["white_core_3"]
            detector.radius_thr = paras_dict["cell_core_radius_range_3"]
            detector.max_pixel = paras_dict["cell_max_3"]
            pass
        else:#this is mode 0 and 1
            # detector.edge_thr = detector.background_pixel_mean
            # detector.core_thr = detector.bg_gau_mean + 3.0 * detector.bg_gau_std
            # detector.radius_thr = [detector.cell_core_r - 3 * detector.cell_core_r_std, detector.cell_core_r + 3 * detector.cell_core_r_std]

            if (paras_dict["Mode"] == '1'):
                detector.max_pixel = paras_dict["cell_max_1"]
                pass
            # detector.max_pixel = 200
    else:
        classifier.image_amount = tracker.image_amount = detector.image_amount = 30
        detector.background_pixel = 100
        detector.edge_thr = detector.background_pixel_mean = 99.57
        detector.background_pixel_std = 18.02
        detector.bg_gau_mean = 100.24
        detector.bg_gau_std = 4.67
        detector.cell_core_r = 2.88
        detector.cell_core_r_std = 0.46
        detector.noise_radius_thresh = 1.0
        detector.core_thr = 120

        detector.radius_thr = [1, 10]
        # detector.image_amount = 10000
        paras_dict["Mode"] = '0'
        print("Developing Mode using Detection Mode: ", paras_dict["Mode"])

    frame_count = 0

    det_out = None
    tra_out = None
    make_video = True
    # make_video = False

    # file = open(image_path + "detect_result_" + time.strftime("%d_%H_%M", time.localtime()) + ".txt", "w")

    frame_prev = None


    f_det_txt = open(out_path + "/images_ucf/" + "Beacon_" + str(Beacon) + "/Beacon_" + str(Beacon) + "/det/det.txt", "w")

    print("detect and track:")
    for frame_count in range(detector.image_amount):
        # print(str(Beacon) + "_" + str(frame_count), end = " ", flush=True)
        print("\r", frame_count, end = "/" + str(detector.image_amount), flush=True)
        pre_pro_img_dir = out_path + "images_ucf/Beacon_" + str(Beacon) + "/Beacon_" + str(Beacon) + "/img1/"
        ret, frame_org = read_frame(pre_pro_img_dir, frame_count, data_type, scale, crop_width = crop_width, crop_height = crop_height, color = True)

        fluo_img = None
        if(fluo == True):
            fluo_p = pre_pro_img_dir + "fluo{0:0=6d}".format(frame_count) + ".jpg"
            fluo_img = cv2.imread(fluo_p, cv2.IMREAD_GRAYSCALE)

        if(crop_width == 0 and crop_height == 0):
            crop_width = int(frame_org.shape[1]/scale)
            crop_height = int(frame_org.shape[0]/scale)

        # frame = cv2.imread("/home/qibing/Work/ground_truth/output/images_ucf/Beacon_0/t096.tif", cv2.IMREAD_GRAYSCALE)
        # frame_org = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        # ret = True

        if(ret == False):
            detector.image_amount = frame_count
            print(__name__, "done")
            break

        gray_org = cv2.cvtColor(frame_org, cv2.COLOR_BGR2GRAY)
        frame_det = frame_org.copy()

        # centers = detector.detect_by_white_core(frame_det, scale, frame_count)
        # centers = detector.detect_hybrid(frame, scale)
        # centers = detector.detect_by_contour_ex(frame_det, scale)
        temp_t = time.time()
        # frame_det, centers = detector.detect_edge_test(frame_det, frame_count, scale)


        # if(paras_dict['Mode'] == '0' or paras_dict['Mode'] == '2'):
        #     # frame_det, centers = detector.detect_by_edge_core_and_level_RFP(out_path, frame_det, frame_count, scale)
        #     frame_det, centers = detector.detect_by_edge_core_and_level(out_path, frame_det, frame_count, scale)
        # elif(paras_dict['Mode'] == '1' or paras_dict['Mode'] == '3'):
        #     frame_det, centers = detector.detect_by_white_core_and_level(frame_det, frame_count, scale)
        # else:
        #     print("Mode is not defined: ", paras_dict['Mode'])
        #     pass
        # # print("det t:", time.time() - temp_t)

        temp_t = time.time()
        frame_det, centers = detector.det_ml(out_path, frame_det, frame_count, scale, Beacon, f_det_txt, gen_cell = False, fluo = fluo, fluo_img = fluo_img)

        frame_det_2 = frame_org.copy()
        # mask = np.zeros_like(frame_det_2)
        masked = gray_org.copy() # I do not use blue channel, since it may have fluorescence signal.

        # draw_scale = 1
        # for c in centers[0]:
        #     f_x = c.horizontal_x
        #     f_y = c.vertical_y
        #     score = c.score
        #     if(score > 0):
        #         cv2.circle(masked, (round(f_x * draw_scale), round(f_y * draw_scale)), 7 * draw_scale, (255, 255, 255), -1)
        #
        # cv2.namedWindow('masked', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('masked', 900, 900)
        # cv2.imshow('masked', masked)
        # cv2.waitKey()

        centers_2 = [[]]
        # frame_det, centers = detector.det_out_focus(out_path, frame_det, frame_count, scale, Beacon, f_det_txt)

        #There are beacons in focus or out of focus. I need to process them separately. The following is used to detect the cells out of focus, which are usually black.
        # frame_det_2, centers_2 = detector.det_out_focus_2(out_path, frame_det_2, frame_count, scale, Beacon, f_det_txt, masked, gen_cell = True)
        # frame_det_2, centers_2 = detector.det_out_focus_3_radius(out_path, masked, frame_count, scale, Beacon, f_det_txt, masked)
        # print("det t:", time.time() - temp_t)

        draw_scale = 6
        frame_det_3 = frame_org.copy()
        frame_det_3 = cv2.resize(frame_det_3, (frame_det_3.shape[1] * draw_scale, frame_det_3.shape[0] * draw_scale), interpolation=cv2.INTER_CUBIC)
        colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 255, 255), (255, 0, 0)]
        for i, c in enumerate(centers[0] + centers_2[0]):
            # print(i, c)
            f_x = c.horizontal_x
            f_y = c.vertical_y
            score = c.score

            if(score > -1):
                # cv2.circle(frame_det_3, (round(f_x * draw_scale), round(f_y * draw_scale)), 5 * draw_scale, colors[int(score)], ((1 * draw_scale) >> 2))
                # cv2.putText(frame_det_3, str(i) + "_" + str(score),
                #             (round(f_x + 8) * draw_scale, round(f_y + 2.5) * draw_scale), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.125 * draw_scale, colors[int(score)], max(int(0.25 * draw_scale), 1))

                cv2.circle(frame_det_3, (round(f_x * draw_scale), round(f_y * draw_scale)), 5 * draw_scale, (255, 255, 0), ((1 * draw_scale) >> 1))

        # cv2.namedWindow('frame_det_3', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame_det_3', 900, 900)
        # cv2.imshow('frame_det_3', frame_det_3)
        # cv2.waitKey()
        # continue

        tracker.predict()
        # detections = np.vstack(centers)
        detections = np.vstack([centers[0] + centers_2[0]])
        frame_tra = frame_org.copy()
        frame_tra = tracker.update(detections[0], frame_count, frame_tra, detector.image_amount)

        # cv2.namedWindow('frame_tra', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame_tra', 900, 900)
        # cv2.imshow('frame_tra', frame_tra)
        # cv2.waitKey()


        # if len(centers) > 0:
        #     # centers_cat = np.vstack(centers)
        #     frame_tra = frame_org.copy()
        #     # frame_tra = classifier.match_track(centers_cat, frame_prev, frame_tra, frame_count, scale)
        #     temp_t = time.time()
        #     frame_tra = classifier.match_track_3_times(centers, frame_prev, frame_tra, frame_count, scale)
        #     # print("match t:", time.time() - temp_t)
        #
        #     # ph_detector.match_track(centers, frame, frame_count)
        #
        #     cell_count = 0
        #     for i in range(len(centers)):
        #         # print(len(arr), arr[:, 3].sum(), end=" ")
        #         cell_count = cell_count + len(centers[i])
        #         # file.write(str(len(centers[i])) + " " + str(centers[i][:, 3].sum()) + " ")
        #
        #         # if(centers_cat == None):
        #         #     centers_cat = centers[i]
        #         # else:
        #         #     centers_cat = np.concatenate((centers_cat, centers[i]), axis=0)
        #
        #         # for point in centers[i]:
        #             # cv2.circle(frame_org, (int(point[0]), int(point[1])), 6, colors[i], 1)
        #
        #     # print(centers_cat)
        #     pass
        # else:
        #     print("not detected")

        # cv2.putText(frame, str(frame_count) + " " + str(cell_count), (5*scale, 10*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (0, 255, 255), int(0.3 * scale))
        # cv2.putText(frame_org, str(frame_count) + " " + str(cell_count), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), int(0.3))
        # cv2.putText(frame, str(len(centers)), (30*scale, 10*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (0, 255, 255), int(0.3 * scale))

        if make_video == True:
            if det_out is None:
                frame_det_4 = cv2.resize(frame_det_3, (crop_width * draw_scale, crop_height * draw_scale), interpolation=cv2.INTER_CUBIC)
                # det_out = cv2.VideoWriter(out_path + "cell_detect_" + time.strftime("%d_%H_%M", time.localtime()) + ".mp4", fourcc, 3.0, (frame_det.shape[1], frame_det.shape[0]), isColor=True)
                det_out = cv2.VideoWriter(out_path + "Beacon-" + str(Beacon) + "_detect.mp4", fourcc, 3.0, (frame_det_4.shape[1], frame_det_4.shape[0]), isColor=True)
            if tra_out is None:
                # tra_out = cv2.VideoWriter(out_path + "cell_track_" + time.strftime("%d_%H_%M", time.localtime()) + ".mp4", fourcc, 3.0, (frame_tra.shape[1], frame_tra.shape[0]), isColor=True)
                tra_out = cv2.VideoWriter(out_path + "Beacon-" + str(Beacon) + "_track.mp4", fourcc, 3.0, (frame_tra.shape[1], frame_tra.shape[0]), isColor=True)

        if (det_out != None):
            frame_det_4 = cv2.resize(frame_det_3, (crop_width * draw_scale, crop_height * draw_scale), interpolation=cv2.INTER_CUBIC)
            det_out.write(frame_det_4)
            cv2.imwrite(out_path + "det" + str(Beacon) + "_" +str(frame_count) + ".png", frame_det_4)
        if (tra_out != None):
            tra_out.write(frame_tra)
            cv2.imwrite(out_path + "tra" + str(Beacon) + "_" +str(frame_count) + ".png", frame_tra)

            # cv2.namedWindow('det', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('det', 900, 900)
            # cv2.imshow('det', frame_det_3)
            # # cv2.imwrite(image_path + "det_" + str(frame_count) + ".png", frame_det[154:625, 748:1340, :])
            # cv2.waitKey()

        # cv2.namedWindow('tra', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('tra', 900, 900)
        # cv2.imshow('tra', frame_tra)
        # cv2.imwrite(image_path + "tra_" + str(frame_count) + ".png", frame_tra[154:625, 748:1340, :])
        # cv2.waitKey()

        # cv2.imwrite(out_path + "det_" + str(frame_count) + ".png", frame_det)
        # cv2.imwrite(out_path + "tra_" + str(frame_count) + ".png", frame_tra)

        frame_prev = frame_org.copy()
    print()


    # print("Done!")

    if (det_out != None):
        det_out.release()

    if (tra_out != None):
        tra_out.release()

    classifier.background_pixel = detector.background_pixel
    classifier.cell_core_r = detector.cell_core_r
    classifier.cell_core_r_mean = detector.cell_core_r_mean


    gt_video_path = ""
    gt_video_path = re.sub(r'RawData.*', 'TimeLapseVideos/', path) + "Beacon-" + str(Beacon) + "processed.avi"
    # gt_video_path = home_dir + "Work/ground_truth/RFP.mp4"
    # # classifier.analyse_classification_7(out_path, detector.image_amount, gt_video_path, scale, Beacon)
    # mark_ground_truth(classifier, out_path + "images_ucf/Beacon_" + str(Beacon) + "/", Beacon, data_type, 8, detector.image_amount, out_path, gt_video_path)
    # #

    # classification
    tracker.analyse_classification_tmp(out_path, detector.image_amount, gt_video_path, scale, Beacon)#, outpath, frame_count, gt_video_path, scale, Beacon, gt = False

    #Cell Death prediction.
    tracker.analyse_classification_8(out_path, detector.image_amount, None, 8, Beacon, gt = False)

    # mark cell classification
    mark_ground_truth(tracker, pre_pro_img_dir, Beacon, data_type, 6, detector.image_amount, out_path, gt_video_path, fluo = fluo)

    # mark



    # mark(classifier, path, Beacon, data_type, scale)

    cv2.destroyAllWindows()

    # t1 = time.time()
    # t1_str = time.ctime(t1)
    #
    # if (not os.path.exists(image_path)):
    #     os.makedirs(image_path)
    # log = t0_str + "\n" + t1_str + "\n" + str((t1 - t0)) + " seconds.\n"
    # log_file = open(image_path + "/" + t0_str + "_log.txt", 'w')
    # log_file.write(log)
    # print(log)
    # log_file.close()
    # np.savetxt("/home/qibing/disk_t/" + pt + "/log", log)


# def mark(worker, path, Beacon, data_type, scale):
#
#     out2 = None
#     frame_count = 0
#
#     image_path = path + "/RawData/Beacon-" + str(Beacon) + "/"
#
#     while True:
#         ret, frame = read_frame(image_path, frame_count, data_type, scale)
#         if(ret == False):
#             print("done")
#             return
#
#         # print("mark cells frame_count:" + str(frame_count))
#
#         if(len(frame.shape) == 2):
#             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#
#         # frame = worker.mark(frame, frame_count, scale)
#         # ph_detector.mark_cells(frame, frame_count)
#         frame = worker.mark_mitosis(frame, frame_count, scale, path, frame_count)
#
#         if(not os.path.exists(image_path)):
#             os.makedirs(image_path)
#
#         if(out2 is None):
#             out2 = cv2.VideoWriter(image_path + "cell_classified_" + time.strftime("%d_%H_%M", time.localtime()) + ".mp4",fourcc, 3.0, (crop_width * 3, crop_height * 3), isColor=True)
#
#         if out2:
#             frame_vid = cv2.resize(frame, (crop_width * 3, crop_height * 3), interpolation=cv2.INTER_CUBIC)
#             out2.write(frame_vid)
#
#         # cv2.namedWindow('Tracking',cv2.WINDOW_NORMAL)
#         # cv2.resizeWindow('Tracking', 900,900)
#         # cv2.imshow('Tracking', frame)
#         # cv2.waitKey()
#
#         frame_count = frame_count + 1
#
#     print("Done!")
#     # cap2.release()
#     if(out2):
#         out2.release()
#     cv2.destroyAllWindows()

def mark_ground_truth(worker, image_path, Beacon, data_type, scale, frame_amount, out_path, gt_video_path, fluo = False):

    out2 = None
    vid = None
    frame_count = 0

    gt = False
    # gt = True
    save_img = False
    get_cells = True
    # get_cells = False
    f_det_txt = None
    class_f = open(out_path + "ML/class_ret_Beacon-" + str(Beacon) + ".txt", "w")

    if(gt == True and os.path.exists(gt_video_path)):
        # print(out_path + "Beacon-" + str(Beacon) + "processed.avi")
        vid = cv2.VideoCapture(gt_video_path)
        if(vid):
            skip = frame_amount - int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            print("mark cells skip: ", skip)

    print("mark cells:")
    image_amount_str = str(frame_amount)

    if (not os.path.exists(out_path + "tmp/" + "Beacon-" + str(Beacon) + "/")):
        os.makedirs(out_path + "tmp/" + "Beacon-" + str(Beacon) + "/")

    if(get_cells):
        bea_p = out_path + "/ML/" + "Beacon_" + str(Beacon) + "/"
        os.makedirs(bea_p, exist_ok=True)
        os.makedirs(bea_p + "img1/", exist_ok=True)
        os.makedirs(bea_p + "cells/", exist_ok=True)
        os.makedirs(bea_p + "det/", exist_ok=True)
        f_det_txt = open(bea_p + "det/det.txt", "w")

    for frame_count in range(frame_amount):
        ret, frame = read_frame(image_path, frame_count, data_type, scale, crop_width = crop_width, crop_height = crop_height, color = True)
        # ret, frame_org = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/Beacon_" + str(Beacon) + "/img1/", frame_count, data_type, scale, crop_width = crop_width, crop_height = crop_height)

        # cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame', 900,900)
        # cv2.imshow('frame', frame)
        # cv2.waitKey()

        if(ret == False):
            print(__name__, "done")
            return

        print("\r", frame_count, end="/" + image_amount_str, flush=True)

        if(len(frame.shape) == 2):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        gt_frame = np.zeros(0)
        ret = False
        if(gt and vid and frame_count >= skip):
            ret, gt_frame = vid.read()

        # frame = worker.mark(frame, frame_count, scale)
        # ret_1, frame_1 = read_frame(image_path, Beacon, frame_count, data_type, 1)

        # print("gt and ret", gt, ret)

        fluo_img = None
        fluo = False
        if(fluo == True):
            fluo_p = image_path + "fluo{0:0=6d}".format(frame_count) + ".jpg"
            fluo_img = cv2.imread(fluo_p, cv2.IMREAD_GRAYSCALE)

            fluo_img_x = fluo_img[0:crop_height, 0:crop_width]
            if (scale > 1):
                fluo_img_x = cv2.resize(fluo_img_x, (fluo_img_x.shape[1] * scale, fluo_img_x.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

            # fluo_img_x = np.where(fluo_img_x < 25, 0, fluo_img_x)# I have done this in preprocessing step. prepro_frames_2.det_ml()
            # fluo_img = fluo_img * 10
            frame = frame.astype(np.float16)
            frame = frame * 0.8
            frame[:, :, 0] += fluo_img_x * 3
            frame = np.clip(frame, 0, 255)
            frame = frame.astype(np.uint8)

        frame, frame_red = worker.mark_gt_3cat(frame, frame_count, scale, gt_frame, crop_height, crop_width, out_path, Beacon, gt and ret, get_cells, f_det_txt, class_f, fluo = fluo)
        # frame, frame_red = worker.mark_gt_3cat_death(frame, frame_count, scale, gt_frame, crop_height, crop_width, out_path, Beacon, gt and ret, get_cells, f_det_txt, class_f, fluo = fluo)
        # frame, frame_red = worker.mark_gt_3cat_death_rewrite(frame, frame_count, scale, gt_frame, crop_height, crop_width, out_path, Beacon, gt and ret, get_cells, f_det_txt, class_f, fluo = fluo)
        # ph_detector.mark_cells(frame, frame_count)

        # size = (crop_width * 8, crop_height * 8)
        size = (crop_width * 6, crop_height * 6)
        # size = (crop_width * 3, crop_height * 3)
        # size = (crop_width, crop_height)
        if(out2 is None):
            # out2 = cv2.VideoWriter(out_path + "cell_classified_" + time.strftime("%d_%H_%M", time.localtime()) + ".mp4",fourcc, 3.0, (crop_width * 3, crop_height * 3), isColor=True)
            out2 = cv2.VideoWriter(out_path + "videos_ucf/Beacon-" + str(Beacon) + "-classified.mp4",fourcc, 3.0, size, isColor=True)

        if out2:
            if(save_img):
                cv2.imwrite(out_path + "tmp/Beacon-" + str(Beacon) + "/mark_img_" + str(frame_count) + ".png", frame)

            frame_vid = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
            out2.write(frame_vid)
            cv2.imwrite(out_path + "videos_ucf/Classify_Beacon-" + str(Beacon) + "_" + str(frame_count) + ".png", frame_vid)

            if(gt and ret and save_img):
                frame_red = cv2.resize(frame_red, (crop_width * 8, crop_height * 8), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(out_path + "tmp/Beacon-" + str(Beacon) + "/imj_j" + str(frame_count) + ".png", frame_red)

    if(get_cells):
        f_det_txt.close()

    class_f.close()

    # cv2.namedWindow('Tracking',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Tracking', 900,900)
    # cv2.imshow('Tracking', frame)
    # cv2.waitKey()

    print("\n")

    print(__name__, "Done!")
    # cap2.release()
    if(out2):
        out2.release()
    cv2.destroyAllWindows()



# def preprocess(frame, save_image_path):
#
#     # cv2.namedWindow('preprocess_0', cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow('preprocess_0', 900, 900)
#     # cv2.imshow('preprocess_0', frame)
#
#     if (len(frame.shape) > 2):
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     else:
#         frame = frame
#
#     width = frame.shape[1]
#     height = frame.shape[0]
#     box_side = 64
#     vertical = int(height / box_side + 1)
#     horizontal = int(width / box_side + 1)
#
#     for i in range(vertical):
#         for j in range(horizontal):
#             start_x = box_side * j
#             start_y = box_side * i
#
#             end_x = 0
#             end_y = 0
#
#             if(i < vertical and j < horizontal):
#                 end_x = start_x + box_side
#                 end_y = start_y + box_side
#             elif(j ==  horizontal):
#                 end_x = 1328
#                 end_y = start_y + box_side
#             elif(i == vertical):
#                 end_x = start_x + box_side
#                 end_y = 1048
#
#
#             window = frame[start_y:end_y, start_x:end_x]
#
#             histSize = 256
#             histRange = (0, 256)  # the upper boundary is exclusive
#             accumulate = False
#             window_b_hist = cv2.calcHist([window], [0], None, [histSize], histRange, accumulate=accumulate)
#
#             window_hist_max_index = np.argmax(window_b_hist)
#
#             hist_max_index = window_hist_max_index
#             hist_trans = np.zeros(256, dtype=np.uint8)
#             target_peak = 100
#
#             for k in range(hist_max_index):
#                 hist_trans[k] = target_peak * k / hist_max_index + 0.5
#
#             for m in range(hist_max_index, 256):
#                 hist_trans[m] = ((255 - target_peak) * m - 255 * hist_max_index + target_peak * 255) / (255 - hist_max_index) + 0.5
#
#             window = hist_trans[window]
#             frame[start_y:end_y, start_x:end_x] = window
#
#     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))#clipLimit=2.0,
#     clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))#clipLimit=2.0,
#     # gray = clahe.apply(gray)
#     frame = clahe2.apply(frame)
#
#     cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Tracking', 900, 900)
#     cv2.imshow('Tracking', frame)
#     cv2.waitKey()
#
#     np.save(save_image_path, frame)
#     new_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#     return new_frame

def preprocess_2(frame, save_image_path, scale = 1):
    # cv2.namedWindow('preprocess_0', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('preprocess_0', 900, 900)
    # cv2.imshow('preprocess_0', frame)

    if (len(frame.shape) > 2):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_org = frame.copy()

    t0 = time.time()
    frame = cv2.medianBlur(frame_org, 81)#There is an unexpected effect when ksize as 81, applied to 8 times scaled image.

    frame = frame / 100
    frame = frame_org / frame
    np.clip(frame, 0, 255, out = frame)
    frame = frame.astype(np.uint8)

    if(scale > 1):
        frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation = cv2.INTER_CUBIC)

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


if __name__ == "__main__":
    # execute main
    # main(sys.argv[1], sys.argv[2])
    print(sys.argv[:])

    if(len(sys.argv[1:]) > 1000):
        main(*sys.argv[1:])
    else:
        home_dir = os.path.expanduser("~") + "/"
        # main(configure_path="./configure.txt", path=home_dir + "Work/ground_truth/preprocess", out_path=home_dir + "Work/ground_truth/test")
        # main(configure_path = "./configure.txt", path = home_dir + "disk_16t/Pt196/RawData/Beacon-153", out_path = home_dir + "disk_16t/Pt196/output/Beacon-153")
        # main(configure_path="./configure.txt", path="/home/qibing/disk_16t/Pt210/RawData/Beacon-73", out_path="Default")
        # main(configure_path="./configure.txt", path="/home/qibing/disk_16t/qibing/Pt298_SOCCO/RawData/Beacon-124", out_path="Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt174/RawData/Beacon-44", out_path = "Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt210/RawData/Beacon-73", out_path = "Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt180/RawData/Beacon-77", out_path = "Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/Pt180/RawData/Beacon-32", out_path = "Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/macrophage/Macrophage_VENDAR_Pt641/RawData/Beacon-147", out_path = "/home/qibing/disk_16t/qibing/output_ml/Macrophage_VENDAR_Pt641/")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/macrophage/Macrophage_VENDAR_Pt641/RawData/Beacon-147", out_path = "/home/qibing/disk_16t/qibing/output_ml_test/Macrophage_VENDAR_Pt641/")
        #/home/qibing/disk_16t/qibing/macrophage/Macrophage_VENDAR_Pt641/RawData/Beacon-12/ /home/qibing/disk_16t/qibing/output_ml/Macrophage_VENDAR_Pt641
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/macrophage/Macrophage_VENDAR_Pt641/RawData/Beacon-2/", out_path = "/home/qibing/disk_16t/qibing/output_ml/Macrophage_VENDAR_Pt641/")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/RawData/Beacon-61/", out_path = "Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/RawData/Beacon-3/", out_path = "Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/RawData/Beacon-51/", out_path = "Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/RawData/Beacon-99/", out_path = "Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/bad_exp/Pt352_SOCCO/RawData/Beacon-83/", out_path = "Default")

        # for beacon in ["3", "51", "99", ]:#, "73" "4", "52", "100"
        #     input_path = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/RawData/Beacon-" + str(beacon) + "/"
        #
        #     try:
        #         p = multiprocessing.Process(target=main, args=("./configure.txt", input_path, "Default"))
        #         p.start()
        #         print(time.strftime("%d_%H_%M ", time.localtime()), p, input_path)
        #
        #     except Exception as e:  # work on python 3.x
        #         print('Exception: ' + str(e))

        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/RawData/Beacon-11/", out_path = "Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/RawData/Beacon-49/", out_path = "Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/RawData/Beacon-53/", out_path = "Default")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt210/RawData/Beacon-1/", out_path = "/home/qibing/disk_16t/qibing/proj2_out/")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/RawData/Beacon-49/", out_path = "/home/qibing/data_by_gpu/Pt796/")
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt935_MACROPHAGE_plate1_08102022/RawData/Beacon-2/", out_path = "/home/qibing/data_by_gpu/Pt935/", fluo=True)

        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/RawData/Beacon-88/", out_path = "/home/qibing/data_by_gpu/Pt796_error/")


        # a = range(1, 6, 1)
        # b = range(11, 16, 1)
        # # c = range(49, 54, 1)
        # # d = range(59, 64, 1)
        # e = range(88, 93, 1)
        # f = range(40, 45, 1)
        # g = range(136, 141, 1)
        # # # for beacon in chain(d, c):
        # # for beacon in [99, 14, 52, 100, 15, 53, 101]:#
        # # for beacon in chain(a, b):#for pt196
        # # for beacon in chain(e,):#for Pt796
        # for beacon in chain(g,):#for Pt796
        # # for beacon in range(1, 4):
        # # for beacon in [11, 49, 97, 12, 50, 98, 13, 51, 99, 14, 52, 100, 15, 53, 101]:#
        # #/home/qibing/data/Pt796_MO_CD138_03102022/RawData/Beacon-49
        # # for beacon in [49,]:
        # #     input_path = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/RawData/Beacon-" + str(beacon) + "/"
        #     # input_path = "/home/qibing/data/Pt796_MO_CD138_03102022/RawData/Beacon-" + str(beacon) + "/"
        #     # input_path = "/home/qibing/disk_16t/qibing/Pt196/RawData/Beacon-" + str(beacon) + "/"
        #
        #     # input_path = "/home/qibing/data/pt196_alignment/images_align/Beacon_" + str(beacon) + "/"
        #     # main(configure_path = "./configure.txt", path = input_path, out_path = "/home/qibing/data/pt196_alignment/")
        #
        #     input_path = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/RawData/Beacon-" + str(beacon) + "/"
        #     main(configure_path="./configure.txt", path=input_path, out_path="/home/qibing/data/Pt796/")

        # try:
            #     p = multiprocessing.Process(target=main, args=("./configure.txt", input_path, "Default"))
            #     p.start()
            #     print(time.strftime("%d_%H_%M ", time.localtime()), p, input_path)
            #
            # except Exception as e:  # work on python 3.x
            #     print('Exception: ' + str(e))

        # main(configure_path="./configure.txt", path= '/home/qibing/disk_16t/qibing/Pt211/TimeLapseVideos/Beacon_31/imgs/', out_path = "/home/qibing/disk_16t/qibing/Pt211/TimeLapseVideos/Beacon_31/", fluo=False)
        # input_path = "/home/qibing/disk_16t/qibing/Pt796_MO_CD138_03102022/RawData/Beacon-97/"
        # main(configure_path="./configure.txt", path=input_path, out_path="/home/qibing/data/Pt796_poster/")

        # input_path = "/home/qibing/disk_16t/qibing/Pt935_MACROPHAGE_plate1_08102022/RawData/Beacon-79/"
        # main(configure_path="./configure.txt", path=input_path, out_path="/home/qibing/data/Pt935_20230216/", fluo=True)

        # input_path = "/home/qibing/disk_16t/qibing/Pt935_MACROPHAGE_plate2_08102022/RawData/Beacon-338/"
        # main(configure_path="./configure.txt", path=input_path, out_path="/home/qibing/data/Pt935_20230216/", fluo=False)

        # input_path = "/home/qibing/disk_16t/qibing/Pt935_MACROPHAGE_plate2_08102022/RawData/Beacon-337/"
        # main(configure_path="./configure.txt", path=input_path, out_path="/home/qibing/data/Pt935_20230216_rm_blur/", fluo=True)

        # input_path = "/home/qibing/disk_16t/qibing/Pt935_MACROPHAGE_plate2_08102022/RawData/Beacon-347/"
        # main(configure_path="./configure.txt", path=input_path, out_path="/home/qibing/data/Pt935_20230216_rm_blur/", fluo=True)



        # # a = range(1, 6, 1)
        # b = range(11, 16, 1)
        # b = range(15, 16, 1)
        #
        # c = range(337, 342, 1)
        # c = range(340, 342, 1)
        # # d = range(347, 352, 1)

        # test_beacons = [5, 15, 341, 351,] + list(range(49, 54, 1)) + list(range(59, 64, 1))
        # test_beacons = [5, 15, ]
        #
        # # for beacon in chain(b):
        # for beacon in test_beacons:
        #     # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt935_MACROPHAGE_plate1_08102022/RawData/Beacon-" + str(beacon) + "/", out_path = "/home/qibing/data_by_gpu/Pt935/", fluo=True)
        #     # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt935_MACROPHAGE_plate2_08102022/RawData/Beacon-" + str(beacon) + "/", out_path = "/home/qibing/data_by_gpu/Pt935/", fluo=True)
        #     main(configure_path = "./configure.txt", path = "/data/qibing/Pt935_MACROPHAGE_plate1_08102022/RawData/Beacon-" + str(beacon) + "/", out_path = "/home/qibing/data_by_gpu/Pt935_eval/", fluo=True)

        # test_beacons = [351,]#341,
        # for beacon in test_beacons:
        #     main(configure_path = "./configure.txt", path = "/data/qibing/Pt935_MACROPHAGE_plate2_08102022/RawData/Beacon-" + str(beacon) + "/", out_path = "/home/qibing/data_by_gpu/Pt935_eval/", fluo=True)

        # test_beacons = list(range(50, 54, 1)) + list(range(59, 64, 1))
        # test_beacons = [51, 59] #50, 51, 59
        # for beacon in test_beacons:
        #     main(configure_path = "./configure.txt", path = "/data/qibing/Pt935_MACROPHAGE_plate1_08102022/RawData/Beacon-" + str(beacon) + "/", out_path = "/home/qibing/data_by_gpu/Pt935_fluo/", fluo=True)

        # test_beacons = list(range(78, 83, 1))
        # # test_beacons = [78,]
        # for beacon in test_beacons:
        #     main(configure_path = "./configure.txt", path = "/data/qibing/Pt935_MACROPHAGE_plate1_08102022/RawData/Beacon-" + str(beacon) + "/", out_path = "/home/qibing/data_by_gpu/Pt935_fluo/", fluo=True)
        # main(configure_path = "./configure.txt", path = sys.argv[1], out_path = "/home/qibing/data_by_gpu/Pt935_fluo/", fluo=True)

        # main(configure_path = "./configure.txt", path = sys.argv[1], out_path = "/home/qibing/data_by_gpu/Pt935_fluo_2lstm_v4/", fluo=True)

        # main(configure_path = "./configure.txt", path = "/home/qibing/data_by_gpu/Pt935_fluo_2lstm_v4_Beacon_92_266_180_1092_838/RawData/Beacon-92/", out_path = "/home/qibing/data_by_gpu/Pt935_fluo_2lstm_v4_Beacon_92_266_180_1092_838/", fluo=True)
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt935_MACROPHAGE_plate1_08102022/RawData/Beacon-92/", out_path = '/home/qibing/data_by_gpu/Pt935_fluo_2lstm_v4_Beacon_92_detect_more/', fluo=True)
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt935_MACROPHAGE_plate2_08102022/RawData/Beacon-337/", out_path = '/home/qibing/data_by_gpu/proj2_death_prediction/', fluo=True)
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt935_MACROPHAGE_plate1_08102022/RawData/Beacon-59/", out_path = '/home/qibing/data_by_gpu/proj2_death_prediction/', fluo=True)
        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt204/RawData/Beacon-131/", out_path = '/home/qibing/data_by_gpu/proj2_death_prediction/', fluo=False)

        # print("qibing: ", sys.argv[1], sys.argv[2])
        # main(configure_path = "./configure.txt", path = sys.argv[1], out_path = sys.argv[2], fluo=True)

        # main(configure_path = "./configure.txt", path = "/home/qibing/disk_16t/qibing/Pt935_MACROPHAGE_plate1_08102022/RawData/Beacon-380", out_path = "/home/qibing/data_by_gpu/proj2_death_prediction/Pt935_test/", fluo=True)



        # # I cannot use multiprocessing and tensorflow at the same time.
        # test_beacons = list(range(78, 83, 1))
        # for beacon in test_beacons:
        #     configure_path = "./configure.txt"
        #     input_path = "/data/qibing/Pt935_MACROPHAGE_plate1_08102022/RawData/Beacon-" + str(beacon) + "/"
        #     output_path = "/home/qibing/data_by_gpu/Pt935_fluo/"
        #     fluo = True
        #
        #     p = multiprocessing.Process(target=main, args=(configure_path, input_path, output_path, fluo))
        #     p.start()
        #     print(p, input_path)
        #     p.join()

# Remember to update lstm model
# Remember to update lstm model
# Remember to update lstm model

        main(configure_path = "./configure.txt", path = sys.argv[1], out_path = sys.argv[2], fluo=True)
