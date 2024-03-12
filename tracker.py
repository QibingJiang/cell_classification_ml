# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np

# from . import kalman_filter
# from . import linear_assignment
# from . import iou_matching
# from . import euclidean_distance
# from .track import Track

import kalman_filter
import linear_assignment
import iou_matching
import euclidean_distance
from track import Track


import cv2
import os
from util import read_frame
import tensorflow as tf



# from tensorflow_docs.vis import embed
from tensorflow import keras
# from imutils import paths

# import matplotlib.pyplot as plt
import tensorflow as tf
# import pandas as pd
import numpy as np
# import imageio
import cv2
import os

"""
## Define hyperparameters
"""

# IMG_SIZE = 224
IMG_SIZE = 96
BATCH_SIZE = 64
EPOCHS = 10
MAX_SEQ_LENGTH = 100
NUM_FEATURES = 2048




array_size = 800


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

feature_extractor = build_feature_extractor()

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

def frames_prediction(v_classify, frames):
    class_vocab = ['macrophage', 'myeloma']

    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = v_classify.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return class_vocab[0] if probabilities[0] > probabilities[1] else class_vocab[1]


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_euclidean_distance = 100
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self.del_tracks = []
        self._next_id = 1
        self.seq_info = None

        # self.coord = np.zeros(array_size * 2, dtype=float)
        # self.coord[:] = np.nan
        self.f_state = None
        self.background_pixel = 0
        self.cell_core_r = 0
        self.cell_core_r_mean = 0
        self.image_amount = 0



    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        # print("tracker predict")
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, frame_idx, frame, image_amount):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # print("tracker update")
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections, frame_idx)

        # Update track set.
        # frame = cv2.imread(self.seq_info["image_filenames"][frame_idx], cv2.IMREAD_GRAYSCALE)
        frame_org = cv2.resize(frame, (frame.shape[1] * 8, frame.shape[0] * 8), interpolation=cv2.INTER_CUBIC)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx], frame_idx, frame_org)
        for track_idx in unmatched_tracks:
            # self.tracks[track_idx].mark_missed(frame_idx)
            pass
        for detection_idx in unmatched_detections:
            # if (detections[detection_idx].score > 0.9 and detections[detection_idx].max_pixel > 150):#detections[detection_idx].area < 800 or
            self._initiate_track(detections[detection_idx], frame_idx, frame_org, image_amount)

        for i in range(len(self.tracks)):
            if(self.tracks[i].is_deleted()):
                self.del_tracks.append(self.tracks[i])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        # active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

        scale = 6
        frame1 = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        # frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)

        cnt = 0
        for t in self.tracks:
            # print(t)
            x = t.coord[frame_idx][0]
            y = t.coord[frame_idx][1]
            # tra_scores = t.score.sum()  and tra_scores > 10
            if(x != 0 and y != 0):
                cv2.circle(frame1, (int(x * scale), int(y * scale)), 5 * scale, (255, 255, 0), ((1 * scale) >> 1))
                cv2.putText(frame1, str(t.track_id), (int(x + 6) * scale, int(y + 3) * scale),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), ((1 * scale) >> 1))

                cnt += 1

        # cv2.imshow("tra", frame1)
        # cv2.waitKey()
        # cv2.putText(frame1, str(frame_idx), (5*scale, 10*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (255, 255, 255), int(0.3 * scale))
        cv2.putText(frame1, str(frame_idx) + " " + str(cnt), (5*scale, 10*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (255, 255, 255), int(0.3 * scale))
        return frame1

    def _match(self, detections, frame_idx):

        def gated_metric(tracks, dets, track_indices, detection_indices, frame_idx):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # ids = [t.track_id for t in self.tracks]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                # self.tracks, detections, confirmed_tracks, frame_idx=frame_idx, max_eu_cost = 2)
                self.tracks, detections, track_indices=None, frame_idx = frame_idx, max_eu_cost = 2)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        # iou_track_candidates = unconfirmed_tracks + [
        #     k for k in unmatched_tracks_a if
        #     self.tracks[k].time_since_update == 1]
        # unmatched_tracks_a = [
        #     k for k in unmatched_tracks_a if
        #     self.tracks[k].time_since_update != 1]

        # matches_b, unmatched_tracks_b, unmatched_detections = \
        #     linear_assignment.min_cost_matching(
        #         iou_matching.iou_cost, self.max_iou_distance, self.tracks,
        #         detections, iou_track_candidates, unmatched_detections)

        # Associate remaining tracks together with unconfirmed tracks using Euclidean distance.
        # matches_b, unmatched_tracks_b, unmatched_detections = \
        #     linear_assignment.min_cost_matching(
        #         euclidean_distance.cost, self.max_euclidean_distance, self.tracks,
        #         detections, iou_track_candidates, unmatched_detections)

        # V2
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.matching_cascade(
                euclidean_distance.cost, self.max_euclidean_distance, self.max_age,
                # self.tracks, detections, iou_track_candidates, unmatched_detections, frame_idx=frame_idx, max_eu_cost = 3)
                self.tracks, detections, unmatched_tracks_a, unmatched_detections, frame_idx = frame_idx, max_eu_cost = 3)

        matches_c, unmatched_tracks_c, unmatched_detections = \
            linear_assignment.nearestNeighbor(
                euclidean_distance.cost, self.max_euclidean_distance, self.max_age,
                self.tracks, detections, unmatched_tracks_b, unmatched_detections, frame_idx=frame_idx)


        matches = matches_a + matches_b + matches_c
        # print(frame_idx, matches_a, matches_b, matches_c)
        # l_0 = [self.tracks[m[0]].track_id for m in matches_a]
        # l_1 = [self.tracks[m[0]].track_id for m in matches_b]
        # l_2 = [self.tracks[m[0]].track_id for m in matches_c]
        # print(l_0, l_1, l_2)
        # unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b + unmatched_tracks_c))
        unmatched_tracks = list(set(unmatched_tracks_c))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, frame_idx, frame_org, image_amount):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, detection.tlwh, frame_idx, frame_org, detection.area,img_amount = image_amount, cell_d = detection))
        # tlwh = np.array([detection.horizontal_x - 9, detection.vertical_y - 9, 18, 18])
        # self.tracks.append(Track(
        #     mean, covariance, self._next_id, self.n_init, self.max_age,
        #     detection.feature, tlwh, frame_idx, frame_org, detection.area))

        self._next_id += 1

    # max - min
    def analyse_classification_8(self, outpath, frame_count, gt_video_path, scale, Beacon, gt = False):
        # print("tracker save.")
        self.image_amount = frame_count
        window_radius = 4

        file = None
        # gt = False

        # make it comment
        file = open(outpath + "Misc/info_ucf/Beacon_" + str(Beacon) + "_tracks_death_log" + ".txt", "w")
        file.write("track_id, image_index, cell_diff, cell_diff_max_min, cell_diff_max_min_der, area, area_diff, area_diff_max_min, area_diff_max_min_der, cell_death\n")

        live_dead_table = np.zeros((self.image_amount, 4))
        # live_dead_table = np.zeros((self.image_amount, 4), dtype=int)

        # for tracks in (self.tracks, self.del_tracks):
        #     count_tmp = 0
        #     i = 0
        #     while (i < len(tracks)):
        #         count_tmp += 1
        #         if np.count_nonzero(tracks[i].area > 0) < 20:
        #             tracks.remove(tracks[i])
        #         else:
        #             i += 1
        #     # print(count_tmp, len(tracks))

        diff_thr = 527665
        # area_thr = 32
        # area_thr = 32/64
        area_thr = 0.5


        # diff_thr = 170000
        # area_thr = 35

        # diff_thr = 212.7 * self.cell_core_r_mean * self.cell_core_r_mean * self.background_pixel
        # area_thr = 4.42 * self.cell_core_r_mean * self.cell_core_r_mean

        # print("diff_thr, area_thr: ", self.cell_core_r, self.background_pixel, diff_thr, area_thr)
        for tracks in (self.tracks, self.del_tracks):
            for tra_i in range(0, len(tracks), 1):
                if (file):
                    track_mat = np.zeros((self.image_amount, 10), dtype = float)
                    track_mat[:,0] = tracks[tra_i].track_id
                    track_mat[:,1] = np.arange(self.image_amount)
                    track_mat[:,2] = tracks[tra_i].cell_diff[:self.image_amount]

                cell_diff_row = tracks[tra_i].cell_diff.copy()
                #################################################################### start

                tmp = np.where(cell_diff_row[:] > 3 * diff_thr)
                max_v = np.nan
                min_v = np.nan
                diff_mm = np.zeros_like(cell_diff_row)
                diff_mm_der = np.zeros_like(cell_diff_row)
                diff_mm[:] = np.nan
                diff_mm_der[:] = np.nan

                if (len(tmp[0]) == 0):
                    loc = 0
                else:

                    for i in range(self.image_amount - 1, -1, -1):
                        if (cell_diff_row[i] > 0):
                            if (np.isnan(max_v) or cell_diff_row[i] > max_v):
                                max_v = cell_diff_row[i]

                            if (np.isnan(min_v) or cell_diff_row[i] < min_v):
                                min_v = cell_diff_row[i]

                            diff_mm[i] = max_v - min_v

                    diff_mm_der[1:] = np.abs(np.diff(diff_mm))
                    loc = np.nan
                    try:
                        loc_tmp = np.nanargmax(diff_mm_der)
                        # print("cell_diff_max_drop", diff_mm_der[loc_tmp])
                        if(diff_mm_der[loc_tmp] > 1):#death_max_in_win
                            loc = loc_tmp
                        else:
                            loc = 100000
                    except ValueError as e:
                        if (e.args[0] == 'All-NaN slice encountered'):
                            pass
                        else:
                            print("Qibing error: ", e)
                            exit()

                #################################################################### end

                # print("qibing: ", cell_diff_row, diff_mm, diff_mm_der, loc)

                if (file):
                    track_mat[:,3] = diff_mm[:self.image_amount]
                    track_mat[:,4] = diff_mm_der[:self.image_amount]
                    track_mat[:,5] =tracks[tra_i].area[:self.image_amount]

                area_diff = np.zeros(array_size)
                area_diff[1:] = np.abs(np.diff(tracks[tra_i].area))

                if (file):
                    track_mat[:,6] = area_diff[:self.image_amount]


                #################################################################### start
                tmp = np.where(area_diff[:] > 3 * area_thr)

                max_v = np.nan
                min_v = np.nan
                area_diff_mm = np.zeros_like(area_diff)
                area_diff_mm[:] = np.nan
                area_diff_mm_der = np.zeros_like(area_diff)
                area_diff_mm_der[:] = np.nan

                if (len(tmp[0]) == 0):
                    loc_area = 0
                else:
                    for i in range(self.image_amount - 1, -1, -1):
                        if (area_diff[i] > 0):
                            if (np.isnan(max_v) or area_diff[i] > max_v):
                                max_v = area_diff[i]

                            if (np.isnan(min_v) or area_diff[i] < min_v):
                                min_v = area_diff[i]

                            area_diff_mm[i] = max_v - min_v

                    area_diff_mm_der[1:] = np.abs(np.diff(area_diff_mm))
                    loc_area = np.nan
                    try:
                        loc_area_tmp = np.nanargmax(area_diff_mm_der)
                        # print("area_max_drop", area_diff_mm_der[loc_area_tmp])
                        if(area_diff_mm_der[loc_area_tmp] > 1): #600
                            loc_area = loc_area_tmp
                        else:
                            loc_area = 100000


                    except ValueError as e:
                        if (e.args[0] == 'All-NaN slice encountered'):
                            pass
                        else:
                            print("error: ", e)
                            exit()

                if (loc_area < loc):
                    loc = loc_area

                #################################################################### end

                if (file):
                    track_mat[:,7] = area_diff_mm[:self.image_amount]
                    track_mat[:,8] = area_diff_mm_der[:self.image_amount]


                one_track = np.zeros(array_size)
                if (0 < loc < (self.image_amount)):
                    one_track[:loc] = 1
                    one_track[loc:] = 0
                elif (loc > (self.image_amount)):
                    one_track[:] = 1
                else:
                    one_track[:] = 0

                if (file):
                    track_mat[:,9] = one_track[:self.image_amount]

                [print(*tra_e, sep = ',', file = file) for tra_e in track_mat]
                tracks[tra_i].live_state = one_track * 2  # just covert 1 to 2

        if (file):
            file.close()

        # live_area = np.zeros(self.image_amount)
        # # tmp_live_dead_table = np.zeros(self.image_amount)
        # for tracks in (self.tracks, self.del_tracks):
        #     for i in range(0, len(tracks), 1):
                # y = tracks[i].area

                # sub_y_idx = np.where(y > 0)[0]
                # y_p = y[sub_y_idx]
                # x = np.arange(sub_y_idx[0], sub_y_idx[-1], 1)
                # part_new_y = np.interp(x, sub_y_idx, y_p)
                # new_y = y.copy()
                # new_y[sub_y_idx[0]:sub_y_idx[-1]] = part_new_y
                # tracks[i].area = new_y

                # for j in range(self.image_amount):
                #     if (tracks[i].live_state[j] > 1):
                #         if (tracks[i].area[j] > 0):
                #             live_area[j] += tracks[i].area[j]
                #             # live_area[j] += new_y[j]
                #     else:
                #         break

        for j in range(0, self.image_amount):
            for tracks in (self.tracks, self.del_tracks):
                for i in range(0, len(tracks), 1):
                    # cell_x = tracks[i].coordinates[j][0]
                    # cell_y = tracks[i].coordinates[j][1]
                    if (tracks[i].type == "myeloma" and tracks[i].area[j] > 0):#tracks[i].type == "myeloma" and

                        if (tracks[i].live_state[j] > 1):
                            live_dead_table[j][0] = live_dead_table[j][0] + 1
                            live_dead_table[j][2] = live_dead_table[j][2] + tracks[i].area[j]
                        else:
                            live_dead_table[j][1] = live_dead_table[j][1] + 1
                            live_dead_table[j][3] = live_dead_table[j][3] + tracks[i].area[j]

                # live_dead_table[j][2] = live_dead_table[j][0] + live_dead_table[j][1]

        if (not os.path.exists(outpath)):
            os.makedirs(outpath)

        # np.savetxt(outpath + "live_dead_table_" + str(window_radius) + "_" + str(diff_thr) + "_" + str(area_thr) + time.strftime("%d_%H_%M", time.localtime()) + ".txt", live_dead_table, fmt='%d')
        np.savetxt(outpath + "Misc/info_ucf/Beacon_" + str(Beacon) + "_live_dead_table.txt", live_dead_table, fmt='%d')  # , fmt='%d'

        with open(outpath + "Misc/Results_ucf/Results_" + "{0:0=3d}".format(Beacon) + "_old.csv", 'w') as f:
            f.write("Beacon-" + "{0:0=3d}".format(Beacon) + ',')
            print(*live_dead_table[:, 2], sep=',', file=f)

        # with open(outpath + "Results/Results_" + "{0:0=3d}".format(Beacon) + ".csv", 'w') as f:
        #     f.write("Beacon-" + "{0:0=3d}".format(Beacon) + ',')
        #     print(*live_area, sep=',', file=f)#It is the padded area


    # max in window
    def analyse_classification_5(self, outpath, frame_count, gt_video_path, scale, Beacon, gt = False):
        # print("tracker save.")
        self.image_amount = frame_count
        window_radius = 4

        file = None
        gt = True

        # file = open(outpath + "file3_" + time.strftime("%d_%H_%M", time.localtime()) + ".txt", "w")

        live_dead_table = np.zeros((self.image_amount, 4))
        # live_dead_table = np.zeros((self.image_amount, 4), dtype=int)

        for tracks in (self.tracks, self.del_tracks):
            count_tmp = 0
            i = 0
            while(i < len(tracks)):
                count_tmp += 1
                if np.count_nonzero(tracks[i].area > 0) < 20:
                    tracks.remove(tracks[i])
                else:
                    i += 1
            # print(count_tmp, len(tracks))

        # print()
        # diff_thr = 7200000
        # area_thr = 960
        # diff_thr = 500000
        # area_thr = 30

        # diff_thr = 737.52 * self.cell_core_r * self.cell_core_r * self.background_pixel
        # area_thr = 4.43 * self.cell_core_r * self.cell_core_r

        # diff_thr = 636.17 * self.cell_core_r_mean * self.cell_core_r_mean * self.background_pixel
        # area_thr = 3.82 * self.cell_core_r_mean * self.cell_core_r_mean

        diff_thr = 527665
        area_thr = 32


        # print("diff_thr, area_thr: ", self.cell_core_r, self.background_pixel, diff_thr, area_thr)
        for tracks in (self.tracks, self.del_tracks):
            for tra_i in range(0, len(tracks), 1):

                # if(np.count_nonzero(tracks[tra_i].coordinates > 0) < 38):
                #     del tracks[tra_i]

                if(file):
                    file.write("track: %d;\n" % tracks[tra_i].track_id)
                    file.write("cell diff: ")
                    for j in range(array_size):
                        file.write("%s, " % tracks[tra_i].cell_diff[j])
                    file.write(";\n")

                cell_diff_row = np.zeros_like(tracks[tra_i].cell_diff)
                cell_diff_row[:] = np.nan

                for j in range(window_radius, self.image_amount):
                    temp_array = tracks[tra_i].cell_diff[j - window_radius: j + window_radius + 1]
                    temp_array = temp_array[~np.isnan(temp_array)]

                    if (len(temp_array) > window_radius * 1.5):
                        local_max = np.max(temp_array)
                        cell_diff_row[j] = local_max
                    else:
                        pass

                if (file):
                    file.write("cell diff 2: ")
                    for j in range(array_size):
                        file.write("%s, " % cell_diff_row[j])
                    file.write(";\n")

                cell_diff_row = np.where(cell_diff_row < diff_thr, 0, cell_diff_row)

                if (file):
                    file.write("cell diff 3: ")
                    for j in range(array_size):
                        file.write("%s, " % cell_diff_row[j])
                    file.write(";\n")

                    file.write("area: ")
                    for j in range(array_size):
                        file.write("%s, " % tracks[tra_i].area[j])
                    file.write(";\n")

                area_row = np.zeros_like(tracks[tra_i].area)
                area_row[:] = np.nan

                area_diff = np.diff(tracks[tra_i].area)
                area_diff = np.absolute(area_diff)

                for j in range(window_radius, self.image_amount):
                    temp_array = area_diff[j - window_radius: j + window_radius + 1]
                    temp_array = temp_array[~np.isnan(temp_array)]

                    if (len(temp_array) > window_radius * 1.5):
                        local_max = np.max(temp_array)
                        area_row[j] = local_max
                    else:
                        pass

                if (file):
                    file.write("area_2: ")
                    for j in range(array_size):
                        file.write("%s, " % area_row[j])
                    file.write(";\n")

                area_row = np.where(area_row < area_thr, 0, area_row)

                if (file):
                    file.write("area_3: ")
                    for j in range(array_size):
                        file.write("%s, " % area_row[j])
                    file.write(";\n")

                for arr in [cell_diff_row, area_row]:
                    flag = 0
                    for j in range(len(arr)):
                        if arr[j] == 0:# find the death 0
                            # print(j, arr)
                            arr[j:] = 0
                            # arr[:j] = 1
                            k = j
                            while k >= 0:
                                if(arr[k] > 0):
                                    # print("qibing test 0: ", arr)
                                    break
                                else:
                                    k = k - 1

                            if k == -1:
                                arr[:] = 0
                                # print("qibing test 3: ", arr)
                            else:
                                # print("qibing test 1: ", arr)
                                arr[:k + 1] = 1
                                arr[k + 1:] = 0
                                # print("qibing test 2: ", arr)

                            flag = 1
                            break

                    if flag == 0:
                        if arr.size - np.count_nonzero(np.isnan(arr)) == 0:
                            # print("qibing test: ", arr)
                            arr[:] = 0
                        else:
                            arr[:] = 1 # There are some cells rarely show up, and cannot find death 0. fall into this category.

                if (file):
                    file.write("cell diff result: ")
                    for j in range(array_size):
                        file.write("%s, " % cell_diff_row[j])
                    file.write(";\n")

                    file.write("area result: ")
                    for j in range(array_size):
                        file.write("%s, " % area_row[j])
                    file.write(";\n")


                one_track = cell_diff_row + area_row
                # one_track = area_row * 2

                if (file):
                    file.write("final: ")
                    for s in range(array_size):
                        file.write("%s, " % one_track[s])
                    file.write(";\n")

                tracks[tra_i].live_state = one_track

        if (file):
            file.close()

        live_area = np.zeros(self.image_amount)
        # tmp_live_dead_table = np.zeros(self.image_amount)
        for tracks in (self.tracks, self.del_tracks):
            for i in range(0, len(tracks), 1):
                y = tracks[i].area

                sub_y_idx = np.where(y > 0)[0]
                y_p = y[sub_y_idx]
                x = np.arange(sub_y_idx[0], sub_y_idx[-1], 1)
                part_new_y = np.interp(x, sub_y_idx, y_p)
                new_y = y.copy()
                new_y[sub_y_idx[0]:sub_y_idx[-1]] = part_new_y

                tracks[i].area = new_y
                for j in range(self.image_amount):
                    if(tracks[i].live_state[j] > 1):
                        if(tracks[i].area[j] > 0):
                            live_area[j] += tracks[i].area[j]
                            # live_area[j] += new_y[j]
                    else:
                        break


        for j in range(0, self.image_amount):
            for tracks in (self.tracks, self.del_tracks):
                for i in range(0, len(tracks), 1):
                    if (tracks[i].area[j] > 0): # tracks[i].type == "myeloma" and

                        if (tracks[i].live_state[j] > 1):
                            live_dead_table[j][0] = live_dead_table[j][0] + 1
                            live_dead_table[j][2] = live_dead_table[j][2] + tracks[i].area[j]
                        else:
                            live_dead_table[j][1] = live_dead_table[j][1] + 1
                            live_dead_table[j][3] = live_dead_table[j][3] + tracks[i].area[j]


                # live_dead_table[j][2] = live_dead_table[j][0] + live_dead_table[j][1]


        if (not os.path.exists(outpath)):
            os.makedirs(outpath)

        # np.savetxt(outpath + "live_dead_table_" + str(window_radius) + "_" + str(diff_thr) + "_" + str(area_thr) + time.strftime("%d_%H_%M", time.localtime()) + ".txt", live_dead_table, fmt='%d')
        np.savetxt(outpath + "/info_ucf/Beacon_" + str(Beacon) + "_live_dead_table.txt", live_dead_table)#, fmt='%d'

        with open(outpath + "/Results_ucf/Results_" + "{0:0=3d}".format(Beacon) + ".csv", 'w') as f:
            f.write("Beacon-" + "{0:0=3d}".format(Beacon) + ',')
            print(*live_dead_table[:, 2], sep=',', file = f)

        with open(outpath + "/Results_ucf/Results_pad_area_" + "{0:0=3d}".format(Beacon) + ".csv", 'w') as f:
            f.write("Beacon-" + "{0:0=3d}".format(Beacon) + ',')
            print(*live_area, sep=',', file = f)



        # f_cell_tracks = open(outpath + "info_ucf/Beacon_" + str(Beacon) + "_cell_tracks.txt", 'w')
        # for tracks in (self.tracks, self.del_tracks):
        #     for idx in range(len(tracks)):
        #         print("track_id:", tracks[idx].track_id, sep='', file=f_cell_tracks)
        #         for i in range(self.image_amount):
        #             if(tracks[idx].full_trace[i] != None):
        #                 # print(*(tracks[idx].full_trace[i][[3, 10, 11]]), sep=',', file=f_cell_tracks)
        #                 print("%.2f,%.2f,%.2f,%.2f,%.2f"%tuple(tracks[idx].full_trace[i][[0, 1, 3, 10, 11]]), sep=',', file=f_cell_tracks)
        #             else:
        #                 print(np.nan, np.nan, np.nan, sep=',', file=f_cell_tracks)
        # f_cell_tracks.close()

    def analyse_classification_tmp(self, outpath, frame_count, gt_video_path, scale, Beacon, gt = False):
        # print("analyse_classification_tmp")

        # v_classify = tf.keras.models.load_model("v_classify.h5")
        # v_classify = tf.keras.models.load_model("v_classify.h5", compile=False)
        # v_classify.compile()

        cat_1 = ['macrophage', 'myeloma']
        lstm_l1 = tf.keras.models.load_model('/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea/v_classify_0.h5', compile=False)
        lstm_l1.compile()

        cat_2 = ['monocyte', 'myeloma']
        # lstm_l2 = tf.keras.models.load_model('/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea_2nd_layer/v_classify_0.h5', compile=False)
        lstm_l2 = tf.keras.models.load_model('/home/qibing/work_ssd/work/cell_ml_bgr_b/train_pt935_v7_7bea_2nd_layer_with_blurred_bea/v_classify_0.h5', compile=False)
        lstm_l2.compile()

        path_mv = os.path.join(outpath + "Misc/info_ucf/", "Beacon_" + str(Beacon))
        os.makedirs(path_mv, exist_ok=True)
        f_id = open(os.path.join(path_mv, "track_id.txt"), "w")
        f = open(os.path.join(path_mv, "track_motion.txt"), "w")

        for t_i, tracks in enumerate([self.del_tracks, self.tracks]):
            for i in range(len(tracks)):
                print("Cell Track classification: ", i, len(tracks), len(self.del_tracks), len(self.tracks))
                x_s_diff = np.diff(tracks[i].tlwh_s[:, 0])
                y_s_diff = np.diff(tracks[i].tlwh_s[:, 1])
                motion_vector = np.sqrt(x_s_diff * x_s_diff + y_s_diff * y_s_diff)
                tracks[i].motion_std = np.nanstd(motion_vector)
                f_id.write(str(tracks[i].track_id) + " ")
                print(*motion_vector, file=f)

                tmp_score = tracks[i].score.copy()
                for k in range(len(tmp_score) - 1, -1, -1):
                    if(tmp_score[k] <= 0.9):
                        tmp_score[k] = np.nan
                    if(tmp_score[k] > 0.9):
                        break
                zeros = np.count_nonzero(tmp_score <= 0.9)
                ones = np.count_nonzero(tmp_score > 0.9)
                if(ones == 0 or (float(zeros)/float(ones) > 0.5 and zeros > 5)):
                    tracks[i].good = False

                frames = []
                for c_i, c in enumerate(tracks[i].full_trace):
                    if(c):
                        frames.append(c.img)# c.img is 96*96*3
                    if(c_i > 96 and c):
                        tracks[i].fluo = tracks[i].fluo or c.fluo
                        # print(c_i, c.fluo)
                # print("tracks[i].fluo", tracks[i].fluo)

                if(len(frames) > 0):
                    # tracks[i].type = frames_prediction(v_classify, np.array(frames))
                    # tracks[i].type = "macrophage"

                    print(len(frames))

                    frame_features, frame_mask = prepare_single_video(np.array(frames))

                    prob_1 = lstm_l1.predict([frame_features, frame_mask])[0]
                    tracks[i].type = cat_1[np.argmax(prob_1)]
                    if(tracks[i].type != "macrophage"):# I should not call it myeloma, accuratley this is myeloma or monocyte.
                        prob_2 = lstm_l2.predict([frame_features, frame_mask])[0]
                        tracks[i].type = cat_2[np.argmax(prob_2)]


        # live_dead_table = np.zeros((self.image_amount, 4))
        #
        # for j in range(0, self.image_amount):
        #     for tracks in (self.tracks, self.del_tracks):
        #         for i in range(0, len(tracks), 1):
        #             # if (tracks[i].area[j] > 0):#tracks[i].type == "myeloma" and
        #             if (not (np.isnan(tracks[i].score[j])) ):#tracks[i].type == "myeloma" and and tracks[i].good == True
        #                 if (tracks[i].live_state[j] > 1):
        #                     live_dead_table[j][0] = live_dead_table[j][0] + 1
        #                     # live_dead_table[j][2] = live_dead_table[j][2] + tracks[i].area[j]
        #                     live_dead_table[j][2] = live_dead_table[j][2] + tracks[i].full_trace[j].area
        #                 else:
        #                     live_dead_table[j][1] = live_dead_table[j][1] + 1
        #                     live_dead_table[j][3] = live_dead_table[j][3] + tracks[i].full_trace[j].area
        #
        # np.savetxt(outpath + "Misc/info_ucf/Beacon_" + str(Beacon) + "_live_dead_table.txt", live_dead_table, fmt='%d')  # , fmt='%d'

        category_tab = np.zeros((self.image_amount, 6))

        for j in range(0, self.image_amount):
            for tracks in (self.tracks, self.del_tracks):
                for i in range(0, len(tracks), 1):
                    # if (tracks[i].area[j] > 0):#tracks[i].type == "myeloma" and
                    # if (not np.isnan(tracks[i].score[j])):#tracks[i].type == "myeloma" and
                    if(tracks[i].full_trace[j] != None):
                        if (tracks[i].type == "macrophage"):
                            category_tab[j][0] = category_tab[j][0] + 1
                            # category_tab[j][2] = category_tab[j][2] + tracks[i].full_trace[j].area
                            category_tab[j][3] = category_tab[j][3] + tracks[i].full_trace[j].area
                        elif (tracks[i].type == "monocyte"):
                            category_tab[j][1] = category_tab[j][1] + 1
                            category_tab[j][4] = category_tab[j][4] + tracks[i].full_trace[j].area
                        elif (tracks[i].type == "myeloma"):
                            category_tab[j][2] = category_tab[j][2] + 1
                            category_tab[j][5] = category_tab[j][5] + tracks[i].full_trace[j].area
                        else:
                            print("unknown cell category")
                            pass

        np.savetxt(outpath + "Misc/info_ucf/Beacon_" + str(Beacon) + "_category_table.txt", category_tab, fmt='%d')  # , fmt='%d'



    def mark_gt(self, frame, frame_index, scale, gt_frame, crop_height, crop_width, out_path, Beacon, add_imageJ, get_cells, f_det_txt, class_f, fluo = False):
        # get_cells = True
        cells_path = None
        debug = 0
        frame_red = None

        label = False
        # label = True
        frame_label = np.zeros(((crop_height, crop_width)), np.uint16)
        frame_label_2 = np.zeros_like(frame_label)
        label_path = out_path + "label/Beacon-" + str(Beacon) + "/"

        if (get_cells):
            cells_path = out_path + "ML/Beacon_" + str(Beacon) + "/cells/"
            os.makedirs(cells_path, exist_ok=True)
            # ret, frame_cell = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/",
            #                              frame_count=frame_index, data_type=1, scale=1,
            #                              crop_width=crop_width, crop_height=crop_height)
            ret, frame_cell = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/Beacon_" + str(Beacon) + "/img1/",
                                         frame_count=frame_index, data_type=1, scale=1,
                                         crop_width=crop_width, crop_height=crop_height, color = True)
            # = read_frame(image_path, frame_count, data_type, scale, crop_width = crop_width, crop_height = crop_height)
            #                 = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/Beacon_" + str(Beacon) + "/img1/", frame_count, data_type, scale, crop_width = crop_width, crop_height = crop_height)

            cv2.imwrite(out_path + "ML/Beacon_" + str(Beacon) + "/img1/{0:0=6d}".format(frame_index) + ".jpg", frame_cell)

        if frame_index == 0:
            if (label):
                if (not os.path.exists(label_path)):
                    os.makedirs(label_path)
                if (not os.path.exists(label_path + "version_0/")):
                    os.makedirs(label_path + "version_0/")

            if(debug == 1):
                # self.f_state = open(out_path + "info_ucf/Beacon_" + str(Beacon) + "_f_state_" + str(stable_period) + "_" + str(red_level) + ".txt", "w")
                pass

        if(add_imageJ):
            ret, red, frame_red = self.process_gt_frame_2(frame_index, gt_frame, crop_height, crop_width, scale)

            if ret:
                new_red = frame[:, :, 2].astype(float) + red.astype(float)
                new_red = np.clip(new_red, 0, 255).astype(np.uint8)
                # frame[:, :, 1] = new_red
                frame[:, :, 2] = new_red

        MO_cnt = 0
        MM_cnt = 0
        zero_cell = 0

        gt_count = 0
        gt_d_count = 0

        fluo_mo = 0
        no_fluo_mo = 0
        fluo_mm = 0
        no_fluo_mm = 0

        frame_org = frame[:, :, 0].copy()

        for tracks in (self.del_tracks, self.tracks):
            for i in range(len(tracks)):
                if (get_cells):
                    single_cell_path = cells_path + "{0:0=4d}".format(tracks[i].track_id) + "/"
                    os.makedirs(single_cell_path, exist_ok=True)

                if(tracks[i].good == False):
                    # print("tracks[i].good == False", tracks[i].track_id)
                    # continue
                    pass

                # x3 = tracks[i].coord[frame_index][0]
                # y3 = tracks[i].coord[frame_index][1]

                # if (not (np.isnan(tracks[i].score[frame_index]))):
                # if (not(np.isnan(x3) or np.isnan(y3)) and x3 > 0): # and np.count_nonzero(tracks[i].coordinates > 0) == 2
                if(tracks[i].full_trace[frame_index] != None):
                    x3 = tracks[i].full_trace[frame_index].horizontal_x
                    y3 = tracks[i].full_trace[frame_index].vertical_y

                    cell = tracks[i].full_trace[frame_index]
                    radius = cell.radius
                    draw_r = int(max(radius, 5 * scale))
                    draw_r = int(min(draw_r, 10 * scale))

                    live_status = tracks[i].live_state[frame_index]
                    track_id = tracks[i].track_id
                    # if (live_status > 1):
                    # if(tracks[i].motion_std > 2):

                    rect_x0 = int(x3 * scale - draw_r * 1.2)
                    rect_y0 = int(y3 * scale - draw_r * 1.2)

                    rect_x1 = int(x3 * scale + draw_r * 1.2)
                    rect_y1 = int(y3 * scale + draw_r * 1.2)

                    if (tracks[i].type == "macrophage"):
                        # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (255, 255, 0), int(0.5 * scale))
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 0, 255), int(0.5 * scale))
                        # cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), int(0.3 * scale))
                        # cv2.putText(frame, str(tracks[i].full_trace[frame_index].area), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), int(0.3 * scale))

                        MO_cnt += 1

                        if(label):
                            contour = tracks[i].full_trace[frame_index][6]
                            contour = contour / scale
                            contour = contour.astype(np.int32)
                            cv2.drawContours(frame_label, [contour], -1, (1, 1, 1), -1)
                            cv2.drawContours(frame_label_2, [contour], -1, (track_id, track_id, track_id), -1)

                        if (tracks[i].g_truth[frame_index] == 1):
                            gt_count += 1
                        elif tracks[i].g_truth[frame_index] == -1:# mark the error
                            cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), int(0.3 * scale))
                            pass
                        else:
                            # print("wrong tracks[i].g_truth[frame_index]", i, frame_index)
                            pass

                        if(fluo == True):
                            # if(tracks[i].full_trace[frame_index].fluo):
                            if(tracks[i].fluo):
                                fluo_mo += 1
                                # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), int(draw_r*1.2), (255, 255, 0), int(0.5 * scale))
                                cv2.rectangle(frame, (rect_x0, rect_y0), (rect_x1, rect_y1), (255, 255, 0), int(0.5 * scale))
                            else:
                                no_fluo_mo += 1

                    elif (tracks[i].type == "myeloma"):
                        # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 255, 255), int(0.5 * scale))
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 255, 0), int(0.5 * scale))
                        # cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))
                        # cv2.putText(frame, str(tracks[i].full_trace[frame_index].area), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))

                        if(label):
                            contour = tracks[i].full_trace[frame_index][6]
                            contour = contour / scale
                            contour = contour.astype(np.int32)
                            cv2.drawContours(frame_label, [contour], -1, (2, 2, 2), -1)
                            cv2.drawContours(frame_label_2, [contour], -1, (track_id, track_id, track_id), -1)

                        MM_cnt += 1


                        # if(tracks[i].full_trace[frame_index].fluo):
                        if(fluo == True):
                            if(tracks[i].fluo):
                                fluo_mm += 1
                                # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), int(draw_r*1.2), (255, 255, 0), int(0.5 * scale))
                                cv2.rectangle(frame, (rect_x0, rect_y0), (rect_x1, rect_y1), (255, 255, 0), int(0.5 * scale))
                            else:
                                no_fluo_mm += 1
                    else:
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (255, 255, 255), int(0.5 * scale))
                        print("Some track is wrong. The cell is not classified. It is not macrophage or myeloma.")

                    if(get_cells == True):
                        # cell_radius = 9
                        cell_radius = 6
                        scale_cell = 1
                        # ret, frame_cell = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/", frame_count = frame_index, data_type = 1, scale = scale_cell, crop_width = crop_width, crop_height = crop_height)
                        if((y3 - cell_radius) > 0 and (y3 + cell_radius) < (frame_cell.shape[0] / scale_cell) and x3 - cell_radius > 0 and (x3 + cell_radius) < (frame_cell.shape[1] / scale_cell)):
                            one_cell = frame_cell[int((y3 - cell_radius) * scale_cell):int((y3 + cell_radius) * scale_cell), int((x3 - cell_radius) * scale_cell):int((x3 + cell_radius) * scale_cell), 2]
                            cell_img_path = cells_path + "{0:0=4d}/".format(tracks[i].track_id) + "{0:0=4d}".format(tracks[i].track_id) + "_" + "{0:0=3d}".format(frame_index) + ".tif"
                            cv2.imwrite(cell_img_path, one_cell)
                            print(frame_index, -1, x3 - cell_radius, y3 - cell_radius, cell_radius * 2, cell_radius * 2, 1, -1, -1, -1, file=f_det_txt, sep=',')



        if (label):
            # frame_label = cv2.resize(frame_label, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(label_path + "label_" + "{0:0=3d}".format(frame_index) + ".tif", frame_label)

            # frame_label_2 = cv2.resize(frame_label_2, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(label_path + "version_0/label_" + "{0:0=3d}".format(frame_index) + ".tif", frame_label_2)


        # cv2.putText(frame, str(frame_index), (5 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (0, 255, 255), int(0.5 * scale))
        # cv2.putText(frame, str(frame_index), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (138, 221, 48), 1)

        # c_type = ["macrophage", "myeloma"]
        c_type = ["MO", "MM"]
        #
        # if(add_imageJ):
        #     cv2.putText(frame, c_type[0] + ":" + str(live_count) + "(" + str(gt_count) + "," + str(live_count - gt_count) + ")", (30 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.4 * scale, (255, 255, 0), int(0.5 * scale))
        #     cv2.putText(frame, c_type[1] + ":" + str(dead_count) + "(" + str(gt_d_count) + "," + str(dead_count - gt_d_count) + ")", (150 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale,
        #                 (0, 255, 255), int(0.5 * scale))
        # else:
        #     cv2.putText(frame, c_type[0] + ":" + str(live_count),
        #                 (30 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.4 * scale, (255, 255, 0), int(0.5 * scale))
        #     cv2.putText(frame, c_type[1] + ":" + str(dead_count),
        #                 (150 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale,
        #                 (0, 255, 255), int(0.5 * scale))

        # if(add_imageJ):
        #     cv2.putText(frame, c_type[0] + ":" + str(live_count) + "(" + str(gt_count) + "," + str(live_count - gt_count) + ")", (int(30 * scale * frame.shape[0]/512), 10 * scale), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.4 * scale * frame.shape[0]/512, (255, 255, 0), int(0.5 * scale))
        #     cv2.putText(frame, c_type[1] + ":" + str(dead_count) + "(" + str(gt_d_count) + "," + str(dead_count - gt_d_count) + ")", (int(150 * scale * frame.shape[0]/512), 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale * frame.shape[0]/512,
        #                 (0, 255, 255), int(0.5 * scale))
        # else:
        #     cv2.putText(frame, c_type[0] + ":" + str(live_count),
        #                 (int(30 * scale * frame.shape[0]/512), 10 * scale), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.4 * scale * frame.shape[0]/512, (255, 255, 0), int(0.5 * scale))
        #     cv2.putText(frame, c_type[1] + ":" + str(dead_count),
        #                 (int(150 * scale * frame.shape[0]/512), 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale * frame.shape[0]/512,
        #                 (0, 255, 255), int(0.5 * scale))

        if(add_imageJ):

            cv2.putText(frame, str(frame_index) + " " + c_type[0] + ":" + str(MO_cnt) + "(" + str(gt_count) + "," + str(MO_cnt - gt_count) + ")", (30 * scale, int(12 * scale * frame.shape[0]/4096)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * scale * frame.shape[0]/4096, (255, 255, 0), int(0.5 * scale * frame.shape[0]/4096))
            cv2.putText(frame, c_type[1] + ":" + str(MM_cnt) + "(" + str(gt_d_count) + "," + str(MM_cnt - gt_d_count) + ")", (60 * scale, int(12 * scale * frame.shape[0]/4096) * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale * frame.shape[0]/4096,
                        (0, 255, 255), int(0.5 * scale * frame.shape[0]/4096))
        else:
            cv2.putText(frame, str(frame_index) + " " + c_type[0] + ":" + str(MO_cnt) + "(" + str(fluo_mo) + "," + str(no_fluo_mo) + ")",
                        (90 * scale, int(36 * scale * frame.shape[0]/4096)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2 * scale * frame.shape[0]/4096, (0, 0, 255), int(1 * scale * frame.shape[0]/4096))

            cv2.putText(frame, c_type[1] + ":" + str(MM_cnt) + "(" + str(fluo_mm) + "," + str(no_fluo_mm) + ")",
                        (180 * scale, int(36 * scale * frame.shape[0]/4096) * 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale * frame.shape[0]/4096,
                        (0, 255, 0), int(1 * scale * frame.shape[0]/4096))

        print(MO_cnt, fluo_mo, no_fluo_mo, MM_cnt, fluo_mm, no_fluo_mm, file=class_f, sep=',')

        if(debug == 1):
            print(MO_cnt, gt_count, MM_cnt, gt_d_count, file = self.f_state)

        if (add_imageJ and len(gt_frame) > 0 and MO_cnt > 0):
            # accu = float(gt_count) / live_count
            diff = float(MO_cnt - gt_count + MM_cnt - gt_d_count)
            total = float(MO_cnt + MM_cnt)
            cv2.putText(frame, "accu: " + str(float("{0:.2f}".format(((total - diff) / total)))), (300 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * scale, (0, 0, 255), int(0.5 * scale))

        return frame, frame_red


    def mark_gt_3cat(self, frame, frame_index, scale, gt_frame, crop_height, crop_width, out_path, Beacon, add_imageJ, get_cells, f_det_txt, class_f, fluo = False):
        # get_cells = True
        cells_path = None
        debug = 0
        frame_red = None

        label = False
        # label = True
        frame_label = np.zeros(((crop_height, crop_width)), np.uint16)
        frame_label_2 = np.zeros_like(frame_label)
        label_path = out_path + "label/Beacon-" + str(Beacon) + "/"

        if (get_cells):
            cells_path = out_path + "ML/Beacon_" + str(Beacon) + "/cells/"
            os.makedirs(cells_path, exist_ok=True)
            # ret, frame_cell = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/",
            #                              frame_count=frame_index, data_type=1, scale=1,
            #                              crop_width=crop_width, crop_height=crop_height)
            ret, frame_cell = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/Beacon_" + str(Beacon) + "/img1/",
                                         frame_count=frame_index, data_type=1, scale=1,
                                         crop_width=crop_width, crop_height=crop_height, color = True)
            # = read_frame(image_path, frame_count, data_type, scale, crop_width = crop_width, crop_height = crop_height)
            #                 = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/Beacon_" + str(Beacon) + "/img1/", frame_count, data_type, scale, crop_width = crop_width, crop_height = crop_height)

            cv2.imwrite(out_path + "ML/Beacon_" + str(Beacon) + "/img1/{0:0=6d}".format(frame_index) + ".jpg", frame_cell)

        if frame_index == 0:
            if (label):
                if (not os.path.exists(label_path)):
                    os.makedirs(label_path)
                if (not os.path.exists(label_path + "version_0/")):
                    os.makedirs(label_path + "version_0/")

            if(debug == 1):
                # self.f_state = open(out_path + "info_ucf/Beacon_" + str(Beacon) + "_f_state_" + str(stable_period) + "_" + str(red_level) + ".txt", "w")
                pass

        if(add_imageJ):
            ret, red, frame_red = self.process_gt_frame_2(frame_index, gt_frame, crop_height, crop_width, scale)

            if ret:
                new_red = frame[:, :, 2].astype(float) + red.astype(float)
                new_red = np.clip(new_red, 0, 255).astype(np.uint8)
                # frame[:, :, 1] = new_red
                frame[:, :, 2] = new_red

        MO_cnt = 0
        MM_cnt = 0
        Mono_cnt = 0
        zero_cell = 0

        gt_count = 0
        gt_d_count = 0

        fluo_mo = 0
        no_fluo_mo = 0
        fluo_mm = 0
        no_fluo_mm = 0
        fluo_mono = 0
        no_fluo_mono = 0


        frame_org = frame[:, :, 0].copy()

        for tracks in (self.del_tracks, self.tracks):
            for i in range(len(tracks)):
                if (get_cells):
                    single_cell_path = cells_path + "{0:0=4d}".format(tracks[i].track_id) + "/"
                    os.makedirs(single_cell_path, exist_ok=True)

                if(tracks[i].good == False):
                    # print("tracks[i].good == False", tracks[i].track_id)
                    # continue
                    pass

                # x3 = tracks[i].coord[frame_index][0]
                # y3 = tracks[i].coord[frame_index][1]

                # if (not (np.isnan(tracks[i].score[frame_index]))):
                # if (not(np.isnan(x3) or np.isnan(y3)) and x3 > 0): # and np.count_nonzero(tracks[i].coordinates > 0) == 2
                if(tracks[i].full_trace[frame_index] != None):
                    x3 = tracks[i].full_trace[frame_index].horizontal_x
                    y3 = tracks[i].full_trace[frame_index].vertical_y

                    cell = tracks[i].full_trace[frame_index]
                    radius = cell.radius
                    draw_r = int(max(radius, 5 * scale))
                    draw_r = int(min(draw_r, 10 * scale))

                    live_status = tracks[i].live_state[frame_index]
                    track_id = tracks[i].track_id
                    # if (live_status > 1):
                    # if(tracks[i].motion_std > 2):

                    rect_x0 = int(x3 * scale - draw_r * 1.2)
                    rect_y0 = int(y3 * scale - draw_r * 1.2)

                    rect_x1 = int(x3 * scale + draw_r * 1.2)
                    rect_y1 = int(y3 * scale + draw_r * 1.2)

                    if (tracks[i].type == "macrophage"):
                        # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (255, 255, 0), int(0.5 * scale))
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 0, 255), int(0.5 * scale))
                        # cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), int(0.3 * scale))
                        # cv2.putText(frame, str(tracks[i].full_trace[frame_index].area), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), int(0.3 * scale))

                        MO_cnt += 1

                        if (tracks[i].g_truth[frame_index] == 1):
                            gt_count += 1
                        elif tracks[i].g_truth[frame_index] == -1:# mark the error
                            cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), int(0.3 * scale))
                            pass
                        else:
                            # print("wrong tracks[i].g_truth[frame_index]", i, frame_index)
                            pass

                        if(fluo == True):
                            # if(tracks[i].full_trace[frame_index].fluo):
                            if(tracks[i].fluo):
                                fluo_mo += 1
                                # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), int(draw_r*1.2), (255, 255, 0), int(0.5 * scale))
                                cv2.rectangle(frame, (rect_x0, rect_y0), (rect_x1, rect_y1), (255, 255, 0), int(0.5 * scale))
                            else:
                                no_fluo_mo += 1

                    elif (tracks[i].type == "myeloma"):
                        # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 255, 255), int(0.5 * scale))
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 255, 0), int(0.5 * scale))
                        # cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))
                        # cv2.putText(frame, str(tracks[i].full_trace[frame_index].area), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))

                        MM_cnt += 1


                        # if(tracks[i].full_trace[frame_index].fluo):
                        if(fluo == True):
                            if(tracks[i].fluo):
                                fluo_mm += 1
                                # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), int(draw_r*1.2), (255, 255, 0), int(0.5 * scale))
                                cv2.rectangle(frame, (rect_x0, rect_y0), (rect_x1, rect_y1), (255, 255, 0), int(0.5 * scale))
                            else:
                                no_fluo_mm += 1

                    elif (tracks[i].type == "monocyte"):
                        # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 255, 255), int(0.5 * scale))
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 255, 255), int(0.5 * scale))
                        # cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))
                        # cv2.putText(frame, str(tracks[i].full_trace[frame_index].area), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))

                        Mono_cnt += 1

                        # if(tracks[i].full_trace[frame_index].fluo):
                        if(fluo == True):
                            if(tracks[i].fluo):
                                fluo_mono += 1
                                # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), int(draw_r*1.2), (255, 255, 0), int(0.5 * scale))
                                cv2.rectangle(frame, (rect_x0, rect_y0), (rect_x1, rect_y1), (255, 255, 0), int(0.5 * scale))
                            else:
                                no_fluo_mono += 1

                    else:
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (255, 255, 255), int(0.5 * scale))
                        print("Some track is wrong. The cell is not classified. It is not macrophage or myeloma or monocyte.")

                    if(get_cells == True):
                        # cell_radius = 9
                        cell_radius = 6
                        scale_cell = 1
                        # ret, frame_cell = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/", frame_count = frame_index, data_type = 1, scale = scale_cell, crop_width = crop_width, crop_height = crop_height)
                        if((y3 - cell_radius) > 0 and (y3 + cell_radius) < (frame_cell.shape[0] / scale_cell) and x3 - cell_radius > 0 and (x3 + cell_radius) < (frame_cell.shape[1] / scale_cell)):
                            one_cell = frame_cell[int((y3 - cell_radius) * scale_cell):int((y3 + cell_radius) * scale_cell), int((x3 - cell_radius) * scale_cell):int((x3 + cell_radius) * scale_cell), 2]
                            cell_img_path = cells_path + "{0:0=4d}/".format(tracks[i].track_id) + "{0:0=4d}".format(tracks[i].track_id) + "_" + "{0:0=3d}".format(frame_index) + ".tif"
                            cv2.imwrite(cell_img_path, one_cell)
                            print(frame_index, -1, x3 - cell_radius, y3 - cell_radius, cell_radius * 2, cell_radius * 2, 1, -1, -1, -1, file=f_det_txt, sep=',')



        if (label):
            # frame_label = cv2.resize(frame_label, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(label_path + "label_" + "{0:0=3d}".format(frame_index) + ".tif", frame_label)

            # frame_label_2 = cv2.resize(frame_label_2, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(label_path + "version_0/label_" + "{0:0=3d}".format(frame_index) + ".tif", frame_label_2)


        # cv2.putText(frame, str(frame_index), (5 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (0, 255, 255), int(0.5 * scale))
        # cv2.putText(frame, str(frame_index), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (138, 221, 48), 1)

        # c_type = ["macrophage", "myeloma"]
        # c_type = ["MO", "MM", "Mono"]
        c_type = ["MO", "MM", "Others"]
        #
        # if(add_imageJ):
        #     cv2.putText(frame, c_type[0] + ":" + str(live_count) + "(" + str(gt_count) + "," + str(live_count - gt_count) + ")", (30 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.4 * scale, (255, 255, 0), int(0.5 * scale))
        #     cv2.putText(frame, c_type[1] + ":" + str(dead_count) + "(" + str(gt_d_count) + "," + str(dead_count - gt_d_count) + ")", (150 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale,
        #                 (0, 255, 255), int(0.5 * scale))
        # else:
        #     cv2.putText(frame, c_type[0] + ":" + str(live_count),
        #                 (30 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.4 * scale, (255, 255, 0), int(0.5 * scale))
        #     cv2.putText(frame, c_type[1] + ":" + str(dead_count),
        #                 (150 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale,
        #                 (0, 255, 255), int(0.5 * scale))

        # if(add_imageJ):
        #     cv2.putText(frame, c_type[0] + ":" + str(live_count) + "(" + str(gt_count) + "," + str(live_count - gt_count) + ")", (int(30 * scale * frame.shape[0]/512), 10 * scale), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.4 * scale * frame.shape[0]/512, (255, 255, 0), int(0.5 * scale))
        #     cv2.putText(frame, c_type[1] + ":" + str(dead_count) + "(" + str(gt_d_count) + "," + str(dead_count - gt_d_count) + ")", (int(150 * scale * frame.shape[0]/512), 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale * frame.shape[0]/512,
        #                 (0, 255, 255), int(0.5 * scale))
        # else:
        #     cv2.putText(frame, c_type[0] + ":" + str(live_count),
        #                 (int(30 * scale * frame.shape[0]/512), 10 * scale), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.4 * scale * frame.shape[0]/512, (255, 255, 0), int(0.5 * scale))
        #     cv2.putText(frame, c_type[1] + ":" + str(dead_count),
        #                 (int(150 * scale * frame.shape[0]/512), 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale * frame.shape[0]/512,
        #                 (0, 255, 255), int(0.5 * scale))

        if(add_imageJ):

            cv2.putText(frame, str(frame_index) + " " + c_type[0] + ":" + str(MO_cnt) + "(" + str(gt_count) + "," + str(MO_cnt - gt_count) + ")", (30 * scale, int(12 * scale * frame.shape[0]/4096)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * scale * frame.shape[0]/4096, (255, 255, 0), int(0.5 * scale * frame.shape[0]/4096))
            cv2.putText(frame, c_type[1] + ":" + str(MM_cnt) + "(" + str(gt_d_count) + "," + str(MM_cnt - gt_d_count) + ")", (60 * scale, int(12 * scale * frame.shape[0]/4096) * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale * frame.shape[0]/4096,
                        (0, 255, 255), int(0.5 * scale * frame.shape[0]/4096))
        else:
            # cv2.putText(frame, str(frame_index) + " " + c_type[0] + ":" + str(MO_cnt) + "(" + str(fluo_mo) + "," + str(no_fluo_mo) + ")",
            #             (90 * scale, int(36 * scale * frame.shape[0]/4096)), cv2.FONT_HERSHEY_SIMPLEX,
            #             1.2 * scale * frame.shape[0]/4096, (0, 0, 255), int(1 * scale * frame.shape[0]/4096))
            #
            # cv2.putText(frame, c_type[1] + ":" + str(MM_cnt) + "(" + str(fluo_mm) + "," + str(no_fluo_mm) + ")",
            #             (90 * scale, int(36 * scale * frame.shape[0]/4096) * 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale * frame.shape[0]/4096,
            #             (0, 255, 0), int(1 * scale * frame.shape[0]/4096))
            #
            # cv2.putText(frame, c_type[2] + ":" + str(Mono_cnt) + "(" + str(fluo_mono) + "," + str(no_fluo_mono) + ")",
            #             (90 * scale, int(36 * scale * frame.shape[0]/4096) * 3), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale * frame.shape[0]/4096,
            #             (0, 255, 255), int(1 * scale * frame.shape[0]/4096))

            cv2.putText(frame, str(frame_index) + " " + c_type[0] + ":" + str(MO_cnt),
                        (90 * scale, int(36 * scale * frame.shape[0]/4096)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2 * scale * frame.shape[0]/4096, (0, 0, 255), int(1 * scale * frame.shape[0]/4096))

            cv2.putText(frame, c_type[1] + ":" + str(MM_cnt),
                        (90 * scale, int(36 * scale * frame.shape[0]/4096) * 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale * frame.shape[0]/4096,
                        (0, 255, 0), int(1 * scale * frame.shape[0]/4096))

            cv2.putText(frame, c_type[2] + ":" + str(Mono_cnt),
                        (90 * scale, int(36 * scale * frame.shape[0]/4096) * 3), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale * frame.shape[0]/4096,
                        (0, 255, 255), int(1 * scale * frame.shape[0]/4096))


        print(MO_cnt, fluo_mo, no_fluo_mo, MM_cnt, fluo_mm, no_fluo_mm, Mono_cnt, fluo_mono, no_fluo_mono, file=class_f, sep=',')

        if(debug == 1):
            print(MO_cnt, gt_count, MM_cnt, gt_d_count, file = self.f_state)

        if (add_imageJ and len(gt_frame) > 0 and MO_cnt > 0):
            # accu = float(gt_count) / live_count
            diff = float(MO_cnt - gt_count + MM_cnt - gt_d_count)
            total = float(MO_cnt + MM_cnt)
            cv2.putText(frame, "accu: " + str(float("{0:.2f}".format(((total - diff) / total)))), (300 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * scale, (0, 0, 255), int(0.5 * scale))

        return frame, frame_red

    def mark_gt_3cat_death(self, frame, frame_index, scale, gt_frame, crop_height, crop_width, out_path, Beacon, add_imageJ, get_cells, f_det_txt, class_f, fluo = False):
        # get_cells = True
        cells_path = None
        debug = 0
        frame_red = None

        label = False
        # label = True
        frame_label = np.zeros(((crop_height, crop_width)), np.uint16)
        frame_label_2 = np.zeros_like(frame_label)
        label_path = out_path + "label/Beacon-" + str(Beacon) + "/"

        fluo = False

        if (get_cells):
            cells_path = out_path + "ML/Beacon_" + str(Beacon) + "/cells/"
            os.makedirs(cells_path, exist_ok=True)
            # ret, frame_cell = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/",
            #                              frame_count=frame_index, data_type=1, scale=1,
            #                              crop_width=crop_width, crop_height=crop_height)
            ret, frame_cell = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/Beacon_" + str(Beacon) + "/img1/",
                                         frame_count=frame_index, data_type=1, scale=1,
                                         crop_width=crop_width, crop_height=crop_height, color = True)
            # = read_frame(image_path, frame_count, data_type, scale, crop_width = crop_width, crop_height = crop_height)
            #                 = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/Beacon_" + str(Beacon) + "/img1/", frame_count, data_type, scale, crop_width = crop_width, crop_height = crop_height)

            cv2.imwrite(out_path + "ML/Beacon_" + str(Beacon) + "/img1/{0:0=6d}".format(frame_index) + ".jpg", frame_cell)

        if frame_index == 0:
            if (label):
                if (not os.path.exists(label_path)):
                    os.makedirs(label_path)
                if (not os.path.exists(label_path + "version_0/")):
                    os.makedirs(label_path + "version_0/")

            if(debug == 1):
                # self.f_state = open(out_path + "info_ucf/Beacon_" + str(Beacon) + "_f_state_" + str(stable_period) + "_" + str(red_level) + ".txt", "w")
                pass

        if(add_imageJ):
            ret, red, frame_red = self.process_gt_frame_2(frame_index, gt_frame, crop_height, crop_width, scale)

            if ret:
                new_red = frame[:, :, 2].astype(float) + red.astype(float)
                new_red = np.clip(new_red, 0, 255).astype(np.uint8)
                # frame[:, :, 1] = new_red
                frame[:, :, 2] = new_red

        MO_cnt = 0
        MM_cnt = 0
        Mono_cnt = 0
        zero_cell = 0

        gt_count = 0
        gt_d_count = 0

        fluo_mo = 0
        no_fluo_mo = 0
        fluo_mm = 0
        no_fluo_mm = 0
        fluo_mono = 0
        no_fluo_mono = 0

        live_count = 0
        dead_count = 0


        frame_org = frame[:, :, 0].copy()

        for tracks in (self.del_tracks, self.tracks):
            for i in range(len(tracks)):
                if (get_cells):
                    single_cell_path = cells_path + "{0:0=4d}".format(tracks[i].track_id) + "/"
                    os.makedirs(single_cell_path, exist_ok=True)

                if(tracks[i].good == False):
                    # print("tracks[i].good == False", tracks[i].track_id)
                    # continue
                    pass

                # x3 = tracks[i].coord[frame_index][0]
                # y3 = tracks[i].coord[frame_index][1]

                # if (not (np.isnan(tracks[i].score[frame_index]))):
                # if (not(np.isnan(x3) or np.isnan(y3)) and x3 > 0): # and np.count_nonzero(tracks[i].coordinates > 0) == 2
                if(tracks[i].full_trace[frame_index] != None):
                    x3 = tracks[i].full_trace[frame_index].horizontal_x
                    y3 = tracks[i].full_trace[frame_index].vertical_y

                    cell = tracks[i].full_trace[frame_index]
                    radius = cell.radius
                    draw_r = int(max(radius, 5 * scale))
                    draw_r = int(min(draw_r, 10 * scale))

                    live_status = tracks[i].live_state[frame_index]
                    track_id = tracks[i].track_id
                    # if (live_status > 1):
                    # if(tracks[i].motion_std > 2):

                    rect_x0 = int(x3 * scale - draw_r * 1.2)
                    rect_y0 = int(y3 * scale - draw_r * 1.2)

                    rect_x1 = int(x3 * scale + draw_r * 1.2)
                    rect_y1 = int(y3 * scale + draw_r * 1.2)

                    if (tracks[i].type == "macrophage"):
                        # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (255, 255, 0), int(0.5 * scale))
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 0, 255), int(0.5 * scale))
                        # cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), int(0.3 * scale))
                        # cv2.putText(frame, str(tracks[i].full_trace[frame_index].area), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), int(0.3 * scale))

                        MO_cnt += 1

                        if (tracks[i].g_truth[frame_index] == 1):
                            gt_count += 1
                        elif tracks[i].g_truth[frame_index] == -1:# mark the error
                            cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), int(0.3 * scale))
                            pass
                        else:
                            # print("wrong tracks[i].g_truth[frame_index]", i, frame_index)
                            pass

                        if(fluo == True):
                            # if(tracks[i].full_trace[frame_index].fluo):
                            if(tracks[i].fluo):
                                fluo_mo += 1
                                # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), int(draw_r*1.2), (255, 255, 0), int(0.5 * scale))
                                cv2.rectangle(frame, (rect_x0, rect_y0), (rect_x1, rect_y1), (255, 255, 0), int(0.5 * scale))
                            else:
                                no_fluo_mo += 1

                    elif (tracks[i].type == "myeloma"):
                        # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 255, 255), int(0.5 * scale))
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 255, 0), int(0.5 * scale))
                        # cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))
                        # cv2.putText(frame, str(tracks[i].full_trace[frame_index].area), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))

                        MM_cnt += 1


                        # if(tracks[i].full_trace[frame_index].fluo):
                        if(fluo == True):
                            if(tracks[i].fluo):
                                fluo_mm += 1
                                # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), int(draw_r*1.2), (255, 255, 0), int(0.5 * scale))
                                cv2.rectangle(frame, (rect_x0, rect_y0), (rect_x1, rect_y1), (255, 255, 0), int(0.5 * scale))
                            else:
                                no_fluo_mm += 1

                        if (live_status > 1):
                            live_count += 1
                        else:
                            dead_count += 1

                    elif (tracks[i].type == "monocyte"):
                        # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 255, 255), int(0.5 * scale))
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 255, 255), int(0.5 * scale))
                        # cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))
                        # cv2.putText(frame, str(tracks[i].full_trace[frame_index].area), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))

                        Mono_cnt += 1

                        # if(tracks[i].full_trace[frame_index].fluo):
                        if(fluo == True):
                            if(tracks[i].fluo):
                                fluo_mono += 1
                                # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), int(draw_r*1.2), (255, 255, 0), int(0.5 * scale))
                                cv2.rectangle(frame, (rect_x0, rect_y0), (rect_x1, rect_y1), (255, 255, 0), int(0.5 * scale))
                            else:
                                no_fluo_mono += 1

                    else:
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (255, 255, 255), int(0.5 * scale))
                        print("Some track is wrong. The cell is not classified. It is not macrophage or myeloma or monocyte.")

                    if(get_cells == True):
                        # cell_radius = 9
                        cell_radius = 6
                        scale_cell = 1
                        # ret, frame_cell = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/", frame_count = frame_index, data_type = 1, scale = scale_cell, crop_width = crop_width, crop_height = crop_height)
                        if((y3 - cell_radius) > 0 and (y3 + cell_radius) < (frame_cell.shape[0] / scale_cell) and x3 - cell_radius > 0 and (x3 + cell_radius) < (frame_cell.shape[1] / scale_cell)):
                            one_cell = frame_cell[int((y3 - cell_radius) * scale_cell):int((y3 + cell_radius) * scale_cell), int((x3 - cell_radius) * scale_cell):int((x3 + cell_radius) * scale_cell), 2]
                            cell_img_path = cells_path + "{0:0=4d}/".format(tracks[i].track_id) + "{0:0=4d}".format(tracks[i].track_id) + "_" + "{0:0=3d}".format(frame_index) + ".tif"
                            cv2.imwrite(cell_img_path, one_cell)
                            print(frame_index, -1, x3 - cell_radius, y3 - cell_radius, cell_radius * 2, cell_radius * 2, 1, -1, -1, -1, file=f_det_txt, sep=',')



        if (label):
            # frame_label = cv2.resize(frame_label, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(label_path + "label_" + "{0:0=3d}".format(frame_index) + ".tif", frame_label)

            # frame_label_2 = cv2.resize(frame_label_2, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(label_path + "version_0/label_" + "{0:0=3d}".format(frame_index) + ".tif", frame_label_2)


        # cv2.putText(frame, str(frame_index), (5 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (0, 255, 255), int(0.5 * scale))
        # cv2.putText(frame, str(frame_index), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (138, 221, 48), 1)

        # c_type = ["macrophage", "myeloma"]
        c_type = ["MO", "MM", "Mono"]
        #

        live_dead = True
        fluo_show = False
        row_1 = str(frame_index) + " " + c_type[0] + ":" + str(MO_cnt)
        if(fluo_show):
            row_1 += "(fluo: " + str(fluo_mo) + ",no fluo" + str(no_fluo_mo) + ")"
        cv2.putText(frame, row_1,
                    (90 * scale, int(36 * scale * frame.shape[0]/4096)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2 * scale * frame.shape[0]/4096, (0, 0, 255), int(1 * scale * frame.shape[0]/4096))

        row_2 = c_type[1] + ":" + str(MM_cnt)
        if(fluo_show):
            row_2 += "(fluo: " + str(fluo_mm) + ",no fluo" + str(no_fluo_mm) + ")"
        if(live_dead):
            row_2 += "live: " + str(live_count) + " dead: " + str(dead_count)
        cv2.putText(frame, row_2,
                    (90 * scale, int(36 * scale * frame.shape[0]/4096) * 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale * frame.shape[0]/4096,
                    (0, 255, 0), int(1 * scale * frame.shape[0]/4096))

        row_3 = c_type[2] + ":" + str(Mono_cnt)
        if(fluo_show):
            row_3 += "(fluo: " + str(fluo_mono) + ",no fluo" + str(no_fluo_mono) + ")"
        cv2.putText(frame, row_3,
                    (90 * scale, int(36 * scale * frame.shape[0]/4096) * 3), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale * frame.shape[0]/4096,
                    (0, 255, 255), int(1 * scale * frame.shape[0]/4096))

        print(MO_cnt, fluo_mo, no_fluo_mo, MM_cnt, fluo_mm, no_fluo_mm, Mono_cnt, fluo_mono, no_fluo_mono, file=class_f, sep=',')

        if(debug == 1):
            print(MO_cnt, gt_count, MM_cnt, gt_d_count, file = self.f_state)

        if (add_imageJ and len(gt_frame) > 0 and MO_cnt > 0):
            # accu = float(gt_count) / live_count
            diff = float(MO_cnt - gt_count + MM_cnt - gt_d_count)
            total = float(MO_cnt + MM_cnt)
            cv2.putText(frame, "accu: " + str(float("{0:.2f}".format(((total - diff) / total)))), (300 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * scale, (0, 0, 255), int(0.5 * scale))

        return frame, frame_red

    def mark_gt_3cat_death_rewrite(self, frame, frame_index, scale, gt_frame, crop_height, crop_width, out_path, Beacon, add_imageJ, get_cells, f_det_txt, class_f, fluo = False):
        # get_cells = True
        cells_path = None
        debug = 0
        frame_red = None

        fluo = False

        if (get_cells):
            cells_path = out_path + "ML/Beacon_" + str(Beacon) + "/cells/"
            os.makedirs(cells_path, exist_ok=True)
            # ret, frame_cell = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/",
            #                              frame_count=frame_index, data_type=1, scale=1,
            #                              crop_width=crop_width, crop_height=crop_height)
            ret, frame_cell = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/Beacon_" + str(Beacon) + "/img1/",
                                         frame_count=frame_index, data_type=1, scale=1,
                                         crop_width=crop_width, crop_height=crop_height, color = True)
            # = read_frame(image_path, frame_count, data_type, scale, crop_width = crop_width, crop_height = crop_height)
            #                 = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/Beacon_" + str(Beacon) + "/img1/", frame_count, data_type, scale, crop_width = crop_width, crop_height = crop_height)

            cv2.imwrite(out_path + "ML/Beacon_" + str(Beacon) + "/img1/{0:0=6d}".format(frame_index) + ".jpg", frame_cell)


        fluo_mo = 0
        no_fluo_mo = 0
        fluo_mm = 0
        no_fluo_mm = 0
        fluo_mono = 0
        no_fluo_mono = 0


        cate_live = {}
        cate_live["macrophage"] = [0, 0]
        cate_live["monocyte"] = [0, 0]
        cate_live["myeloma"] = [0, 0]


        frame_org = frame[:, :, 0].copy()

        for tracks in (self.del_tracks, self.tracks):
            for i in range(len(tracks)):
                if (get_cells):
                    single_cell_path = cells_path + "{0:0=4d}".format(tracks[i].track_id) + "/"
                    os.makedirs(single_cell_path, exist_ok=True)

                if(tracks[i].good == False):
                    # print("tracks[i].good == False", tracks[i].track_id)
                    # continue
                    pass

                # x3 = tracks[i].coord[frame_index][0]
                # y3 = tracks[i].coord[frame_index][1]

                # if (not (np.isnan(tracks[i].score[frame_index]))):
                # if (not(np.isnan(x3) or np.isnan(y3)) and x3 > 0): # and np.count_nonzero(tracks[i].coordinates > 0) == 2
                if(tracks[i].full_trace[frame_index] != None):
                    x3 = tracks[i].full_trace[frame_index].horizontal_x
                    y3 = tracks[i].full_trace[frame_index].vertical_y

                    cell = tracks[i].full_trace[frame_index]
                    radius = cell.radius
                    draw_r = int(max(radius, 5 * scale))
                    draw_r = int(min(draw_r, 10 * scale))

                    live_status = tracks[i].live_state[frame_index]
                    track_id = tracks[i].track_id
                    # if (live_status > 1):
                    # if(tracks[i].motion_std > 2):

                    rect_x0 = int(x3 * scale - draw_r * 1.2)
                    rect_y0 = int(y3 * scale - draw_r * 1.2)

                    rect_x1 = int(x3 * scale + draw_r * 1.2)
                    rect_y1 = int(y3 * scale + draw_r * 1.2)

                    if (live_status > 1):
                        cate_live[tracks[i].type][0] += 1
                    else:
                        cate_live[tracks[i].type][1] += 1

                    if (tracks[i].type == "macrophage"):
                        # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (255, 255, 0), int(0.5 * scale))
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 0, 255), int(0.5 * scale))
                        # cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), int(0.3 * scale))
                        # cv2.putText(frame, str(tracks[i].full_trace[frame_index].area), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), int(0.3 * scale))

                        # MO_cnt += 1

                        if(fluo == True):
                            # if(tracks[i].full_trace[frame_index].fluo):
                            if(tracks[i].fluo):
                                fluo_mo += 1
                                # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), int(draw_r*1.2), (255, 255, 0), int(0.5 * scale))
                                cv2.rectangle(frame, (rect_x0, rect_y0), (rect_x1, rect_y1), (255, 255, 0), int(0.5 * scale))
                            else:
                                no_fluo_mo += 1


                    elif (tracks[i].type == "myeloma"):
                        # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 255, 255), int(0.5 * scale))
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 255, 0), int(0.5 * scale))
                        # cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))
                        # cv2.putText(frame, str(tracks[i].full_trace[frame_index].area), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))

                        # MM_cnt += 1


                        # if(tracks[i].full_trace[frame_index].fluo):
                        if(fluo == True):
                            if(tracks[i].fluo):
                                fluo_mm += 1
                                # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), int(draw_r*1.2), (255, 255, 0), int(0.5 * scale))
                                cv2.rectangle(frame, (rect_x0, rect_y0), (rect_x1, rect_y1), (255, 255, 0), int(0.5 * scale))
                            else:
                                no_fluo_mm += 1

                    elif (tracks[i].type == "monocyte"):
                        # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 255, 255), int(0.5 * scale))
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 255, 255), int(0.5 * scale))
                        # cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))
                        # cv2.putText(frame, str(tracks[i].full_trace[frame_index].area), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))

                        # Mono_cnt += 1

                        # if(tracks[i].full_trace[frame_index].fluo):
                        if(fluo == True):
                            if(tracks[i].fluo):
                                fluo_mono += 1
                                # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), int(draw_r*1.2), (255, 255, 0), int(0.5 * scale))
                                cv2.rectangle(frame, (rect_x0, rect_y0), (rect_x1, rect_y1), (255, 255, 0), int(0.5 * scale))
                            else:
                                no_fluo_mono += 1

                    else:
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (255, 255, 255), int(0.5 * scale))
                        print("Some track is wrong. The cell is not classified. It is not macrophage or myeloma or monocyte.")

                    if(get_cells == True):
                        # cell_radius = 9
                        cell_radius = 6
                        scale_cell = 1
                        # ret, frame_cell = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/", frame_count = frame_index, data_type = 1, scale = scale_cell, crop_width = crop_width, crop_height = crop_height)
                        if((y3 - cell_radius) > 0 and (y3 + cell_radius) < (frame_cell.shape[0] / scale_cell) and x3 - cell_radius > 0 and (x3 + cell_radius) < (frame_cell.shape[1] / scale_cell)):
                            one_cell = frame_cell[int((y3 - cell_radius) * scale_cell):int((y3 + cell_radius) * scale_cell), int((x3 - cell_radius) * scale_cell):int((x3 + cell_radius) * scale_cell), 2]
                            cell_img_path = cells_path + "{0:0=4d}/".format(tracks[i].track_id) + "{0:0=4d}".format(tracks[i].track_id) + "_" + "{0:0=3d}".format(frame_index) + ".tif"
                            cv2.imwrite(cell_img_path, one_cell)
                            print(frame_index, -1, x3 - cell_radius, y3 - cell_radius, cell_radius * 2, cell_radius * 2, 1, -1, -1, -1, file=f_det_txt, sep=',')


        # c_type = ["macrophage", "myeloma"]
        # c_type = ["MO", "MM", "Mono"]
        #
        row_0 = str(frame_index)
        cv2.putText(frame, row_0,
                    (90 * scale, int(36 * scale * frame.shape[0]/4096)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2 * scale * frame.shape[0]/4096, (0, 0, 255), int(1 * scale * frame.shape[0]/4096))

        c_type = ["macrophage", "monocyte", "myeloma"]
        colors = [(0, 0, 255), (0, 255, 0), (0, 255, 255)]

        live_dead = True
        fluo_show = False

        for t in range(len(c_type)):
            row_1 = c_type[t] + ":" + str(sum(cate_live[c_type[t]]))
            if (live_dead):
                row_1 += " live: " + str(cate_live[c_type[t]][0]) + " dead: " + str(cate_live[c_type[t]][1])

            cv2.putText(frame, row_1,
                        (90 * scale, int(36 * scale * frame.shape[0]/4096) * (t + 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2 * scale * frame.shape[0]/4096, colors[t], int(1 * scale * frame.shape[0]/4096))

        return frame, frame_red


    def process_gt_frame_2(self, frame_index, gt_frame, crop_height, crop_width, scale):

        pad_wid = 200
        if (len(gt_frame) == 0):
            return False, None

        # print(gt_frame.shape())
        gt_frame = gt_frame[0:crop_height, 0:crop_width]
        # print(gt_frame.shape())

        # gt_frame = cv2.resize(gt_frame, (gt_frame.shape[1] * scale, gt_frame.shape[0] * scale),
        #                       interpolation=cv2.INTER_CUBIC)

        # cv2.imshow("gt_frame", gt_frame)
        # cv2.waitKey()

        coord = self.coord[frame_index * 2:frame_index * 2 + 2]
        coord = coord.astype(int)
        gt_frame_pad = cv2.copyMakeBorder(gt_frame, pad_wid, pad_wid, pad_wid, pad_wid, cv2.BORDER_CONSTANT)
        gt_frame = gt_frame_pad[
                   pad_wid + coord[0]:pad_wid + coord[0] + gt_frame.shape[0],
                   pad_wid + coord[1]:pad_wid + coord[1] + gt_frame.shape[1]]

        # frame_0 = frame[:, :, 0]
        frame_1 = gt_frame[:, :, 1]
        frame_2 = gt_frame[:, :, 2]
        # print(frame_2.shape())
        red = frame_2.astype(np.float) - frame_1.astype(np.float)
        red_uint8 = np.clip(red, 0, 255).astype(np.uint8)

        red_uint8 = cv2.resize(red_uint8, (red_uint8.shape[1] * scale, red_uint8.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

        # ret, th4 = cv2.threshold(red_uint8, 10, 255, cv2.THRESH_BINARY)

        # contours, hierarchy = cv2.findContours(th4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)

        # return contours, th4
        return True, red_uint8, gt_frame
