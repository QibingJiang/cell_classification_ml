import cv2
import numpy as np
# import pandas as pd
import time
# import hungarian
# from hungarian_algorithm import algorithm
import string
import os

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

# from inspect import currentframe, getframeinfo
# from skimage.measure import compare_ssim
# from operator import add
from statistics import mean
from util import read_frame

debug = 0
# array_size = 300
array_size = 800
stable_period = 4
red_level = 3 #for video from imageJ
# red_level = 50
# red_level = 20

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

class CellTrack(object):

    def __init__(self, prediction, trackIdCount):
        self.track_id = trackIdCount  # identification of each track object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = []  # trace path
        self.live_score = 0.0
        self.death_score = 0.0
        self.image_amount = 0
        self.match_score = 0

        self.coordinates = np.zeros([array_size, 2], dtype=float)
        self.coordinates[:, :] = np.nan

        self.live_state = np.zeros(array_size, dtype=float)
        self.live_state[:] = np.nan

        self.loc_var = np.zeros(array_size, dtype=float)
        self.loc_var[:] = np.nan

        self.cell_diff = np.zeros(array_size, dtype=float)
        self.cell_diff[:] = np.nan

        self.area = np.zeros(array_size, dtype=float)
        self.area[:] = np.nan

        self.g_truth = np.zeros(array_size, dtype=float)
        self.g_truth[:] = np.nan
        self.g_truth_t = -1

        self.shift = np.zeros((array_size, 2), dtype=float)
        # self.shift[:] = np.nan

        self.last_cell = np.zeros(0, dtype=int)

        self.overlap_change = np.zeros(array_size, dtype=float)
        self.overlap_change[:] = np.nan
        self.children = []
        self.parent = None
        self.full_trace = [None] * array_size

        self.mitosis_time = np.zeros(array_size, dtype=float)
        self.mitosis_time[:] = np.nan


class CellClassifier(object):

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length, trackIdCount):

        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.del_tracks = []
        self.trackIdCount = trackIdCount
        self.alive_mat = []
        self.coordinate_matrix = []
        self.tracks_s = []
        self.old_tracks_s = []
        self.coord = np.zeros(array_size * 2, dtype=float)
        self.coord[:] = np.nan
        self.f_state = None
        self.background_pixel = 0
        self.cell_core_r = 0
        self.cell_core_r_mean = 0
        self.image_amount = 0

    def match_track(self, detections, frame_prev, frame, frame_index, scale):

        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = CellTrack(detections[i], self.trackIdCount)
                self.tracks.append(track)

                one_row = np.zeros(array_size, dtype=float)
                one_row[:] = np.nan
                self.alive_mat.append(one_row)

                one_row = np.zeros(array_size * 2, dtype=int)
                self.coordinate_matrix.append(one_row)

                self.trackIdCount += 1

        N = len(self.tracks)
        M = len(detections)
        # print('tracks, dections:', N, M)
        D = max(N, M)
        cost = np.full([D, D], fill_value=2000, dtype=np.int32)  # Cost matrix

        predict = np.zeros([N, 2], dtype=np.uint8)
        for i in range(N):
            predict[i][0] = self.tracks[i].prediction[0]
            predict[i][1] = self.tracks[i].prediction[1]

        det = np.zeros([M, 2], dtype=np.uint8)
        for i in range(M):
            det[i][0] = detections[i][0]
            det[i][1] = detections[i][1]

        cost_new = distance_matrix(predict, det)  # N, M. row is track, column is detected cell
        cost_new = np.where(cost_new < self.dist_thresh, cost_new, 2000)

        cost[0:N, 0:M] = cost_new
        assignment = []
        for _ in range(N):
            assignment.append(-1)

        # print(cost)

        t3 = time.time()

        # print("%s" % time.ctime(t3))
        answers = hungarian.lap(cost)
        t4 = time.time()
        # print("%s" % time.ctime(t4))

        if (debug == 1):
            print("hungarian time:", t4 - t3)

        # row_ind, col_ind = linear_sum_assignment(cost_new)
        # print(cost[row_ind, col_ind].sum())
        # for i in range(len(row_ind)):
        #     assignment[row_ind[i]] = col_ind[i]
        # print("Nothing")

        # print(predict, det, cost_new)
        # print(answers)

        # print(cost_new.dtype)

        if (N > M):  # less cells are detected.
            for i in range(M):
                assignment[answers[1][i]] = i  # the second row in answers noted the matched tracks(rows).
                # print(cost[answers[1][i]][i], end = ' ')
                pass
            # print(answers[1], len(answers[1]))
            sum0 = sum(cost[answers[1][0:M], range(M)])  # The sum of hungarian result.
            # print("sum0: ", sum0)
        else:
            for i in range(N):
                assignment[i] = answers[0][i]  # the first row in answers noted the matched detected cells(columns).
                # print(cost[i][answers[0][i]], end = ' ')
                pass
            # print(answers[0], len(answers[0]))
            sum1 = sum(cost[range(N), answers[0][0:N]])
            # print("sum1: ", sum1)

        # print("\n")

        # if(frame_index > 0):
        #     cv2.namedWindow('frame_prev', cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow('frame_prev', 900, 900)
        #     cv2.imshow('frame_prev', frame_prev)
        #
        #     cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow('frame', 900, 900)
        #     cv2.imshow('frame', frame)

        for i in range(len(assignment)):
            if (assignment[i] == -1 or cost[i][assignment[i]] > self.dist_thresh):
                assignment[i] = -1
                # un_assigned_tracks.append(i)
                self.tracks[i].skipped_frames += 1
            else:
                # if(frame_index > 0):
                #     p1 = predict[i][0]
                #     p0 = predict[i][1]
                #     c1 = det[assignment[i]][0]
                #     c0 = det[assignment[i]][1]
                #     cell_prev = frame_prev[(p0 - 5) * scale:(p0 + 5) * scale, (p1 - 5) * scale:(p1 + 5) * scale]
                #     cell_curr = frame[(c0 - 5) * scale:(c0 + 5) * scale, (c1 - 5) * scale:(c1 + 5) * scale]
                #     # ret = cv2.matchTemplate(cell_prev, cell_curr, cv2.TM_SQDIFF)
                #     ret = compare_ssim(cell_prev, cell_curr)
                #     self.tracks[i].match_score = ret
                #
                #     print(self.tracks[i].track_id, self.tracks[i].match_score, end = ';', flush=True)
                #
                #
                #     # cv2.imshow('cell_prev', cell_prev)
                #     # cv2.imshow('cell_curr', cell_curr)
                #     # cv2.waitKey()
                self.tracks[i].skipped_frames = 0
        print()

        i = 0
        while (i < len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                # print("track " + str(self.tracks[i].track_id) + " skipped " + str(self.tracks[i].skipped_frames) + "frames")
                self.del_tracks.append(self.tracks[i])
                del self.tracks[i]
                del assignment[i]
            else:
                i = i + 1

        # print(N - len(self.tracks), "tracks have been deleted")

        un_assigned_detects = []
        for i in range(len(detections)):
            if i not in assignment:
                un_assigned_detects.append(i)
                assignment.append(i)

        new_track_id_start = self.trackIdCount

        if (len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = CellTrack(detections[un_assigned_detects[i]], self.trackIdCount)
                self.tracks.append(track)

                one_row = np.zeros(array_size, dtype=float)
                one_row[:] = np.nan
                self.alive_mat.append(one_row)

                one_row = np.zeros(array_size * 2, dtype=int)
                self.coordinate_matrix.append(one_row)

                self.trackIdCount += 1

        if (len(frame.shape) == 2):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        for i in range(len(assignment)):

            if (assignment[i] != -1):
                if debug == 1:
                    print(assignment[i])
                self.tracks[i].trace.append(detections[assignment[i]])
                self.tracks[i].prediction = detections[assignment[i]]

                if (len(self.tracks[i].trace) > 0):

                    x3 = int(self.tracks[i].trace[len(self.tracks[i].trace) - 1][0])
                    y3 = int(self.tracks[i].trace[len(self.tracks[i].trace) - 1][1])
                    ratio = self.tracks[i].trace[len(self.tracks[i].trace) - 1][2]

                    self.coordinate_matrix[self.tracks[i].track_id][frame_index * 2] = x3
                    self.coordinate_matrix[self.tracks[i].track_id][frame_index * 2 + 1] = y3
                    self.alive_mat[self.tracks[i].track_id][frame_index] = ratio

                    # if(self.tracks[i].track_id % 3  == 1):
                    if (self.tracks[i].track_id < new_track_id_start):
                        cv2.circle(frame, (x3 * scale, y3 * scale), 5 * scale, (255, 255, 0), ((1 * scale) >> 2))
                        cv2.putText(frame, str(self.tracks[i].track_id), (int(x3 + 5) * scale, int(y3 + 3) * scale),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 127, 255), 2)
                    else:
                        cv2.circle(frame, (x3 * scale, y3 * scale), 5 * scale, (0, 0, 255), ((1 * scale) >> 2))
                        cv2.putText(frame, str(self.tracks[i].track_id), (int(x3 + 9) * scale, int(y3 + 4) * scale),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 0, 255), 2)

            if (len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) - self.max_trace_length):
                    del self.tracks[i].trace[j]

        cv2.putText(frame, str(frame_index), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (138, 221, 48), 2)

        # cv2.namedWindow('Tracking',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Tracking', 900,900)
        # cv2.imshow('Tracking', frame)
        # cv2.waitKey()

        return frame

    def match_and_mitosis(self, centers_s, frame_prev, frame, frame_index, scale):

        colors = [(255, 255, 0), (255, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 0), (0, 0, 255)]

        if (len(frame.shape) > 2):
            frame_draw = frame.copy()
            frame_draw = frame_draw.astype(np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame_draw = frame.copy()

        if (frame_index == 0):
            for i in range(len(centers_s)):
                for point in centers_s[i]:
                    track = CellTrack(point, self.trackIdCount)
                    self.tracks.append(track)

                    one_row = np.zeros(array_size, dtype=float)
                    one_row[:] = np.nan
                    self.alive_mat.append(one_row)

                    one_row = np.zeros(array_size * 2, dtype=int)
                    one_row[-1] = track.track_id
                    self.coordinate_matrix.append(one_row)

                    track.trace.append(point)
                    track.full_trace[frame_index] = point

                    self.coordinate_matrix[track.track_id][frame_index * 2] = point[0]
                    self.coordinate_matrix[track.track_id][frame_index * 2 + 1] = point[1]
                    self.alive_mat[track.track_id][frame_index] = point[2]

                    x3 = point[0]
                    y3 = point[1]

                    track.coordinates[frame_index][0:2] = point[0:2]
                    track.live_state[frame_index] = point[2]
                    track.area[frame_index] = point[3]

                    box = track.trace[-1][7]
                    cv2.drawContours(frame_draw, [box], 0, (255, 255, 0), 2)
                    cv2.putText(frame_draw, str(track.track_id), (int(x3), int(y3)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 127, 255), 2)

                    self.trackIdCount += 1
            return frame_draw

        assignment = []
        for i in range(len(self.tracks)):
            assignment.append(np.full(3, -1, dtype=np.int16))

        dist_thresh = [50, 6, 6, 15, 8]  # the value can be further limited

        new_p = []
        new_d = []

        for le in range(len(centers_s)):
            p = np.zeros([len(self.tracks), 3], dtype=np.uint16)

            for i in range(len(p)):
                p[i][0] = int(self.tracks[i].trace[-1][0])
                p[i][1] = int(self.tracks[i].trace[-1][1])
                p[i][2] = i

            if (len(new_p) > 0):
                new_p_arr = np.array(new_p, dtype=np.uint16)
                p = np.concatenate((p, new_p_arr))

            d = np.zeros([len(centers_s[le]), 4], dtype=np.uint16)

            for i in range(len(d)):
                d[i][0] = centers_s[le][i][0]
                d[i][1] = centers_s[le][i][1]
                d[i][2] = le
                d[i][3] = i

            if (len(new_d) > 0):
                d = np.concatenate((d, np.array(new_d)))

            new_p, new_d, cost = self.my_hungarian(frame, frame_index, scale, p, d, assignment, dist_thresh[le], colors[le])


        new_track_id_start = self.trackIdCount

        if len(new_d) != 0:
            for i in range(len(new_d)):
                cell = centers_s[new_d[i][2]][new_d[i][3]]
                track = CellTrack(cell, self.trackIdCount)
                # track.trace.append(cell) # the cell is appended later.
                self.tracks.append(track)

                one_row = np.zeros(array_size, dtype=float)
                one_row[:] = np.nan
                self.alive_mat.append(one_row)

                one_row = np.zeros(array_size * 2, dtype=int)
                one_row[-1] = track.track_id
                self.coordinate_matrix.append(one_row)

                assignment.append(np.full(3, -1, dtype=np.int16))
                assignment[-1][0] = track.track_id
                assignment[-1][1] = new_d[i][2]
                assignment[-1][2] = new_d[i][3]

                # self.mitosis(cost, new_d, frame_index, track, cell, centers_s, assignment, frame_draw, i)
                # dist_s = cost[:, new_d[i][3]]
                # p_idx = np.argmin(dist_s)
                # dist = cost[p_idx, new_d[i][3]]
                # m = len(self.tracks[p_idx].trace) - 1
                #
                # # while m > -1:
                # #     if(self.tracks[p_idx].trace[m][9] == frame_index - 1):
                # #         break
                # #     m -= 1
                #
                # if(dist < self.tracks[p_idx].trace[m][8] * 2):
                #     print(frame_index, self.tracks[p_idx].track_id, track.track_id, cell[3], self.tracks[p_idx].trace[m][3])
                #     x3 = cell[0]
                #     y3 = cell[1]
                #     area = cell[3]
                #     box = cell[7]
                #
                #     bro_det = centers_s[assignment[p_idx][1]][assignment[p_idx][2]]
                #     bro_x3 = bro_det[0]
                #     bro_y3 = bro_det[1]
                #     bro_area = bro_det[3]
                #     bro_box = bro_det[7]
                #     ratio = area / bro_area
                #
                #     if(0.5 < ratio < 2):
                #         cv2.drawContours(frame_draw, [box], 0, (255, 255, 255), 2)
                #         cv2.putText(frame_draw, str(track.track_id), (int(x3), int(y3)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #                     (255, 127, 255), 2)
                #
                #         cv2.drawContours(frame_draw, [bro_box], 0, (255, 255, 255), 2)
                #         cv2.putText(frame_draw, str(self.tracks[p_idx].track_id), (int(bro_x3), int(bro_y3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 127, 255), 2)
                #
                #         # track.parent = id(self.tracks[p_idx])
                #         # self.tracks[p_idx].children.append(id(track))
                #         track.parent = self.tracks[p_idx].track_id
                #         self.tracks[p_idx].children.append(track.track_id)

                self.trackIdCount += 1

        for i in range(len(assignment)):

            idx_0 = assignment[i][1]
            idx_1 = assignment[i][2]
            if(idx_0 == -1 or idx_1 == -1):
                continue

            self.tracks[i].trace.append(centers_s[idx_0][idx_1])
            self.tracks[i].full_trace[frame_index] = centers_s[idx_0][idx_1]

            if len(self.tracks[i].trace) > 0:
                x3 = self.tracks[i].trace[-1][0]
                y3 = self.tracks[i].trace[-1][1]
                ratio = self.tracks[i].trace[-1][2]
                area = self.tracks[i].trace[-1][3]
                le = int(self.tracks[i].trace[-1][4])
                loc_var = self.tracks[i].trace[-1][5]
                box = self.tracks[i].trace[-1][7]

                self.tracks[i].coordinates[frame_index][0:2] = [x3, y3]
                self.tracks[i].live_state[frame_index] = ratio
                self.tracks[i].loc_var[frame_index] = loc_var
                self.tracks[i].area[frame_index] = area

                if(self.tracks[i].track_id < new_track_id_start):
                    cv2.drawContours(frame_draw, [box], 0, (255, 255, 0), 2)
                    cv2.putText(frame_draw, str(self.tracks[i].track_id), (int(x3), int(y3)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 127, 255), 2)
                else:
                    cv2.drawContours(frame_draw, [box], 0, (0, 255, 255), 2)
                    cv2.putText(frame_draw, str(self.tracks[i].track_id), (int(x3), int(y3)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 127, 255), 2)


        cv2.putText(frame_draw, str(frame_index), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (138, 221, 48), 2)

        return frame_draw

    def mitosis(self, cost, new_d, frame_index, track, cell, centers_s, assignment, frame_draw, i):
        # dist_s = cost[:, new_d[i][3]]
        dist_s = cost[:, i]
        p_idx = np.argmin(dist_s)
        dist = cost[p_idx, i]
        m = len(self.tracks[p_idx].trace) - 1

        # while m > -1:
        #     if(self.tracks[p_idx].trace[m][9] == frame_index - 1):
        #         break
        #     m -= 1

        if (dist < self.tracks[p_idx].trace[m][8] * 2):
            x3 = cell[0]
            y3 = cell[1]
            area = cell[3]
            box = cell[7]

            if (assignment[p_idx][1] == -1 or assignment[p_idx][2] == -1):
                return
            bro_det = centers_s[assignment[p_idx][1]][assignment[p_idx][2]]
            bro_x3 = bro_det[0]
            bro_y3 = bro_det[1]
            bro_area = bro_det[3]
            bro_box = bro_det[7]
            ratio = area / bro_area

            print(ratio)
            if (0.5 < ratio < 2):
                cv2.drawContours(frame_draw, [box], 0, (255, 255, 255), 2)
                cv2.putText(frame_draw, str(track.track_id), (int(x3), int(y3)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 127, 255), 2)

                cv2.drawContours(frame_draw, [bro_box], 0, (255, 255, 255), 2)
                cv2.putText(frame_draw, str(self.tracks[p_idx].track_id), (int(bro_x3), int(bro_y3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 127, 255), 2)

                # track.parent = id(self.tracks[p_idx])
                # self.tracks[p_idx].children.append(id(track))
                track.parent = self.tracks[p_idx].track_id
                track.mitosis_time[frame_index - 5:frame_index + 5] = 1

                self.tracks[p_idx].children.append(track.track_id)
                self.tracks[p_idx].mitosis_time[frame_index - 5:frame_index + 5] = 1

    def match_track_3_times(self, centers_s, frame_prev, frame, frame_index, scale):

        # colors = [(255, 255, 0), (255, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 0), (0, 0, 255)]
        colors = [(255, 255, 0), (255, 255, 0), (255, 255, 0), (255, 255, 0), (255, 255, 0), (255, 255, 0)]

        if (len(frame.shape) > 2):
            frame_org = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_org = frame.copy()

        if (len(frame.shape) == 2):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if (frame_index == 0):
            self.tracks_s = []
            self.tracks_s.append([])
            self.tracks_s.append([])
            self.tracks_s.append([])

            for i in range(len(centers_s)):
                for point in centers_s[i]:
                    track = CellTrack(point, self.trackIdCount)
                    self.tracks.append(track)

                    # one_row = np.zeros(array_size, dtype=float)
                    # one_row[:] = np.nan
                    # self.alive_mat.append(one_row)
                    #
                    # one_row = np.zeros(array_size * 2, dtype=int)
                    # one_row[-1] = track.track_id
                    # self.coordinate_matrix.append(one_row)

                    track.trace.append(point)
                    track.full_trace[frame_index] = point

                    # self.coordinate_matrix[track.track_id][frame_index * 2] = point[0]
                    # self.coordinate_matrix[track.track_id][frame_index * 2 + 1] = point[1]
                    # self.alive_mat[track.track_id][frame_index] = point[2]

                    x3 = point[0]
                    y3 = point[1]

                    track.coordinates[frame_index][0:2] = point[0:2]
                    track.live_state[frame_index] = point[2]
                    track.loc_var[frame_index] = point[5]
                    track.area[frame_index] = point[3]

                    track.last_cell = frame_org[int((y3 - 5) * scale):int((y3 + 5) * scale), int((x3 - 5) * scale):int((x3 + 5) * scale)].copy()

                    # print("last_cell: ", track.track_id, frame_index, x3, y3, track.last_cell.shape)

                    if (i < 3):
                        cv2.circle(frame, (int(point[0] * scale), int(point[1] * scale)), 5 * scale, colors[i], ((1 * scale) >> 2))
                        cv2.putText(frame, str(track.track_id), (int(point[0] + 6) * scale, int(point[1] + 3) * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, colors[i], 2)

                    self.tracks_s[i].append(len(self.tracks) - 1)
                    self.trackIdCount += 1
            cv2.putText(frame, str(frame_index), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (138, 221, 48), 2)
            return frame
            # return one_sample_cell

        # assignment = np.zeros([len(self.tracks), 3], dtype=np.int16) #0: track_id, 1 and 2 is the cell index in centers_s
        # assignment[:, :] = -1
        assignment = []
        for i in range(len(self.tracks)):
            assignment.append(np.full(3, -1, dtype=np.int16))
            assignment[i][0] = self.tracks[i].track_id

        # dist_thresh = [15, 8, 8, 20, 8]  # the value can be further limited
        dist_thresh = [8, 6, 6, 15, 8]  # the value can be further limited
        # match the very white cells(vwc)
        # detections = centers_s[0]

        new_p = []
        new_d = []

        for le in range(len(centers_s)):
            # print("level: ", le)
            p = np.zeros([len(self.tracks_s[le]), 3], dtype=np.uint16)

            for i in range(len(p)):
                idx = self.tracks_s[le][i]
                # idx = self.tracks
                p[i][0] = self.tracks[idx].trace[-1][0]
                p[i][1] = self.tracks[idx].trace[-1][1]
                p[i][2] = idx

            if (len(new_p) > 0):
                new_p_arr = np.array(new_p, dtype=np.uint16)
                p = np.concatenate((p, new_p_arr))
            p_N = len(p)

            d = np.zeros([len(centers_s[le]), 4], dtype=np.uint16)

            for i in range(len(d)):
                d[i][0] = centers_s[le][i][0]
                d[i][1] = centers_s[le][i][1]
                d[i][2] = le
                d[i][3] = i

            if (len(new_d) > 0):
                d = np.concatenate((d, np.array(new_d)))
            d_N = len(d)

            new_p, new_d, cost = self.my_hungarian(frame, frame_index, scale, p, d, assignment, dist_thresh[le], colors[le])

            # if le == 1 and len(new_p) > 0 and len(new_d) > 0:
            #     new_p, new_d = self.my_hungarian(frame, scale, new_p, new_d, assignment, dist_thresh[3], colors[3])

        # print(new_d, new_p)

        # # for moving cells
        # p = []
        # d = []
        # i = 0
        # new_p = new_p.tolist()
        # while i < len(new_p):
        #     idx = new_p[i][2]
        #     if (self.tracks[idx].trace[-1][4] < 2):
        #         p.append(new_p[i])
        #         del new_p[i]
        #     else:
        #         i += 1
        #
        # new_d = new_d.tolist()
        # if (len(p) > 0):
        #     i = 0
        #     while i < len(new_d):
        #         cell = centers_s[new_d[i][2]][new_d[i][3]]
        #         if (cell[4] < 2):
        #             d.append(new_d[i])
        #             del new_d[i]
        #         else:
        #             i += 1
        # p = np.array(p)
        # d = np.array(d)
        #
        # if len(d) > 0:
        #     new_new_p, new_new_d = self.my_hungarian(frame, scale, p, d, assignment, dist_thresh[3], colors[3])
        #     new_p = new_p + new_new_p.tolist()
        #     new_d = new_d + new_new_d.tolist()
        #

        # for tracks in old images
        if len(self.old_tracks_s) > 0:
            # p = np.zeros([len(self.old_tracks_s[le]), 3], dtype=np.uint8)
            p = []
            for le in range(3):
                for i in range(len(self.old_tracks_s[le])):
                    idx = self.old_tracks_s[le][i]
                    # p[i][0] = self.tracks[idx].trace[-1][0]
                    # p[i][1] = self.tracks[idx].trace[-1][1]
                    # p[i][2] = idx
                    p.append([self.tracks[idx].trace[-1][0], self.tracks[idx].trace[-1][1], idx])

            p = np.array(p, dtype=np.uint16)
            d = np.array(new_d)

            if len(p) > 0 and len(d) > 0:
                new_p, new_d, cost = self.my_hungarian(frame, frame_index, scale, p, d, assignment, dist_thresh[4], colors[4])

        for i in range(len(assignment)):
            if (assignment[i][1] == -1):
                self.tracks[i].skipped_frames += 1
            else:
                self.tracks[i].skipped_frames = 0

        i = 0
        temp_num = len(self.tracks)
        while (i < len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                # print("track " + str(self.tracks[i].track_id) + " skipped " + str(self.tracks[i].skipped_frames) + "frames")
                self.del_tracks.append(self.tracks[i])
                del self.tracks[i]
                del assignment[i]
            else:
                i = i + 1

        # print(temp_num - len(self.tracks), "tracks have been deleted")


        new_track_id_start = self.trackIdCount

        if len(new_d) != 0:

            p = []
            # for idx in range(len(self.del_tracks)):
            #     p.append([self.del_tracks[idx].trace[-1][0], self.del_tracks[idx].trace[-1][1]])

            for idx in range(len(self.tracks)):
                p.append([self.tracks[idx].trace[-1][0], self.tracks[idx].trace[-1][1]])

            p = np.array(p, dtype=np.uint16)
            N = len(p)
            M = len(new_d)
            D = max(N, M)
            cost = np.full([D, D], fill_value=2000, dtype=np.int32)  # Cost matrix
            cost_new = distance_matrix(p[:, 0:2], new_d[:, 0:2])  # N, M. row is track, column is detected cell
            cost_new = np.where(cost_new < 20, cost_new, 2000)
            cost[0:N, 0:M] = cost_new

            for i in range(len(new_d)):
                cell = centers_s[new_d[i][2]][new_d[i][3]]
                track = CellTrack(cell, self.trackIdCount)
                # track.trace.append(cell) # The cell will appended later.
                self.tracks.append(track)

                one_row = np.zeros(array_size, dtype=float)
                one_row[:] = np.nan
                self.alive_mat.append(one_row)

                one_row = np.zeros(array_size * 2, dtype=int)
                one_row[-1] = track.track_id
                self.coordinate_matrix.append(one_row)

                assignment.append(np.full(3, -1, dtype=np.int16))
                assignment[-1][0] = track.track_id
                assignment[-1][1] = new_d[i][2]
                assignment[-1][2] = new_d[i][3]

                cv2.circle(frame, (int(cell[0] * scale), int(cell[1] * scale)), 5 * scale, colors[5], ((1 * scale) >> 2))
                cv2.putText(frame, str(track.track_id), (int(cell[0] + 6) * scale, int(cell[1] + 3) * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, colors[5], 2)

                # self.mitosis(cost, new_d, frame_index, track, cell, centers_s, assignment, frame, i)

                self.trackIdCount += 1

        new_tracks_s = [[], [], []]

        # the unmatched track become old tracks for the next image.
        old_tracks_s = [[], [], []]

        count_0 = 0
        count_1 = 1
        for i in range(len(assignment)):
            if int(assignment[i][1]) == -1:
                cell = self.tracks[i].trace[-1]
                old_tracks_s[int(cell[4])].append(i)
            else:
                idx_0 = assignment[i][1]
                idx_1 = assignment[i][2]

                self.tracks[i].trace.append(centers_s[idx_0][idx_1])
                self.tracks[i].full_trace[frame_index] = centers_s[idx_0][idx_1]

                if len(self.tracks[i].trace) > 0:
                    x3 = self.tracks[i].trace[-1][0]
                    y3 = self.tracks[i].trace[-1][1]
                    ratio = self.tracks[i].trace[-1][2]
                    area = self.tracks[i].trace[-1][3]
                    le = int(self.tracks[i].trace[-1][4])
                    loc_var = self.tracks[i].trace[-1][5]

                    self.tracks[i].coordinates[frame_index][0:2] = [x3, y3]
                    self.tracks[i].live_state[frame_index] = ratio
                    self.tracks[i].loc_var[frame_index] = loc_var
                    self.tracks[i].area[frame_index] = area

                    one_cell = frame_org[int((y3 - 5) * scale):int((y3 + 5) * scale), int((x3 - 5) * scale):int((x3 + 5) * scale)].copy()
                    # one_cell = frame_org[(y3 - 5) * scale:(y3 + 5) * scale, (x3 - 5) * scale:(x3 + 5) * scale]

                    trace_len = len(self.tracks[i].trace)
                    shift = [0, 0]
                    if trace_len > 1:
                        # old_x = self.tracks[i].trace[trace_len - 2][0]
                        # old_y = self.tracks[i].trace[trace_len - 2][1]
                        # old_cell = frame_prev[int((old_y - 5) * scale):int((old_y + 5) * scale), int((old_x - 5) * scale):int((old_x + 5) * scale)]
                        old_cell = self.tracks[i].last_cell

                        d = 5
                        if 2*d < x3 < (frame_org.shape[1] / scale - 2 * d) and 2 * d < y3 < (frame_org.shape[0] / scale - 2 * d):
                            image_a = old_cell
                            image_b = frame_org[int((y3 - 2 * d) * scale):int((y3 + 2 * d) * scale), int((x3 - 2 * d) * scale):int((x3 + 2 * d) * scale)].copy()

                            template = image_a
                            ret = cv2.matchTemplate(image_b, template, cv2.TM_SQDIFF)
                            resu = cv2.minMaxLoc(ret)
                            shift = [resu[2][1] - d * scale, resu[2][0] - d * scale] #resu[2][0] is x, resu[2][1] is y.
                            # new_x = x3 * scale + shift[1]
                            # new_y = y3 * scale + shift[0]
                            # new_cell = frame_org[int(new_y - 5 * scale):int(new_y + 5 * scale), int(new_x - 5 * scale):int(new_x + 5 * scale)]
                            # diff_2 = np.sum(np.abs(new_cell.astype(float) - old_cell.astype(float)))
                            # sq_diff = ((new_cell.astype(float) - old_cell.astype(float)) ** 2).sum()
                            diff = resu[0]
                            count_0 += 1
                        else:
                            diff = ((one_cell.astype(float) - old_cell.astype(float)) ** 2).sum()
                            shift = [0, 0]
                            count_1 += 1

                        self.tracks[i].shift[frame_index][0] = self.tracks[i].shift[frame_index - 1][0] + shift[0]
                        self.tracks[i].shift[frame_index][1] = self.tracks[i].shift[frame_index - 1][1] + shift[1]

                        self.tracks[i].cell_diff[frame_index] = diff

                        new_x = x3 * scale + shift[1]
                        new_y = y3 * scale + shift[0]
                        new_cell = frame_org[int(new_y - 5 * scale):int(new_y + 5 * scale), int(new_x - 5 * scale):int(new_x + 5 * scale)].copy()

                        ret, th0 = cv2.threshold(old_cell, 170, 100, cv2.THRESH_BINARY)
                        ret, th1 = cv2.threshold(new_cell, 170, 100, cv2.THRESH_BINARY)
                        th2 = th0 + th1
                        change = np.count_nonzero(th2 == 100)

                        self.tracks[i].overlap_change[frame_index] = change

                        # pt1 = (int((x3 - 5) * scale), int((y3 - 5) * scale))
                        # pt2 = (int((x3 + 5) * scale), int((y3 + 5) * scale))
                        # cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 1)
                        #
                        # pt1 = (int(new_x - 5 * scale), int(new_y - 5 * scale))
                        # pt2 = (int(new_x + 5 * scale), int(new_y + 5 * scale))
                        # cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 1)
                        #
                        # cv2.namedWindow('Tracking',cv2.WINDOW_NORMAL)
                        # cv2.resizeWindow('Tracking', 900,900)
                        # cv2.imshow('Tracking', frame)
                        # cv2.waitKey()

                    new_x = x3 + shift[1] / scale
                    new_y = y3 + shift[0] / scale
                    self.tracks[i].trace[-1][0:2] = [new_x, new_y]
                    self.tracks[i].coordinates[frame_index][0:2] = [new_x, new_y]

                    self.tracks[i].last_cell = one_cell
                    new_tracks_s[le].append(i)

                    # if(frame_index == 5 and self.tracks[i].track_id == 33):
                    # print("qibing: ", self.tracks[i].trace[-1])

        # print("count_0, count_1: ", count_0, count_1)
        self.tracks_s = new_tracks_s
        self.old_tracks_s = old_tracks_s

        # cv2.putText(frame, str(frame_index), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (138, 221, 48), 2)

        # cv2.namedWindow('Tracking',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Tracking', 900,900)
        # cv2.imshow('Tracking', frame)
        # # cv2.imwrite("/home/qibing/qibing.png", frame)
        # cv2.waitKey()

        # frame_org = None
        return frame
        # return one_sample_cell

    def my_hungarian_0(self, frame, frame_index, scale, predict, det, assignment, dist_thresh, color):
        N = len(predict)
        M = len(det)

        D = max(N, M)
        cost = np.full([D, D], fill_value=2000, dtype=np.int32)  # Cost matrix

        cost_new = distance_matrix(predict[:, 0:2], det[:, 0:2])  # N, M. row is track, column is detected cell
        cost_new = np.where(cost_new < dist_thresh, cost_new, 2000)

        cost[0:N, 0:M] = cost_new

        t3 = time.time()
        answers = hungarian.lap(cost)
        # print(cost)
        t4 = time.time()

        if (debug == 1):
            print("hungarian time:", t4 - t3)

        new_p = []
        new_d = []
        for i in range(len(predict)):  # the predict index is the cost row index.
            id_0 = predict[i][2]
            id_1 = answers[0][i]  # the first row in answers noted the matched detected cells(columns).
            if (id_1 < len(det)):
                if (cost[i][id_1] < dist_thresh):
                    assignment[id_0][1] = det[id_1][2]
                    assignment[id_0][2] = det[id_1][3]
                    cv2.circle(frame, (int(det[id_1][0] * scale), int(det[id_1][1] * scale)), 5 * scale, color,
                               ((1 * scale) >> 2))
                    cv2.putText(frame, str(self.tracks[id_0].track_id),
                                (int(det[id_1][0] + 6) * scale, int(det[id_1][1] + 3) * scale),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, color, 2)

                else:
                    new_p.append(predict[i])
                    new_d.append(det[id_1])

            pass

        for i in range(N, D):
            id = answers[0][i]
            new_d.append(det[id])

        for i in range(M, D):
            id = answers[1][i]
            new_p.append(predict[id])

        new_p = np.array(new_p, dtype=np.uint16)
        new_d = np.array(new_d, dtype=np.uint16)
        return new_p, new_d, cost
    def my_hungarian(self, frame, frame_index, scale, predict, det, assignment, dist_thresh, color):
        N = len(predict)
        M = len(det)

        D = max(N, M)
        cost = np.full([D, D], fill_value=2000, dtype=np.int32)  # Cost matrix

        cost_new = distance_matrix(predict[:, 0:2], det[:, 0:2])  # N, M. row is track, column is detected cell
        # cost_new = distance_matrix(det[:, 0:2], predict[:, 0:2])  # N, M. row is track, column is detected cell
        cost_new = np.where(cost_new < dist_thresh, cost_new, 2000)

        cost[0:N, 0:M] = cost_new
        cost_new = cost_new.astype(np.int32)
        # print("qibing: ", cost_new)
        bigraph_cost = {}
        for i in range(cost_new.shape[0]):
            col_dict = {j:cost_new[i][j] for j in range(cost_new.shape[1])}
            # if(len(col_dict) > 0):
            bigraph_cost[str(i)] = col_dict

        # bigraph_cost = {i:{j:cost_new[i][j]} for i in range(cost_new.shape[0]) for j in range(cost_new.shape[1]) if cost_new[i][j] < 2000}}

        t3 = time.time()
        # ret = algorithm.find_matching(bigraph_cost, matching_type='min', return_type='list')
        row_ind, col_ind = linear_sum_assignment(cost)
        answers = []
        answers.append(col_ind)
        # print(cost_new.shape, cost_new, row_ind, col_ind, sep='\n')
        # print(ret)
        t4 = time.time()

        debug = 0
        if (debug == 1):
            print("hungarian time:", t4 - t3)

        new_p = []
        new_d = []
        for i in range(len(predict)):  # the predict index is the cost row index.
            id_0 = predict[i][2]
            id_1 = answers[0][i]  # the first row in answers noted the matched detected cells(columns).
            if (id_1 < len(det)):
                if (cost[i][id_1] < dist_thresh):
                    assignment[id_0][1] = det[id_1][2]
                    assignment[id_0][2] = det[id_1][3]
                    cv2.circle(frame, (int(det[id_1][0] * scale), int(det[id_1][1] * scale)), 5 * scale, color,
                               ((1 * scale) >> 2))
                    cv2.putText(frame, str(self.tracks[id_0].track_id),
                                (int(det[id_1][0] + 6) * scale, int(det[id_1][1] + 3) * scale),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, color, 2)

                else:
                    new_p.append(predict[i])
                    new_d.append(det[id_1])
            else:
                new_p.append(predict[i])

            pass

        for i in range(N, D):
            id = answers[0][i]
            new_d.append(det[id])

        # for i in range(M, D):
        #     id = answers[1][i]
        #     new_p.append(predict[id])

        new_p = np.array(new_p, dtype=np.uint16)
        new_d = np.array(new_d, dtype=np.uint16)
        return new_p, new_d, cost

    def analyse_classification(self, outpath, frame_count):
        # print("tracker save.")

        self.image_amount = frame_count
        window_radius = 4

        i = 0
        # while (i < len(self.coordinate_matrix)):
        #     if (np.count_nonzero(self.coordinate_matrix[i]) < 60):
        #         del self.coordinate_matrix[i]
        #         del self.alive_mat[i]
        #         # input()
        #     else:
        #         i += 1

        file = open(outpath + "file0.txt", "w")
        for i in range(0, len(self.alive_mat), 1):
            file.write("track: %d;\n" % self.coordinate_matrix[i][-1])

            file.write("coordinate: ")
            for j in range(array_size):
                file.write("%s, " % self.coordinate_matrix[i][j * 2])
                file.write("%s, " % self.coordinate_matrix[i][j * 2 + 1])
            file.write(";\n")

            file.write("ratio: ")
            for j in range(array_size):
                file.write("%s, " % self.alive_mat[i][j])
            file.write(";\n")

            one_track = np.zeros_like(self.alive_mat[i])
            one_track[:] = np.nan

            for j in range(window_radius, self.image_amount):
                temp_array = self.alive_mat[i][j - window_radius: j + window_radius + 1]
                temp_array = temp_array[~np.isnan(temp_array)]

                if (len(temp_array) > 1):
                    local_max = np.max(temp_array)
                    local_min = np.min(temp_array)
                    one_track[j] = local_max - local_min
                else:
                    pass

            file.write("Max_min: ")
            for s in range(array_size):
                file.write("%s, " % one_track[s])
            file.write(";\n")

            for k in range(len(one_track)):
                if (one_track[k] > 0.25):
                    one_track[k] = 1
                else:
                    one_track[k] = 0

            file.write("state: ")
            for s in range(array_size):
                file.write("%s, " % one_track[s])
            file.write(";\n")

            one_track_str = str(one_track)
            one_track_str = one_track_str.replace("\n", "")
            one_track_str = one_track_str.replace("[", "")
            one_track_str = one_track_str.replace("]", "")

            sub_arrays = one_track_str.split('1.')
            max_dead_count = 0

            for m in range(1, len(sub_arrays) - 1, 1):
                dead_count = sub_arrays[m].count('0.')
                if (max_dead_count < dead_count):
                    max_dead_count = dead_count

            if (max_dead_count == 0 and one_track_str.count('1.') < 10):
                print("The track is too stationary", one_track_str)
                one_track[:] = -1
                pass
            else:  # pad start
                n = array_size - 1
                while (not (one_track[n] == 1) and n > -1):
                    one_track[n] = -1
                    n = n - 1

                while (n > -1):
                    one_track[n] = 1
                    n = n - 1
                # pad end
                last_dead_status = np.where(one_track == -1)[0][0]
                # print("last_dead_status: ", last_dead_status)

                # print(i, self.image_amount, last_dead_status, max_dead_count)

                file.write("info: ")
                if (self.image_amount - last_dead_status < max_dead_count):
                    file.write("%s, " % "image_count: " + str(self.image_amount) + ", last_dead_status: " + str(
                        last_dead_status) + ", max_dead_count: " + str(max_dead_count) + ";")

                    for p in range(last_dead_status, len(one_track), 1):
                        one_track[p] = 1
                else:
                    pass
                # print(np.where(one_track == -1))
                file.write(";\n")

            file.write("final: ")
            for s in range(array_size):
                file.write("%s, " % one_track[s])
            file.write(";\n")

            self.alive_mat[i] = one_track

        file.close()

        live_dead_table = np.zeros((self.image_amount, 2), dtype=int)

        for j in range(0, self.image_amount):

            for i in range(0, len(self.alive_mat), 1):
                cell_x = self.coordinate_matrix[i][2 * j + 0]
                cell_y = self.coordinate_matrix[i][2 * j + 1]

                if (cell_x > 0 and cell_y > 0):

                    if (self.alive_mat[i][j] == 1):
                        live_dead_table[j][0] = live_dead_table[j][0] + 1
                    elif (self.alive_mat[i][j] == -1):
                        live_dead_table[j][1] = live_dead_table[j][1] + 1
                    else:
                        pass

            # live_dead_table[j][2] = live_dead_table[j][0] + live_dead_table[j][1]

        if (not os.path.exists(outpath)):
            os.makedirs(outpath)

        np.savetxt(outpath + "live_dead_table.txt", live_dead_table, fmt='%d')

    def analyse_classification_2(self, outpath, frame_count):  # ratio and local variance
        print("tracker save.")

        self.image_amount = frame_count
        window_radius = 4

        # i = 0
        # while (i < len(self.coordinate_matrix)):
        #     if (np.count_nonzero(self.coordinate_matrix[i]) < 60):
        #         del self.coordinate_matrix[i]
        #         del self.alive_mat[i]
        #         # input()
        #     else:
        #         i += 1

        file = open(outpath + "file2_" + time.strftime("%d_%H_%M", time.localtime()) + ".txt", "w")

        live_dead_table = np.zeros((self.image_amount, 2), dtype=int)

        for tracks in (self.del_tracks, self.tracks):
            for tra_i in range(0, len(tracks), 1):
                file.write("track: %d;\n" % tracks[tra_i].track_id)

                file.write("coordinate: ")
                for j in range(array_size):
                    file.write("%s, " % tracks[tra_i].coordinates[j])
                    # file.write("%s, " % tracks[tra_i].coordinates[j * 2 + 1])
                file.write(";\n")

                file.write("ratio: ")
                for j in range(array_size):
                    file.write("%s, " % tracks[tra_i].live_state[j])
                file.write(";\n")

                one_track = np.zeros_like(tracks[tra_i].live_state)
                one_track[:] = np.nan

                for j in range(window_radius, self.image_amount):
                    temp_array = tracks[tra_i].live_state[j - window_radius: j + window_radius + 1]
                    temp_array = temp_array[~np.isnan(temp_array)]

                    if (len(temp_array) > 1):
                        local_max = np.max(temp_array)
                        local_min = np.min(temp_array)
                        one_track[j] = local_max - local_min
                    else:
                        pass

                file.write("Max_min: ")
                for s in range(array_size):
                    file.write("%s, " % one_track[s])
                file.write(";\n")

                for k in range(len(one_track)):
                    if (one_track[k] > 0.2):
                        one_track[k] = 1
                    else:
                        one_track[k] = 0

                file.write("state: ")
                for s in range(array_size):
                    file.write("%s, " % one_track[s])
                file.write(";\n")

                one_track_str = str(one_track)
                one_track_str = one_track_str.replace("\n", "")
                one_track_str = one_track_str.replace("[", "")
                one_track_str = one_track_str.replace("]", "")

                sub_arrays = one_track_str.split('1.')
                max_dead_count = 0

                for m in range(1, len(sub_arrays) - 1, 1):
                    dead_count = sub_arrays[m].count('0.')
                    if (max_dead_count < dead_count):
                        max_dead_count = dead_count

                if (max_dead_count == 0 and one_track_str.count('1.') < 10):
                    print("The track is too stationary", one_track_str)
                    one_track[:] = -1
                    pass
                else:  # pad start
                    n = array_size - 1
                    while (not (one_track[n] == 1) and n > -1):
                        one_track[n] = -1
                        n = n - 1

                    while (n > -1):
                        one_track[n] = 1
                        n = n - 1
                    # pad end
                    last_dead_status = np.where(one_track == -1)[0][0]
                    # print("last_dead_status: ", last_dead_status)

                    # print(i, self.image_amount, last_dead_status, max_dead_count)

                    file.write("info: ")
                    if (self.image_amount - last_dead_status < max_dead_count):
                        file.write("%s, " % "image_count: " + str(self.image_amount) + ", last_dead_status: " + str(
                            last_dead_status) + ", max_dead_count: " + str(max_dead_count) + ";")

                        for p in range(last_dead_status, len(one_track), 1):
                            one_track[p] = 1
                    else:
                        pass
                    # print(np.where(one_track == -1))
                    file.write(";\n")

                file.write("ratio_result: ")
                for s in range(array_size):
                    file.write("%s, " % one_track[s])
                file.write(";\n")

                file.write("loc_var: ")
                for s in range(array_size):
                    file.write("%s, " % tracks[tra_i].loc_var[s])
                file.write(";\n")

                loc_var = np.where(tracks[tra_i].loc_var > 1300, 1, 0)

                file.write("bigger than 1500: ")
                for s in range(array_size):
                    file.write("%s, " % loc_var[s])
                file.write(";\n")

                for i in range(2, array_size - 2):
                    if (np.isnan(tracks[tra_i].loc_var[i - 2: i + 2]).any()):
                        pass
                    else:
                        if (np.mean(loc_var[i - 2: i + 2]) == 0):
                            for j in range(i, array_size - 2):
                                one_track[j] = -1

                file.write("final: ")
                for s in range(array_size):
                    file.write("%s, " % one_track[s])
                file.write(";\n")

                tracks[tra_i].live_state = one_track

            for j in range(0, self.image_amount):
                for i in range(0, len(tracks), 1):
                    cell_x = tracks[i].coordinates[j][0]
                    cell_y = tracks[i].coordinates[j][1]

                    if (cell_x > 0 and cell_y > 0):

                        if (tracks[i].live_state[j] == 1):
                            live_dead_table[j][0] = live_dead_table[j][0] + 1
                        elif (tracks[i].live_state[j] == -1):
                            live_dead_table[j][1] = live_dead_table[j][1] + 1
                        else:
                            pass

                # live_dead_table[j][2] = live_dead_table[j][0] + live_dead_table[j][1]

        file.close()

        if (not os.path.exists(outpath)):
            os.makedirs(outpath)

        np.savetxt(outpath + "live_dead_table.txt", live_dead_table, fmt='%d')

    def analyse_classification_3(self, outpath, frame_count):  # ratio, cell diff, area
        print("tracker save.")

        self.image_amount = frame_count
        window_radius = 4

        file = open(outpath + "file3_" + time.strftime("%d_%H_%M", time.localtime()) + ".txt", "w")

        live_dead_table = np.zeros((self.image_amount, 4), dtype=int)

        for tracks in (self.tracks, self.del_tracks):
            for tra_i in range(0, len(tracks), 1):
                file.write("track: %d;\n" % tracks[tra_i].track_id)

                file.write("coordinate: ")
                for j in range(array_size):
                    file.write("%s, " % tracks[tra_i].coordinates[j])
                    # file.write("%s, " % tracks[tra_i].coordinates[j * 2 + 1])
                file.write(";\n")

                file.write("ratio: ")
                for j in range(array_size):
                    file.write("%s, " % tracks[tra_i].live_state[j])
                file.write(";\n")

                one_track = np.zeros_like(tracks[tra_i].live_state)
                one_track[:] = np.nan

                for j in range(window_radius, self.image_amount):
                    temp_array = tracks[tra_i].live_state[j - window_radius: j + window_radius + 1]
                    temp_array = temp_array[~np.isnan(temp_array)]

                    if (len(temp_array) > 1):
                        local_max = np.max(temp_array)
                        local_min = np.min(temp_array)
                        one_track[j] = local_max - local_min
                    else:
                        pass

                file.write("Max_min: ")
                for s in range(array_size):
                    file.write("%s, " % one_track[s])
                file.write(";\n")

                for k in range(len(one_track)):
                    if (one_track[k] > 0.3):
                        one_track[k] = 1
                    else:
                        one_track[k] = 0

                file.write("state: ")
                for s in range(array_size):
                    file.write("%s, " % one_track[s])
                file.write(";\n")

                one_track_str = str(one_track)
                one_track_str = one_track_str.replace("\n", "")
                one_track_str = one_track_str.replace("[", "")
                one_track_str = one_track_str.replace("]", "")

                sub_arrays = one_track_str.split('1.')
                max_dead_count = 0

                for m in range(1, len(sub_arrays) - 1, 1):
                    dead_count = sub_arrays[m].count('0.')
                    if (max_dead_count < dead_count):
                        max_dead_count = dead_count

                if (max_dead_count == 0 and one_track_str.count('1.') < 10):
                    print("The track is too stationary", one_track_str)
                    one_track[:] = -1
                    pass
                else:  # pad start
                    n = array_size - 1
                    while (not (one_track[n] == 1) and n > -1):
                        one_track[n] = -1
                        n = n - 1

                    while (n > -1):
                        one_track[n] = 1
                        n = n - 1
                    # pad end
                    last_dead_status = np.where(one_track == -1)[0][0]
                    # print("last_dead_status: ", last_dead_status)

                    # print(i, self.image_amount, last_dead_status, max_dead_count)

                    file.write("info: ")
                    if (self.image_amount - last_dead_status < max_dead_count):
                        file.write("%s, " % "image_count: " + str(self.image_amount) + ", last_dead_status: " + str(
                            last_dead_status) + ", max_dead_count: " + str(max_dead_count) + ";")

                        for p in range(last_dead_status, len(one_track), 1):
                            one_track[p] = 1
                    else:
                        pass
                    # print(np.where(one_track == -1))
                    file.write(";\n")

                file.write("ratio_result: ")
                for s in range(array_size):
                    file.write("%s, " % one_track[s])
                file.write(";\n")

                file.write("cell diff: ")
                for j in range(array_size):
                    file.write("%s, " % tracks[tra_i].cell_diff[j])
                file.write(";\n")

                cell_diff_row = np.zeros_like(tracks[tra_i].cell_diff)
                cell_diff_row[:] = np.nan

                for j in range(window_radius, self.image_amount):
                    temp_array = tracks[tra_i].cell_diff[j - window_radius: j + window_radius + 1]
                    temp_array = temp_array[~np.isnan(temp_array)]

                    if (len(temp_array) > 1):
                        local_max = np.max(temp_array)
                        cell_diff_row[j] = local_max
                    else:
                        pass

                j = 0
                for j in range(len(cell_diff_row) - 1, -1, -1):
                    if cell_diff_row[j] > 180000:
                        break
                    else:
                        cell_diff_row[j] = -1

                for k in range(j + 1):
                    cell_diff_row[k] = 1

                file.write("cell diff result: ")
                for j in range(array_size):
                    file.write("%s, " % cell_diff_row[j])
                file.write(";\n")

                file.write("area: ")
                for j in range(array_size):
                    file.write("%s, " % tracks[tra_i].area[j])
                file.write(";\n")

                area_row = np.zeros_like(tracks[tra_i].area)
                area_row[:] = np.nan

                for j in range(window_radius, self.image_amount):
                    temp_array = tracks[tra_i].area[j - window_radius: j + window_radius + 1]
                    temp_array = temp_array[~np.isnan(temp_array)]

                    if (len(temp_array) > 1):
                        local_max = np.max(temp_array)
                        local_min = np.min(temp_array)
                        area_row[j] = local_max - local_min
                    else:
                        pass

                j = 0
                for j in range(len(area_row) - 1, -1, -1):
                    if area_row[j] > 150:
                        break
                    else:
                        area_row[j] = -1

                for k in range(j + 1):
                    area_row[k] = 1

                file.write("area result: ")
                for j in range(array_size):
                    file.write("%s, " % area_row[j])
                file.write(";\n")

                one_track = one_track + cell_diff_row + area_row
                # one_track = cell_diff_row * 3

                file.write("final: ")
                for s in range(array_size):
                    file.write("%s, " % one_track[s])
                file.write(";\n")

                tracks[tra_i].live_state = one_track

            for j in range(0, self.image_amount):
                for i in range(0, len(tracks), 1):
                    cell_x = tracks[i].coordinates[j][0]
                    cell_y = tracks[i].coordinates[j][1]

                    if (cell_x > 0 and cell_y > 0):

                        if (tracks[i].live_state[j] >= 1):
                            live_dead_table[j][0] = live_dead_table[j][0] + 1
                            live_dead_table[j][2] = live_dead_table[j][2] + tracks[i].area[j]
                        else:
                            live_dead_table[j][1] = live_dead_table[j][1] + 1
                            live_dead_table[j][3] = live_dead_table[j][3] + tracks[i].area[j]

                # live_dead_table[j][2] = live_dead_table[j][0] + live_dead_table[j][1]

        file.close()

        if (not os.path.exists(outpath)):
            os.makedirs(outpath)

        np.savetxt(outpath + "live_dead_table.txt", live_dead_table, fmt='%d')

    # I need to compare with Praneeth's video here. Postprocess is required, since the red mask is discontinuous.
    def analyse_classification_4(self, outpath, frame_count, gt_video_path, scale):
        print("tracker save.")

        self.image_amount = frame_count
        window_radius = 2

        file = open(outpath + "file3_" + time.strftime("%d_%H_%M", time.localtime()) + ".txt", "w")

        live_dead_table = np.zeros((self.image_amount, 4), dtype=int)

        count_tmp = 0
        # for tracks in (self.tracks, self.del_tracks):
        #     count_tmp = 0
        #     i = 0
        #     while(i < len(tracks)):
        #         count_tmp += 1
        #         if np.count_nonzero(tracks[i].coordinates > 0) < 38:
        #             tracks.remove(tracks[i])
        #         else:
        #             i += 1
        #     print(count_tmp, len(tracks))

        # print()
        # diff_thr = 7200000
        # area_thr = 960
        diff_thr = 1000000
        area_thr = 100

        for tracks in (self.tracks, self.del_tracks):
            for tra_i in range(0, len(tracks), 1):

                # if(np.count_nonzero(tracks[tra_i].coordinates > 0) < 38):
                #     del tracks[tra_i]

                # print("tra_i: ", tra_i)
                file.write("track: %d;\n" % tracks[tra_i].track_id)
                #
                # file.write("coordinate: ")
                # for j in range(array_size):
                #     file.write("%s, " % tracks[tra_i].coordinates[j])
                #     # file.write("%s, " % tracks[tra_i].coordinates[j * 2 + 1])
                # file.write(";\n")
                #
                # file.write("ratio: ")
                # for j in range(array_size):
                #     file.write("%s, " % tracks[tra_i].live_state[j])
                # file.write(";\n")
                #
                # one_track = np.zeros_like(tracks[tra_i].live_state)
                # one_track[:] = np.nan
                #
                # for j in range(window_radius, self.image_amount):
                #     temp_array = tracks[tra_i].live_state[j - window_radius: j + window_radius + 1]
                #     temp_array = temp_array[~np.isnan(temp_array)]
                #
                #     if (len(temp_array) > 1):
                #         local_max = np.max(temp_array)
                #         local_min = np.min(temp_array)
                #         one_track[j] = local_max - local_min
                #     else:
                #         pass
                #
                # file.write("Max_min: ")
                # for s in range(array_size):
                #     file.write("%s, " % one_track[s])
                # file.write(";\n")
                #
                # for k in range(len(one_track)):
                #     if (one_track[k] > 0.1):
                #         one_track[k] = 1
                #     else:
                #         one_track[k] = 0
                #
                # file.write("ratio ret 0: ")
                # for s in range(array_size):
                #     file.write("%s, " % one_track[s])
                # file.write(";\n")
                #
                file.write("cell diff: ")
                for j in range(array_size):
                    file.write("%s, " % tracks[tra_i].cell_diff[j])
                file.write(";\n")

                cell_diff_row = np.zeros_like(tracks[tra_i].cell_diff)
                cell_diff_row[:] = np.nan

                for j in range(window_radius, self.image_amount):
                    temp_array = tracks[tra_i].cell_diff[j - window_radius: j + window_radius + 1]
                    temp_array = temp_array[~np.isnan(temp_array)]

                    if (len(temp_array) > 0):
                        local_max = np.max(temp_array)
                        cell_diff_row[j] = local_max
                    else:
                        pass

                cell_diff_row = np.where(cell_diff_row > diff_thr, 1, 0)

                # flag = 0
                # for j in range(len(cell_diff_row) - 1, -1, -1):
                #     if cell_diff_row[j] > 600000:
                #         cell_diff_row[0:j+1] = 1
                #         cell_diff_row[j + 1:array_size] = -1
                #         flag = 1
                #         break
                # if flag == 0:
                #     cell_diff_row[:] = -1

                file.write("cell diff result: ")
                for j in range(array_size):
                    file.write("%s, " % cell_diff_row[j])
                file.write(";\n")

                file.write("area: ")
                for j in range(array_size):
                    file.write("%s, " % tracks[tra_i].area[j])
                file.write(";\n")

                area_row = np.zeros_like(tracks[tra_i].area)
                area_row[:] = np.nan

                for j in range(window_radius, self.image_amount):
                    temp_array = tracks[tra_i].area[j - window_radius: j + window_radius + 1]
                    temp_array = temp_array[~np.isnan(temp_array)]

                    if (len(temp_array) > 1):
                        local_max = np.max(temp_array)
                        local_min = np.min(temp_array)
                        area_row[j] = local_max - local_min
                    else:
                        pass

                # for j in range(len(area_row) - 1, -1, -1):
                #     if area_row[j] > 80:
                #         area_row[j] = 1
                #     else:
                #         area_row[j] = 0

                area_row = np.where(area_row > area_thr, 1, 0)

                        # flag = 0
                # for j in range(len(area_row) - 1, -1, -1):
                #     if area_row[j] > 80:
                #         area_row[0:j + 1] = 1
                #         area_row[j + 1:array_size] = -1
                #         flag = 1
                #         break
                #
                # if(flag == 0):
                #     area_row[:] = -1

                file.write("area result: ")
                for j in range(array_size):
                    file.write("%s, " % area_row[j])
                file.write(";\n")

                # file.write("overlap change: ")
                # for j in range(array_size):
                #     file.write("%s, " % tracks[tra_i].overlap_change[j])
                # file.write(";\n")
                #
                # overlap_change_row = np.zeros_like(tracks[tra_i].overlap_change)
                # overlap_change_row[:] = np.nan
                #
                # for j in range(window_radius, self.image_amount):
                #     temp_array = tracks[tra_i].overlap_change[j - window_radius: j + window_radius + 1]
                #     temp_array = temp_array[~np.isnan(temp_array)]
                #
                #     if (len(temp_array) > 0):
                #         local_max = np.max(temp_array)
                #         overlap_change_row[j] = local_max
                #     else:
                #         pass
                #
                # flag = 0
                # for j in range(len(overlap_change_row)):
                #     if overlap_change_row[j] < 10:
                #         overlap_change_row[0:j] = 1
                #         overlap_change_row[j:array_size] = -1
                #         flag = 1
                #         break
                #
                # if flag == 0:
                #     overlap_change_row[:] = 1
                #
                # file.write("overlap change result: ")
                # for j in range(array_size):
                #     file.write("%s, " % overlap_change_row[j])
                # file.write(";\n")

                for arr in [cell_diff_row, area_row]:
                    arr_str = str(arr)
                    arr_str = arr_str.replace("\n", "")
                    arr_str = arr_str.replace("[", "")
                    arr_str = arr_str.replace("]", "")

                    sub_arrays = arr_str.split('1')
                    max_dead_count = 0

                    for m in range(1, len(sub_arrays) - 1, 1):
                        dead_count = sub_arrays[m].count('0')
                        if (max_dead_count < dead_count):
                            max_dead_count = dead_count

                    if (max_dead_count == 0 and arr_str.count('1') < 10):
                        # print("The track is too stationary", arr_str)
                        arr[:] = -1
                        pass
                    else:  # pad start
                        flag = 0
                        for j in range(len(arr) - 1, -1, -1):
                            if arr[j] == 1:
                                arr[0:j+1] = 1
                                arr[j + 1:array_size] = -1
                                flag = 1
                                break
                        if flag == 0:
                            arr[:] = -1

                        # pad end
                        last_dead_status = np.where(arr == -1)[0][0]

                        if (self.image_amount - last_dead_status < max_dead_count):
                            print("cell live in the end.")
                            arr[:] = 1
                        else:
                            pass

                # one_track = one_track + cell_diff_row + area_row
                # one_track = area_row * 3
                # one_track = cell_diff_row + area_row
                one_track = area_row * 2

                file.write("final: ")
                for s in range(array_size):
                    file.write("%s, " % one_track[s])
                file.write(";\n")

                tracks[tra_i].live_state = one_track

        file.close()

        for j in range(0, self.image_amount):
            for tracks in (self.tracks, self.del_tracks):
                for i in range(0, len(tracks), 1):
                    cell_x = tracks[i].coordinates[j][0]
                    cell_y = tracks[i].coordinates[j][1]

                    if (cell_x > 0 and cell_y > 0):

                        if (tracks[i].live_state[j] > 1):
                            live_dead_table[j][0] = live_dead_table[j][0] + 1
                            live_dead_table[j][2] = live_dead_table[j][2] + tracks[i].area[j]
                        else:
                            live_dead_table[j][1] = live_dead_table[j][1] + 1
                            live_dead_table[j][3] = live_dead_table[j][3] + tracks[i].area[j]

                # live_dead_table[j][2] = live_dead_table[j][0] + live_dead_table[j][1]


        if (not os.path.exists(outpath)):
            os.makedirs(outpath)

        # np.savetxt(outpath + "live_dead_table_"+ str(window_radius) + "_" + str(diff_thr) + "_" + str(area_thr) + ".txt", live_dead_table, fmt='%d')
        np.savetxt(outpath + "live_dead_table.txt", live_dead_table, fmt='%d')

        vid = cv2.VideoCapture(gt_video_path)
        # skip = 3
        skip = frame_count - int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        print("skip2: ", skip)

        pad_wid = 200
        for i in range(0, array_size):
            if (i >= skip):
                ret, gt_frame = vid.read()

                if not ret:
                    break

                if(os.path.exists(outpath + "coord.txt")):# False and
                    self.coord = np.loadtxt(outpath + "coord.txt")
                    coord = self.coord[i * 2: i * 2 + 2]
                    coord = coord.astype(int)
                else:
                    image_path = outpath + "input_images/" + "t" + "{0:0=3d}".format(i) + ".tif"
                    if (not os.path.exists(image_path)):
                        # print("file not exist: ", image_path)
                        break
                    else:
                        # print(frame_count, image_path)
                        pass

                    frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    d0 = frame.shape[0] >> 2
                    d1 = frame.shape[1] >> 2
                    template = gt_frame[d0:3 * d0, d1:3 * d1, 0]
                    ret = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF)
                    resu = cv2.minMaxLoc(ret)

                    coord = [resu[2][1] - d0, resu[2][0] - d1]
                    coord = -np.array(coord)
                    print(coord)
                    self.coord[i * 2] = coord[0]
                    self.coord[i * 2 + 1] = coord[1]

                gt_frame_pad = cv2.copyMakeBorder(gt_frame, pad_wid, pad_wid, pad_wid, pad_wid, cv2.BORDER_CONSTANT)
                gt_frame = gt_frame_pad[
                           pad_wid + coord[0]:pad_wid + coord[0] + gt_frame.shape[0],
                           pad_wid + coord[1]:pad_wid + coord[1] + gt_frame.shape[1]]

                # cv2.imshow("gt_frame", gt_frame)
                # cv2.waitKey()

                # frame_0 = frame[:, :, 0]
                frame_1 = gt_frame[:, :, 1]
                frame_2 = gt_frame[:, :, 2]

                red = frame_2.astype(np.float) - frame_1.astype(np.float)
                red_uint8 = np.clip(red, 0, 255).astype(np.uint8)

                for tracks in (self.del_tracks, self.tracks):
                    for idx in range(len(tracks)):
                        x3 = tracks[idx].coordinates[i][0]
                        y3 = tracks[idx].coordinates[i][1]

                        if((not (np.isnan(x3) or np.isnan(y3))) and red_uint8[int(y3)][int(x3)] > 0):
                            tracks[idx].g_truth[i] = 1
                        else:
                            tracks[idx].g_truth[i] = -1
                # print("Hello")

        if (not os.path.exists(outpath + "coord.txt")):
            np.savetxt(outpath + "coord.txt", self.coord)

        f_g_truth = open(outpath + "g_truth" + time.strftime("%d_%H_%M", time.localtime()) + ".txt", 'w')
        f_die_time = open(outpath + "die_time" + time.strftime("%d_%H_%M", time.localtime()) + ".txt", 'w')

        for tracks in (self.del_tracks, self.tracks):
            for idx in range(len(tracks)):
                flag = 0
                for i in range(self.image_amount - 1, -1, -1):
                    if(tracks[idx].g_truth[i] == 1):
                        tracks[idx].g_truth[0:i] = 1
                        # print(tracks[idx].track_id, tracks[idx].g_truth)
                        flag = 1
                        break

                if flag:
                    if(i == self.image_amount - 1):
                        f_g_truth.write(str(tracks[idx].track_id) + " " + str(1000) + "\n")
                    else:
                        f_g_truth.write(str(tracks[idx].track_id) + " " + str(i + 1) + "\n")
                else:
                    f_g_truth.write(str(tracks[idx].track_id) + " " + str(-1) + "\n")

                flag = 0
                for i in range(self.image_amount - 1, -1, -1):
                    if(tracks[idx].live_state[i] >= 1):
                        flag = 1
                        break

                if flag:
                    if(i == self.image_amount - 1):
                        f_die_time.write(str(tracks[idx].track_id) + " " + str(1000) + "\n")
                    else:
                        f_die_time.write(str(tracks[idx].track_id) + " " + str(i + 1) + "\n")
                else:
                    f_die_time.write(str(tracks[idx].track_id) + " " + str(-1) + "\n")


        f_g_truth.close()
        f_die_time.close()

        # for tracks in (self.tracks, self.del_tracks):
        #     for tra_i in range(0, len(tracks), 1):
        #         average = [mean(tracks[tra_i].shift[:, 0]), mean(tracks[tra_i].shift[:, 1])]
        #         print("average", average)
        #         print(tracks[tra_i].shift)
        #         tracks[tra_i].shift = tracks[tra_i].shift - average
        #         print(tracks[tra_i].coordinates)
        #         print(tracks[tra_i].shift)
        #         tracks[tra_i].shift = tracks[tra_i].shift / scale
        #         tracks[tra_i].coordinates[:,0] += tracks[tra_i].shift[:,1]
        #         tracks[tra_i].coordinates[:,1] += tracks[tra_i].shift[:,0]
        #         tracks[tra_i].coordinates += 0.5
        #         print(tracks[tra_i].coordinates)

    def analyse_classification_5(self, outpath, frame_count, gt_video_path, scale, Beacon):
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
                if np.count_nonzero(tracks[i].coordinates > 0) < 20:
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

        diff_thr = 636.17 * self.cell_core_r_mean * self.cell_core_r_mean * self.background_pixel
        area_thr = 3.82 * self.cell_core_r_mean * self.cell_core_r_mean

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


                # for arr in [cell_diff_row, area_row]:
                #     arr_str = str(arr)
                #     arr_str = arr_str.replace("\n", "")
                #     arr_str = arr_str.replace("[", "")
                #     arr_str = arr_str.replace("]", "")
                #
                #     sub_arrays = arr_str.split('1')
                #     max_dead_count = 0
                #
                #     for m in range(1, len(sub_arrays) - 1, 1):
                #         dead_count = sub_arrays[m].count('0')
                #         if (max_dead_count < dead_count):
                #             max_dead_count = dead_count
                #
                #     if (max_dead_count == 0 and arr_str.count('1') < 10):
                #         # print("The track is too stationary", arr_str)
                #         arr[:] = -1
                #         pass
                #     else:  # pad start
                #         flag = 0
                #         for j in range(len(arr) - 1, -1, -1):
                #             if arr[j] == 1:
                #                 arr[0:j+1] = 1
                #                 arr[j + 1:array_size] = -1
                #                 flag = 1
                #                 break
                #         if flag == 0:
                #             arr[:] = -1
                #
                #         # pad end
                #         last_dead_status = np.where(arr == -1)[0][0]
                #
                #         if (self.image_amount - last_dead_status < max_dead_count):
                #             print("cell live in the end.")
                #             arr[:] = 1
                #         else:
                #             pass

                # one_track = one_track + cell_diff_row + area_row
                # one_track = area_row * 3
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

                # y_p = y[~(np.isnan(y))]
                # x_p = np.nonzero(y > 0)[0]
                # x = np.arange(0, len(y), 1)
                # new_y = np.interp(x, x_p, y_p)

                sub_y_idx = np.where(y > 0)[0]
                y_p = y[sub_y_idx]
                x = np.arange(sub_y_idx[0], sub_y_idx[-1], 1)
                part_new_y = np.interp(x, sub_y_idx, y_p)
                new_y = y.copy()
                new_y[sub_y_idx[0]:sub_y_idx[-1]] = part_new_y


                # print("track_id: ", tracks[i].track_id)
                # print(*(tracks[i].live_state))
                # print(*(tracks[i].area))
                # print(*new_y)

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
                    cell_x = tracks[i].coordinates[j][0]
                    cell_y = tracks[i].coordinates[j][1]

                    if (cell_x > 0 and cell_y > 0):

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


        if(gt == True and os.path.exists(gt_video_path)):
        # if(gt == True and os.path.exists(gt_video_path)):
            vid = cv2.VideoCapture(gt_video_path)
            # skip = 3
            skip = frame_count - int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            print("skip2: ", skip)

            pad_wid = 200
            for i in range(0, array_size):
                if (i >= skip):
                    ret, gt_frame = vid.read()

                    if not ret:
                        break

                    # if(os.path.exists(outpath + "coord.txt")):# False and
                    #     if(self.coord[0] == np.nan):
                    #         self.coord = np.loadtxt(outpath + "coord.txt")
                    #     coord = self.coord[i * 2: i * 2 + 2]
                    #     coord = coord.astype(int)
                    # else:
                    image_path = outpath + "images_ucf/Beacon_" + str(Beacon) + "/t" + "{0:0=3d}".format(i) + ".tif"
                    # image_path = outpath + "input_images/" + "t" + "{0:0=3d}".format(i) + ".tif"
                    if (not os.path.exists(image_path)):
                        # print("file not exist: ", image_path)
                        break
                    else:
                        # print(frame_count, image_path)
                        pass

                    frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    d0 = frame.shape[0] >> 2
                    d1 = frame.shape[1] >> 2
                    template = gt_frame[d0:3 * d0, d1:3 * d1, 0]
                    ret = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF)
                    resu = cv2.minMaxLoc(ret)

                    coord = [resu[2][1] - d0, resu[2][0] - d1]
                    coord = -np.array(coord)
                    # print(coord)
                    self.coord[i * 2] = coord[0]
                    self.coord[i * 2 + 1] = coord[1]
                    #else end

                    gt_frame_pad = cv2.copyMakeBorder(gt_frame, pad_wid, pad_wid, pad_wid, pad_wid, cv2.BORDER_CONSTANT)
                    gt_frame = gt_frame_pad[
                               pad_wid + coord[0]:pad_wid + coord[0] + gt_frame.shape[0],
                               pad_wid + coord[1]:pad_wid + coord[1] + gt_frame.shape[1]]

                    # cv2.imshow("gt_frame", gt_frame)
                    # cv2.waitKey()

                    # frame_0 = frame[:, :, 0]
                    frame_1 = gt_frame[:, :, 1]
                    frame_2 = gt_frame[:, :, 2]

                    red = frame_2.astype(np.float) - frame_1.astype(np.float)
                    red_uint8 = np.clip(red, 0, 255).astype(np.uint8)

                    for tracks in (self.tracks, self.del_tracks):
                        for idx in range(len(tracks)):
                            x3 = tracks[idx].coordinates[i][0]
                            y3 = tracks[idx].coordinates[i][1]

                            if(np.isnan(x3) or np.isnan(y3)):
                                tracks[idx].g_truth[i] = np.nan
                            else:
                                if(red_uint8[int(y3)][int(x3)] > red_level):
                                    tracks[idx].g_truth[i] = 1
                                else:
                                    tracks[idx].g_truth[i] = -1
                    # print("Hello")

            # if ((not os.path.exists(outpath + "coord.txt")) and self.coord[0] != np.nan):
            # if (self.coord[0] != np.nan):
            np.savetxt(outpath + "info_ucf/Beacon_" + str(Beacon) + "_coord.txt", self.coord)

            f_g_truth = open(outpath + "info_ucf/Beacon_" + str(Beacon) + "_g_truth.txt", 'w')
            f_die_time = open(outpath + "info_ucf/Beacon_" + str(Beacon) + "_die_time.txt", 'w')
            # f_g_truth = open(outpath + "g_truth" + time.strftime("%d_%H_%M", time.localtime()) + ".txt", 'w')
            # f_die_time = open(outpath + "die_time" + time.strftime("%d_%H_%M", time.localtime()) + ".txt", 'w')

            for tracks in (self.tracks, self.del_tracks):
                for idx in range(len(tracks)):
                    # flag = 0
                    # for i in range(self.image_amount - 1, -1, -1):
                    #     if(tracks[idx].g_truth[i] == 1):
                    #         tracks[idx].g_truth[0:i] = 1
                    #         # print(tracks[idx].track_id, tracks[idx].g_truth)
                    #         flag = 1
                    #         break
                    #

                    flag = 0
                    for i in range(self.image_amount):
                        # tmp_arr = tracks[idx].g_truth[i:i + stable_period]

                        if(mean(tracks[idx].g_truth[i:i + stable_period]) == -1):
                            # print(idx, tracks[idx].track_id, i, tracks[idx].g_truth)
                            tracks[idx].g_truth[:i] = 1
                            tracks[idx].g_truth[i:] = -1
                            flag = 1
                            break

                    tra_len = np.count_nonzero(tracks[idx].coordinates > 0) / 2

                    if flag:
                        if(i == 0):
                            f_g_truth.write(str(tracks[idx].track_id) + " " + str(-1) + " " + str(tra_len) + "\n")
                            # f_g_truth.write(str(tracks[idx].track_id) + " " + str(0) + " " + str(tra_len) + "\n")
                        else:
                            f_g_truth.write(str(tracks[idx].track_id) + " " + str(i) + " " + str(tra_len) + "\n")
                        # f_g_truth.write(str(tracks[idx].track_id) + " " + str(i) + "\n")
                    else:
                        # f_g_truth.write(str(tracks[idx].track_id) + " " + str(1000) + " " + str(tra_len) + "\n")
                        f_g_truth.write(str(tracks[idx].track_id) + " " + str(self.image_amount) + " " + str(tra_len) + "\n")

                    flag = 0
                    for i in range(self.image_amount - 1, -1, -1):
                        if(tracks[idx].live_state[i] > 1):
                            flag = 1
                            break

                    if flag:
                        if(i == self.image_amount - 1):
                            # f_die_time.write(str(tracks[idx].track_id) + " " + str(1000) + "\n")
                            f_die_time.write(str(tracks[idx].track_id) + " " + str(self.image_amount) + "\n")
                            # f_die_time.write(str(tracks[idx].track_id) + " " + str(self.image_amount) + "\n")
                        else:
                            f_die_time.write(str(tracks[idx].track_id) + " " + str(i + 1) + "\n")
                    else:
                        f_die_time.write(str(tracks[idx].track_id) + " " + str(-1) + "\n")
                        # f_die_time.write(str(tracks[idx].track_id) + " " + str(0) + "\n")


            f_g_truth.close()
            f_die_time.close()

        f_cell_tracks = open(outpath + "info_ucf/Beacon_" + str(Beacon) + "_cell_tracks.txt", 'w')
        for tracks in (self.tracks, self.del_tracks):
            for idx in range(len(tracks)):
                print("track_id:", tracks[idx].track_id, sep='', file=f_cell_tracks)
                for i in range(self.image_amount):
                    if(tracks[idx].full_trace[i] != None):
                        # print(*(tracks[idx].full_trace[i][[3, 10, 11]]), sep=',', file=f_cell_tracks)
                        print("%.2f,%.2f,%.2f,%.2f,%.2f"%tuple(tracks[idx].full_trace[i][[0, 1, 3, 10, 11]]), sep=',', file=f_cell_tracks)
                    else:
                        print(np.nan, np.nan, np.nan, sep=',', file=f_cell_tracks)
        f_cell_tracks.close()

        # for tracks in (self.tracks, self.del_tracks):
        #     for tra_i in range(0, len(tracks), 1):
        #         average = [mean(tracks[tra_i].shift[:, 0]), mean(tracks[tra_i].shift[:, 1])]
        #         print("average", average)
        #         print(tracks[tra_i].shift)
        #         tracks[tra_i].shift = tracks[tra_i].shift - average
        #         print(tracks[tra_i].coordinates)
        #         print(tracks[tra_i].shift)
        #         tracks[tra_i].shift = tracks[tra_i].shift / scale
        #         tracks[tra_i].coordinates[:,0] += tracks[tra_i].shift[:,1]
        #         tracks[tra_i].coordinates[:,1] += tracks[tra_i].shift[:,0]
        #         tracks[tra_i].coordinates += 0.5
        #         print(tracks[tra_i].coordinates)

    def analyse_classification_6(self, outpath, frame_count, gt_video_path, scale, Beacon):
        # print("tracker save.")
        self.image_amount = frame_count
        window_radius = 4

        file = None
        gt = True

        # make it comment
        file = open(outpath + "/info_ucf/file3_" + str(Beacon) + time.strftime("_%d_%H_%M", time.localtime()) + ".txt", "w")

        live_dead_table = np.zeros((self.image_amount, 4))
        # live_dead_table = np.zeros((self.image_amount, 4), dtype=int)

        for tracks in (self.tracks, self.del_tracks):
            count_tmp = 0
            i = 0
            while(i < len(tracks)):
                count_tmp += 1
                if np.count_nonzero(tracks[i].coordinates > 0) < 20:
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

        # diff_thr = 636.17 * self.cell_core_r_mean * self.cell_core_r_mean * self.background_pixel * 0.5
        # area_thr = 3.82 * self.cell_core_r_mean * self.cell_core_r_mean

        diff_thr = 212.7 * self.cell_core_r_mean * self.cell_core_r_mean * self.background_pixel
        area_thr = 4.42 * self.cell_core_r_mean * self.cell_core_r_mean

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
                        cell_diff_row[j] = np.mean(temp_array)
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
                        area_row[j] = np.mean(temp_array)
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

                for arr_ex in [[cell_diff_row, diff_thr], [area_row, area_thr]]:
                    arr = arr_ex[0]
                    flag = 0
                    for j in range(len(arr)):
                        if arr[j] == 0:# find the death 0
                            # print(j, arr)
                            arr[j:] = 0
                            # arr[:j] = 1
                            k = j
                            while k >= 0:
                                if(arr[k] > 0):
                                    break
                                else:
                                    k = k - 1

                            if k == -1:
                                arr[:] = 0
                            else:
                                tmp = np.where(arr[:k + 1] > 3 * arr_ex[1])
                                if(len(tmp[0]) == 0):
                                    arr[:] = 0
                                else:
                                    arr[:k + 1] = 1
                                    arr[k + 1:] = 0

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


                # for arr in [cell_diff_row, area_row]:
                #     arr_str = str(arr)
                #     arr_str = arr_str.replace("\n", "")
                #     arr_str = arr_str.replace("[", "")
                #     arr_str = arr_str.replace("]", "")
                #
                #     sub_arrays = arr_str.split('1')
                #     max_dead_count = 0
                #
                #     for m in range(1, len(sub_arrays) - 1, 1):
                #         dead_count = sub_arrays[m].count('0')
                #         if (max_dead_count < dead_count):
                #             max_dead_count = dead_count
                #
                #     if (max_dead_count == 0 and arr_str.count('1') < 10):
                #         # print("The track is too stationary", arr_str)
                #         arr[:] = -1
                #         pass
                #     else:  # pad start
                #         flag = 0
                #         for j in range(len(arr) - 1, -1, -1):
                #             if arr[j] == 1:
                #                 arr[0:j+1] = 1
                #                 arr[j + 1:array_size] = -1
                #                 flag = 1
                #                 break
                #         if flag == 0:
                #             arr[:] = -1
                #
                #         # pad end
                #         last_dead_status = np.where(arr == -1)[0][0]
                #
                #         if (self.image_amount - last_dead_status < max_dead_count):
                #             print("cell live in the end.")
                #             arr[:] = 1
                #         else:
                #             pass

                # one_track = one_track + cell_diff_row + area_row
                # one_track = area_row * 3
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

                # y_p = y[~(np.isnan(y))]
                # x_p = np.nonzero(y > 0)[0]
                # x = np.arange(0, len(y), 1)
                # new_y = np.interp(x, x_p, y_p)

                sub_y_idx = np.where(y > 0)[0]
                y_p = y[sub_y_idx]
                x = np.arange(sub_y_idx[0], sub_y_idx[-1], 1)
                part_new_y = np.interp(x, sub_y_idx, y_p)
                new_y = y.copy()
                new_y[sub_y_idx[0]:sub_y_idx[-1]] = part_new_y


                # print("track_id: ", tracks[i].track_id)
                # print(*(tracks[i].live_state))
                # print(*(tracks[i].area))
                # print(*new_y)

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
                    cell_x = tracks[i].coordinates[j][0]
                    cell_y = tracks[i].coordinates[j][1]

                    if (cell_x > 0 and cell_y > 0):

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


        if(gt == True and os.path.exists(gt_video_path)):
        # if(gt == True and os.path.exists(gt_video_path)):
            vid = cv2.VideoCapture(gt_video_path)
            # skip = 3
            skip = frame_count - int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            print("skip2: ", skip)

            pad_wid = 200
            for i in range(0, array_size):
                if (i >= skip):
                    ret, gt_frame = vid.read()

                    if not ret:
                        break

                    # if(os.path.exists(outpath + "coord.txt")):# False and
                    #     if(self.coord[0] == np.nan):
                    #         self.coord = np.loadtxt(outpath + "coord.txt")
                    #     coord = self.coord[i * 2: i * 2 + 2]
                    #     coord = coord.astype(int)
                    # else:
                    image_path = outpath + "images_ucf/Beacon_" + str(Beacon) + "/t" + "{0:0=3d}".format(i) + ".tif"
                    # image_path = outpath + "input_images/" + "t" + "{0:0=3d}".format(i) + ".tif"
                    if (not os.path.exists(image_path)):
                        # print("file not exist: ", image_path)
                        break
                    else:
                        # print(frame_count, image_path)
                        pass

                    frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    # d0 = frame.shape[0] >> 2
                    # d1 = frame.shape[1] >> 2
                    # template = gt_frame[d0:3 * d0, d1:3 * d1, 0]

                    d0 = frame.shape[0] >> 3
                    d1 = frame.shape[1] >> 3
                    template = gt_frame[d0:7 * d0, d1:7 * d1, 0]

                    ret = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF)
                    resu = cv2.minMaxLoc(ret)

                    coord = [resu[2][1] - d0, resu[2][0] - d1]
                    coord = -np.array(coord)
                    # print(coord)
                    self.coord[i * 2] = coord[0]
                    self.coord[i * 2 + 1] = coord[1]
                    #else end

                    gt_frame_pad = cv2.copyMakeBorder(gt_frame, pad_wid, pad_wid, pad_wid, pad_wid, cv2.BORDER_CONSTANT)
                    gt_frame = gt_frame_pad[
                               pad_wid + coord[0]:pad_wid + coord[0] + gt_frame.shape[0],
                               pad_wid + coord[1]:pad_wid + coord[1] + gt_frame.shape[1]]

                    # cv2.imshow("gt_frame", gt_frame)
                    # cv2.waitKey()

                    # frame_0 = frame[:, :, 0]
                    frame_1 = gt_frame[:, :, 1]
                    frame_2 = gt_frame[:, :, 2]

                    red = frame_2.astype(np.float) - frame_1.astype(np.float)
                    red_uint8 = np.clip(red, 0, 255).astype(np.uint8)

                    for tracks in (self.tracks, self.del_tracks):
                        for idx in range(len(tracks)):
                            x3 = tracks[idx].coordinates[i][0]
                            y3 = tracks[idx].coordinates[i][1]

                            if(np.isnan(x3) or np.isnan(y3)):
                                tracks[idx].g_truth[i] = np.nan
                            else:
                                if(red_uint8[int(y3)][int(x3)] > red_level):
                                    tracks[idx].g_truth[i] = 1
                                else:
                                    tracks[idx].g_truth[i] = -1
                    # print("Hello")

            # if ((not os.path.exists(outpath + "coord.txt")) and self.coord[0] != np.nan):
            # if (self.coord[0] != np.nan):
            np.savetxt(outpath + "info_ucf/Beacon_" + str(Beacon) + "_coord.txt", self.coord)

            f_g_truth = open(outpath + "info_ucf/Beacon_" + str(Beacon) + "_g_truth.txt", 'w')
            f_die_time = open(outpath + "info_ucf/Beacon_" + str(Beacon) + "_die_time.txt", 'w')
            # f_g_truth = open(outpath + "g_truth" + time.strftime("%d_%H_%M", time.localtime()) + ".txt", 'w')
            # f_die_time = open(outpath + "die_time" + time.strftime("%d_%H_%M", time.localtime()) + ".txt", 'w')

            for tracks in (self.tracks, self.del_tracks):
                for idx in range(len(tracks)):
                    # flag = 0
                    # for i in range(self.image_amount - 1, -1, -1):
                    #     if(tracks[idx].g_truth[i] == 1):
                    #         tracks[idx].g_truth[0:i] = 1
                    #         # print(tracks[idx].track_id, tracks[idx].g_truth)
                    #         flag = 1
                    #         break
                    #

                    # for ground truth SFP start

                    # if(tracks[idx].track_id == 278):
                    #     print("track id: ", tracks[idx].track_id, tracks[idx].g_truth)

                    # tracks[idx].g_truth = - tracks[idx].g_truth #for RFP ground truth
                    # for ground truth SFP end

                    flag = 0
                    for i in range(self.image_amount):
                        # tmp_arr = tracks[idx].g_truth[i:i + stable_period]

                        if(mean(tracks[idx].g_truth[i:i + stable_period]) == -1):
                            # print(idx, tracks[idx].track_id, i, tracks[idx].g_truth)
                            tracks[idx].g_truth[:i] = 1
                            tracks[idx].g_truth[i:] = -1
                            flag = 1
                            break


                    tra_len = np.count_nonzero(tracks[idx].coordinates > 0) / 2

                    if flag:
                        if(i == 0):
                            f_g_truth.write(str(tracks[idx].track_id) + " " + str(-1) + " " + str(tra_len) + "\n")
                            # f_g_truth.write(str(tracks[idx].track_id) + " " + str(0) + " " + str(tra_len) + "\n")
                        else:
                            f_g_truth.write(str(tracks[idx].track_id) + " " + str(i) + " " + str(tra_len) + "\n")
                        # f_g_truth.write(str(tracks[idx].track_id) + " " + str(i) + "\n")
                    else:
                        f_g_truth.write(str(tracks[idx].track_id) + " " + str(1000) + " " + str(tra_len) + "\n")
                        # f_g_truth.write(str(tracks[idx].track_id) + " " + str(self.image_amount) + " " + str(tra_len) + "\n")
                        tracks[idx].g_truth[:] = 1

                    flag = 0
                    for i in range(self.image_amount - 1, -1, -1):
                        if(tracks[idx].live_state[i] > 1):
                            flag = 1
                            break

                    if flag:
                        if(i == self.image_amount - 1):
                            f_die_time.write(str(tracks[idx].track_id) + " " + str(1000) + "\n")
                            # f_die_time.write(str(tracks[idx].track_id) + " " + str(self.image_amount) + "\n")
                            # f_die_time.write(str(tracks[idx].track_id) + " " + str(self.image_amount) + "\n")
                        else:
                            f_die_time.write(str(tracks[idx].track_id) + " " + str(i + 1) + "\n")
                    else:
                        f_die_time.write(str(tracks[idx].track_id) + " " + str(-1) + "\n")
                        # f_die_time.write(str(tracks[idx].track_id) + " " + str(0) + "\n")


            f_g_truth.close()
            f_die_time.close()

        f_cell_tracks = open(outpath + "info_ucf/Beacon_" + str(Beacon) + "_cell_tracks.txt", 'w')
        for tracks in (self.tracks, self.del_tracks):
            for idx in range(len(tracks)):
                print("track_id:", tracks[idx].track_id, sep='', file=f_cell_tracks)
                for i in range(self.image_amount):
                    if(tracks[idx].full_trace[i] != None):
                        # print(*(tracks[idx].full_trace[i][[3, 10, 11]]), sep=',', file=f_cell_tracks)
                        print("%.2f,%.2f,%.2f,%.2f,%.2f"%tuple(tracks[idx].full_trace[i][[0, 1, 3, 10, 11]]), sep=',', file=f_cell_tracks)
                    else:
                        print(np.nan, np.nan, np.nan, np.nan, np.nan, sep=',', file=f_cell_tracks)
        f_cell_tracks.close()

        # for tracks in (self.tracks, self.del_tracks):
        #     for tra_i in range(0, len(tracks), 1):
        #         average = [mean(tracks[tra_i].shift[:, 0]), mean(tracks[tra_i].shift[:, 1])]
        #         print("average", average)
        #         print(tracks[tra_i].shift)
        #         tracks[tra_i].shift = tracks[tra_i].shift - average
        #         print(tracks[tra_i].coordinates)
        #         print(tracks[tra_i].shift)
        #         tracks[tra_i].shift = tracks[tra_i].shift / scale
        #         tracks[tra_i].coordinates[:,0] += tracks[tra_i].shift[:,1]
        #         tracks[tra_i].coordinates[:,1] += tracks[tra_i].shift[:,0]
        #         tracks[tra_i].coordinates += 0.5
        #         print(tracks[tra_i].coordinates)

    # max-min
    def analyse_classification_7(self, outpath, frame_count, gt_video_path, scale, Beacon):
        # print("tracker save.")
        self.image_amount = frame_count
        window_radius = 4

        file = None
        gt = True

        # make it comment
        file = open(outpath + "/info_ucf/file3_" + str(Beacon) + time.strftime("_%d_%H_%M", time.localtime()) + ".txt", "w")

        live_dead_table = np.zeros((self.image_amount, 4))
        # live_dead_table = np.zeros((self.image_amount, 4), dtype=int)

        for tracks in (self.tracks, self.del_tracks):
            count_tmp = 0
            i = 0
            while(i < len(tracks)):
                count_tmp += 1
                if np.count_nonzero(tracks[i].coordinates > 0) < 20:
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

        # diff_thr = 636.17 * self.cell_core_r_mean * self.cell_core_r_mean * self.background_pixel * 0.5
        # area_thr = 3.82 * self.cell_core_r_mean * self.cell_core_r_mean

        diff_thr = 212.7 * self.cell_core_r_mean * self.cell_core_r_mean * self.background_pixel
        area_thr = 4.42 * self.cell_core_r_mean * self.cell_core_r_mean

        # print("diff_thr, area_thr: ", self.cell_core_r, self.background_pixel, diff_thr, area_thr)
        for tracks in (self.tracks, self.del_tracks):
            for tra_i in range(0, len(tracks), 1):

                # if(np.count_nonzero(tracks[tra_i].coordinates > 0) < 38):
                #     del tracks[tra_i]

                if(file):
                    file.write("track: %d;\n" % tracks[tra_i].track_id)
                    file.write("cell_diff, ")
                    for j in range(self.image_amount):
                        file.write("%s, " % tracks[tra_i].cell_diff[j])
                    file.write(";\n")

                cell_diff_row = tracks[tra_i].cell_diff.copy()
                # cell_diff_row = np.zeros_like(tracks[tra_i].cell_diff)
                # cell_diff_row[:] = np.nan

                # for j in range(window_radius, self.image_amount):
                #     temp_array = tracks[tra_i].cell_diff[j - window_radius: j + window_radius + 1]
                #     temp_array = temp_array[~np.isnan(temp_array)]
                #
                #     if (len(temp_array) > window_radius * 1.5):
                #         cell_diff_row[j] = np.mean(temp_array)
                #     else:
                #         pass

                # if (file):
                #     file.write("cell_diff_mean, ")
                #     for j in range(self.image_amount):
                #         file.write("%s, " % cell_diff_row[j])
                #     file.write(";\n")

                # cell_diff_row = np.where(cell_diff_row < diff_thr, 0, cell_diff_row)

#################################################################### start

                tmp = np.where(cell_diff_row[:] > 3 * diff_thr)
                if (False and len(tmp[0]) == 0):
                    loc = 0
                else:

                    max_v = np.nan
                    min_v = np.nan
                    diff_mm = np.zeros_like(cell_diff_row)
                    diff_mm_der = np.zeros_like(cell_diff_row)
                    diff_mm[:] = np.nan
                    diff_mm_der[:] = np.nan
                    for i in range(self.image_amount - 1, -1, -1):
                        if(cell_diff_row[i] > 0):
                            if(np.isnan(max_v) or cell_diff_row[i] > max_v):
                                max_v = cell_diff_row[i]

                            if(np.isnan(min_v) or cell_diff_row[i] < min_v):
                                min_v = cell_diff_row[i]

                            diff_mm[i] = max_v - min_v

                    diff_mm_der[1:] = np.abs(np.diff(diff_mm))
                    loc = np.nan
                    try:
                        loc = np.nanargmax(diff_mm_der)
                    except ValueError as e:
                        if (e.args[0] == 'All-NaN slice encountered'):
                            pass
                        else:
                            print("Qibing error: ", e)
                            exit()

#################################################################### end

                # print("qibing: ", cell_diff_row, diff_mm, diff_mm_der, loc)

                if (file):
                    file.write("cell_diff_max_min, ")
                    for j in range(self.image_amount):
                        file.write("%s, " % diff_mm[j])
                    file.write(";\n")

                    file.write("cell_diff_der, ")
                    for j in range(self.image_amount):
                        file.write("%s, " % diff_mm_der[j])
                    file.write(";\n")

                    file.write("%s, " % loc)
                    file.write(";\n")

                    file.write("area, ")
                    for j in range(self.image_amount):
                        file.write("%s, " % tracks[tra_i].area[j])
                    file.write(";\n")

                # area_row = np.zeros_like(tracks[tra_i].area)
                # area_row[:] = np.nan

                area_diff = np.zeros(array_size)
                area_diff[1:] = np.abs(np.diff(tracks[tra_i].area))

                if (file):
                    file.write("area_diff, ")
                    for j in range(self.image_amount):
                        file.write("%s, " % area_diff[j])
                    file.write(";\n")


                # for j in range(window_radius, self.image_amount):
                #     temp_array = area_diff[j - window_radius: j + window_radius + 1]
                #     temp_array = temp_array[~np.isnan(temp_array)]
                #
                #     if (len(temp_array) > window_radius * 1.5):
                #         tmp[j] = np.mean(temp_array)
                #     else:
                #         pass

                # area_diff = tmp
                # if (file):
                #     file.write("area_diff_mean, ")
                #     for j in range(array_size):
                #         file.write("%s, " % area_diff[j])
                #     file.write(";\n")

                # area_row = np.where(area_row < area_thr, 0, area_row)

                #################################################################### start
                tmp = np.where(area_diff[:] > 3 * area_thr)
                if (len(tmp[0]) == 0):
                    loc_area = 0
                else:
                    max_v = np.nan
                    min_v = np.nan
                    area_diff_mm = np.zeros_like(area_diff)
                    area_diff_mm[:] = np.nan
                    area_diff_mm_der = np.zeros_like(area_diff)
                    area_diff_mm_der[:] = np.nan
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
                        loc_area = np.nanargmax(area_diff_mm_der)
                    except ValueError as e:
                        if (e.args[0] == 'All-NaN slice encountered'):
                            pass
                        else:
                            print("error: ", e)
                            exit()

                if (loc_area < loc):
                    loc = loc_area

                # tmp = np.where(arr[:k + 1] > 3 * arr_ex[1])
                # if (len(tmp[0]) == 0):
                #     arr[:] = 0

                #################################################################### end

                if (file):

                    file.write("area_diff_max_min, ")
                    for j in range(self.image_amount):
                        file.write("%s, " % area_diff_mm[j])
                    file.write(";\n")

                    file.write("area_diff_max_min_der, ")
                    for j in range(self.image_amount):
                        file.write("%s, " % area_diff_mm_der[j])
                    file.write(";\n")

                    file.write("%s, " % loc_area)
                    file.write(";\n")


                one_track = np.zeros(array_size)
                if(0 < loc < (self.image_amount)):
                    one_track[:loc] = 1
                    one_track[loc:] = 0
                elif(loc > (self.image_amount)):
                    one_track[:] = 1
                else:
                    one_track[:] = 0

                if (file):
                    file.write("final, ")
                    for s in range(self.image_amount):
                        file.write("%s, " % one_track[s])
                    file.write(";\n")

                tracks[tra_i].live_state = one_track * 2 #just covert 1 to 2

        if (file):
            file.close()

        live_area = np.zeros(self.image_amount)
        # tmp_live_dead_table = np.zeros(self.image_amount)
        # for tracks in (self.tracks, self.del_tracks):
        #     for i in range(0, len(tracks), 1):
        #         y = tracks[i].area
        #
        #         # y_p = y[~(np.isnan(y))]
        #         # x_p = np.nonzero(y > 0)[0]
        #         # x = np.arange(0, len(y), 1)
        #         # new_y = np.interp(x, x_p, y_p)
        #
        #         sub_y_idx = np.where(y > 0)[0]
        #         y_p = y[sub_y_idx]
        #         x = np.arange(sub_y_idx[0], sub_y_idx[-1], 1)
        #         part_new_y = np.interp(x, sub_y_idx, y_p)
        #         new_y = y.copy()
        #         new_y[sub_y_idx[0]:sub_y_idx[-1]] = part_new_y
        #
        #
        #         # print("track_id: ", tracks[i].track_id)
        #         # print(*(tracks[i].live_state))
        #         # print(*(tracks[i].area))
        #         # print(*new_y)
        #
        #         tracks[i].area = new_y
        #         for j in range(self.image_amount):
        #             if(tracks[i].live_state[j] > 1):
        #                 if(tracks[i].area[j] > 0):
        #                     live_area[j] += tracks[i].area[j]
        #                     # live_area[j] += new_y[j]
        #             else:
        #                 break


        for j in range(0, self.image_amount):
            for tracks in (self.tracks, self.del_tracks):
                for i in range(0, len(tracks), 1):
                    cell_x = tracks[i].coordinates[j][0]
                    cell_y = tracks[i].coordinates[j][1]

                    if (cell_x > 0 and cell_y > 0):

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

        # with open(outpath + "/Results_ucf/Results_pad_area_" + "{0:0=3d}".format(Beacon) + ".csv", 'w') as f:
        #     f.write("Beacon-" + "{0:0=3d}".format(Beacon) + ',')
        #     print(*live_area, sep=',', file = f)


        if(gt == True and os.path.exists(gt_video_path)):
        # if(gt == True and os.path.exists(gt_video_path)):
            vid = cv2.VideoCapture(gt_video_path)
            # skip = 3
            skip = frame_count - int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            print("skip2: ", skip)

            pad_wid = 200
            for i in range(0, self.image_amount):
                if (i >= skip):
                    ret, gt_frame = vid.read()

                    if not ret:
                        break

                    # if(os.path.exists(outpath + "coord.txt")):# False and
                    #     if(self.coord[0] == np.nan):
                    #         self.coord = np.loadtxt(outpath + "coord.txt")
                    #     coord = self.coord[i * 2: i * 2 + 2]
                    #     coord = coord.astype(int)
                    # else:
                    image_path = outpath + "images_ucf/Beacon_" + str(Beacon) + "/t" + "{0:0=3d}".format(i) + ".tif"
                    # image_path = outpath + "input_images/" + "t" + "{0:0=3d}".format(i) + ".tif"
                    if (not os.path.exists(image_path)):
                        # print("file not exist: ", image_path)
                        break
                    else:
                        # print(frame_count, image_path)
                        pass

                    frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    # d0 = frame.shape[0] >> 2
                    # d1 = frame.shape[1] >> 2
                    # template = gt_frame[d0:3 * d0, d1:3 * d1, 0]

                    d0 = frame.shape[0] >> 3
                    d1 = frame.shape[1] >> 3
                    template = gt_frame[d0:7 * d0, d1:7 * d1, 0]

                    ret = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF)
                    resu = cv2.minMaxLoc(ret)

                    coord = [resu[2][1] - d0, resu[2][0] - d1]
                    coord = -np.array(coord)
                    # print(coord)
                    self.coord[i * 2] = coord[0]
                    self.coord[i * 2 + 1] = coord[1]
                    #else end

                    gt_frame_pad = cv2.copyMakeBorder(gt_frame, pad_wid, pad_wid, pad_wid, pad_wid, cv2.BORDER_CONSTANT)
                    gt_frame = gt_frame_pad[
                               pad_wid + coord[0]:pad_wid + coord[0] + gt_frame.shape[0],
                               pad_wid + coord[1]:pad_wid + coord[1] + gt_frame.shape[1]]

                    # cv2.imshow("gt_frame", gt_frame)
                    # cv2.waitKey()

                    # frame_0 = frame[:, :, 0]
                    frame_1 = gt_frame[:, :, 1]
                    frame_2 = gt_frame[:, :, 2]

                    red = frame_2.astype(np.float) - frame_1.astype(np.float)
                    red_uint8 = np.clip(red, 0, 255).astype(np.uint8)

                    for tracks in (self.tracks, self.del_tracks):
                        for idx in range(len(tracks)):
                            x3 = tracks[idx].coordinates[i][0]
                            y3 = tracks[idx].coordinates[i][1]

                            if(np.isnan(x3) or np.isnan(y3)):
                                tracks[idx].g_truth[i] = np.nan
                            else:
                                if(red_uint8[int(y3)][int(x3)] > red_level):
                                    tracks[idx].g_truth[i] = 1
                                    tracks[idx].g_truth_t = i + 1
                                else:
                                    tracks[idx].g_truth[i] = -1
                    # print("Hello")

            # if ((not os.path.exists(outpath + "coord.txt")) and self.coord[0] != np.nan):
            # if (self.coord[0] != np.nan):
            np.savetxt(outpath + "info_ucf/Beacon_" + str(Beacon) + "_coord.txt", self.coord)

            f_g_truth = open(outpath + "info_ucf/Beacon_" + str(Beacon) + "_g_truth.txt", 'w')
            f_die_time = open(outpath + "info_ucf/Beacon_" + str(Beacon) + "_die_time.txt", 'w')
            # f_g_truth = open(outpath + "g_truth" + time.strftime("%d_%H_%M", time.localtime()) + ".txt", 'w')
            # f_die_time = open(outpath + "die_time" + time.strftime("%d_%H_%M", time.localtime()) + ".txt", 'w')

            for tracks in (self.tracks, self.del_tracks):
                for idx in range(len(tracks)):
                    # flag = 0
                    # for i in range(self.image_amount - 1, -1, -1):
                    #     if(tracks[idx].g_truth[i] == 1):
                    #         tracks[idx].g_truth[0:i] = 1
                    #         # print(tracks[idx].track_id, tracks[idx].g_truth)
                    #         flag = 1
                    #         break
                    #

                    # for ground truth SFP start

                    # if(tracks[idx].track_id == 278):
                    #     print("track id: ", tracks[idx].track_id, tracks[idx].g_truth)

                    # tracks[idx].g_truth = - tracks[idx].g_truth #for RFP ground truth
                    # for ground truth SFP end

                    # flag = 0
                    # for i in range(self.image_amount):
                    #     # tmp_arr = tracks[idx].g_truth[i:i + stable_period]
                    #
                    #     if(mean(tracks[idx].g_truth[i:i + stable_period]) == -1):
                    #         # print(idx, tracks[idx].track_id, i, tracks[idx].g_truth)
                    #         tracks[idx].g_truth[:i] = 1
                    #         tracks[idx].g_truth[i:] = -1
                    #         flag = 1
                    #         break


                    tra_len = np.count_nonzero(tracks[idx].coordinates > 0) / 2

                    # if flag:
                    #     if(i == 0):
                    #         f_g_truth.write(str(tracks[idx].track_id) + " " + str(-1) + " " + str(tra_len) + "\n")
                    #         # f_g_truth.write(str(tracks[idx].track_id) + " " + str(0) + " " + str(tra_len) + "\n")
                    #     else:
                    #         f_g_truth.write(str(tracks[idx].track_id) + " " + str(i) + " " + str(tra_len) + "\n")
                    #     # f_g_truth.write(str(tracks[idx].track_id) + " " + str(i) + "\n")
                    # else:
                    #     f_g_truth.write(str(tracks[idx].track_id) + " " + str(1000) + " " + str(tra_len) + "\n")
                    #     # f_g_truth.write(str(tracks[idx].track_id) + " " + str(self.image_amount) + " " + str(tra_len) + "\n")
                    #     tracks[idx].g_truth[:] = 1

                    if(tracks[idx].g_truth_t == -1):
                        tracks[idx].g_truth[:] = -1
                        f_g_truth.write(str(tracks[idx].track_id) + " " + str(-1) + " " + str(tra_len) + "\n")
                    elif(-1 < tracks[idx].g_truth_t < self.image_amount):
                        tracks[idx].g_truth[:tracks[idx].g_truth_t] = 1
                        tracks[idx].g_truth[tracks[idx].g_truth_t:] = -1
                        f_g_truth.write(str(tracks[idx].track_id) + " " + str(tracks[idx].g_truth_t) + " " + str(tra_len) + "\n")
                    else:
                        f_g_truth.write(str(tracks[idx].track_id) + " " + str(1000) + " " + str(tra_len) + "\n")
                        tracks[idx].g_truth[:] = 1




                    flag = 0
                    for i in range(self.image_amount - 1, -1, -1):
                        if(tracks[idx].live_state[i] > 1):
                            flag = 1
                            break

                    if flag:
                        if(i == self.image_amount - 1):
                            f_die_time.write(str(tracks[idx].track_id) + " " + str(1000) + "\n")
                            # f_die_time.write(str(tracks[idx].track_id) + " " + str(self.image_amount) + "\n")
                            # f_die_time.write(str(tracks[idx].track_id) + " " + str(self.image_amount) + "\n")
                        else:
                            f_die_time.write(str(tracks[idx].track_id) + " " + str(i + 1) + "\n")
                    else:
                        f_die_time.write(str(tracks[idx].track_id) + " " + str(-1) + "\n")
                        # f_die_time.write(str(tracks[idx].track_id) + " " + str(0) + "\n")


            f_g_truth.close()
            f_die_time.close()

        f_cell_tracks = open(outpath + "info_ucf/Beacon_" + str(Beacon) + "_cell_tracks.txt", 'w')
        for tracks in (self.tracks, self.del_tracks):
            for idx in range(len(tracks)):
                print("track_id:", tracks[idx].track_id, sep='', file=f_cell_tracks)
                for i in range(self.image_amount):
                    if(tracks[idx].full_trace[i] != None):
                        # print(*(tracks[idx].full_trace[i][[3, 10, 11]]), sep=',', file=f_cell_tracks)
                        print("%.2f,%.2f,%.2f,%.2f,%.2f"%tuple(tracks[idx].full_trace[i][[0, 1, 3, 10, 11]]), sep=',', file=f_cell_tracks)
                    else:
                        print(np.nan, np.nan, np.nan, np.nan, np.nan, sep=',', file=f_cell_tracks)
        f_cell_tracks.close()

        # for tracks in (self.tracks, self.del_tracks):
        #     for tra_i in range(0, len(tracks), 1):
        #         average = [mean(tracks[tra_i].shift[:, 0]), mean(tracks[tra_i].shift[:, 1])]
        #         print("average", average)
        #         print(tracks[tra_i].shift)
        #         tracks[tra_i].shift = tracks[tra_i].shift - average
        #         print(tracks[tra_i].coordinates)
        #         print(tracks[tra_i].shift)
        #         tracks[tra_i].shift = tracks[tra_i].shift / scale
        #         tracks[tra_i].coordinates[:,0] += tracks[tra_i].shift[:,1]
        #         tracks[tra_i].coordinates[:,1] += tracks[tra_i].shift[:,0]
        #         tracks[tra_i].coordinates += 0.5
        #         print(tracks[tra_i].coordinates)


    def mark(self, frame, frame_index, scale):

        live_count = 0
        dead_count = 0
        zero_cell = 0

        # for i in range(len(self.alive_mat)):
        #     x3 = self.coordinate_matrix[i][frame_index * 2]
        #     y3 = self.coordinate_matrix[i][frame_index * 2 + 1]
        #     live_status = self.alive_mat[i][frame_index]
        #     track_id = self.coordinate_matrix[i][-1]

        for tracks in (self.del_tracks, self.tracks):
            for i in range(len(tracks)):
                x3 = tracks[i].coordinates[frame_index * 2]
                y3 = tracks[i].coordinates[frame_index * 2 + 1]
                live_status = tracks[i].live_state[frame_index]
                track_id = tracks[i].track_id

                if (x3 > 0 or y3 > 0):
                    if (live_status >= 1):
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), 5 * scale, (255, 255, 0),
                                   int(0.5 * scale))
                        cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), int(0.3 * scale))
                        live_count += 1
                    else:
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), 5 * scale, (0, 255, 255),
                                   int(0.5 * scale))
                        cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))
                        dead_count += 1
                        pass

        cv2.putText(frame, str(frame_index), (5 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale,
                    (0, 255, 255), int(0.3 * scale))
        # cv2.putText(frame, str(frame_index), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (138, 221, 48), 1)
        cv2.putText(frame, "live: " + str(live_count) + ", ", (60 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3 * scale, (255, 255, 0), int(0.3 * scale))
        cv2.putText(frame, "dead: " + str(dead_count), (175 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale,
                    (0, 255, 255), int(0.3 * scale))

        return frame

    def mark_gt(self, frame, frame_index, scale, gt_frame, crop_height, crop_width, out_path, Beacon, add_imageJ, get_cells, f_det_txt):
        # get_cells = True
        cells_path = None
        debug = 1
        frame_red = None

        label = False
        # label = True
        frame_label = np.zeros(((crop_height, crop_width)), np.uint16)
        frame_label_2 = np.zeros_like(frame_label)
        label_path = out_path + "label/Beacon-" + str(Beacon) + "/"

        # if(frame_index < 5):
        #     get_cells = True
        # else:
        #     get_cells = False

        if (get_cells):
            cells_path = out_path + "ML/Beacon_" + str(Beacon) + "/cells/"
            os.makedirs(cells_path, exist_ok=True)
            ret, frame_cell = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/",
                                         frame_count=frame_index, data_type=1, scale=1,
                                         crop_width=crop_width, crop_height=crop_height)

            cv2.imwrite(out_path + "ML/Beacon_" + str(Beacon) + "/img1/{0:0=6d}".format(frame_index) + ".jpg", frame_cell)

        if frame_index == 0:
            if (label):
                if (not os.path.exists(label_path)):
                    os.makedirs(label_path)
                if (not os.path.exists(label_path + "version_0/")):
                    os.makedirs(label_path + "version_0/")

            if(debug == 1):
                self.f_state = open(out_path + "info_ucf/Beacon_" + str(Beacon) + "_f_state_" + str(stable_period) + "_" + str(red_level) + ".txt", "w")
                # self.f_state = open(out_path + "f_state_" + str(stable_period) + "_" + str(red_level) + time.strftime("%d_%H_%M", time.localtime()) + ".txt", "w")

        if(add_imageJ):
            ret, red, frame_red = self.process_gt_frame_2(frame_index, gt_frame, crop_height, crop_width, scale)

            if ret:
                new_red = frame[:, :, 2].astype(float) + red.astype(float)
                new_red = np.clip(new_red, 0, 255).astype(np.uint8)
                # frame[:, :, 1] = new_red
                frame[:, :, 2] = new_red

        live_count = 0
        dead_count = 0
        zero_cell = 0

        gt_count = 0
        gt_d_count = 0

        frame_org = frame[:, :, 0].copy()

        for tracks in (self.del_tracks, self.tracks):
            for i in range(len(tracks)):
                if (get_cells):
                    single_cell_path = cells_path + "{0:0=4d}".format(tracks[i].track_id) + "/"
                    os.makedirs(single_cell_path, exist_ok=True)

                if(tracks[i].good == False):
                    continue

                x3 = tracks[i].coordinates[frame_index][0]
                y3 = tracks[i].coordinates[frame_index][1]

                if (not(np.isnan(x3) or np.isnan(y3))): # and np.count_nonzero(tracks[i].coordinates > 0) == 2
                    cell = tracks[i].full_trace[frame_index]
                    radius = cell[8]
                    draw_r = int(max(radius, 5 * scale))
                    draw_r = int(min(draw_r, 10 * scale))

                    live_status = tracks[i].live_state[frame_index]
                    track_id = tracks[i].track_id
                    if (live_status > 1):
                    # if (tracks[i].g_truth[frame_index] >= 1):

                        # pt1 = (int((x3 - 5) * scale), int((y3 - 5) * scale))
                        # pt2 = (int((x3 + 5) * scale), int((y3 + 5) * scale))
                        # cv2.rectangle(frame, pt1, pt2, (255, 255, 0), 1)

                        # if(frame_index == 286 and track_id in f_286 and (track_id not in f_287)):
                        #     cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), 5 * scale, (255, 255, 0), int(0.5 * scale))
                        #     cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), int(0.3 * scale))
                        #
                        # if(frame_index == 287 and track_id in f_287 and (track_id not in f_288)):
                        #     cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), 5 * scale, (255, 255, 0), int(0.5 * scale))
                        #     cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), int(0.3 * scale))

                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (255, 255, 0), int(0.5 * scale))
                        cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), int(0.3 * scale))

                        live_count += 1

                        if(label):
                            contour = tracks[i].full_trace[frame_index][6]
                            contour = contour / scale
                            contour = contour.astype(np.int32)
                            cv2.drawContours(frame_label, [contour], -1, (1, 1, 1), -1)
                            cv2.drawContours(frame_label_2, [contour], -1, (track_id, track_id, track_id), -1)


                        # if(frame_index == 286 or frame_index == 287 or frame_index == 288):
                        #     print("qibing", track_id)

                        if (tracks[i].g_truth[frame_index] == 1):
                            gt_count += 1
                        elif tracks[i].g_truth[frame_index] == -1:# mark the error
                            cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (255, 255, 0), int(0.3 * scale))
                            pass
                        else:
                            # print("wrong tracks[i].g_truth[frame_index]", i, frame_index)
                            pass

                    # elif (live_status < 0):
                    else:
                        cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), draw_r, (0, 255, 255), int(0.5 * scale))
                        cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))

                        if(label):
                            contour = tracks[i].full_trace[frame_index][6]
                            contour = contour / scale
                            contour = contour.astype(np.int32)
                            cv2.drawContours(frame_label, [contour], -1, (2, 2, 2), -1)
                            cv2.drawContours(frame_label_2, [contour], -1, (track_id, track_id, track_id), -1)


                        # pt1 = (int((x3 - 5) * scale), int((y3 - 5) * scale))
                        # pt2 = (int((x3 + 5) * scale), int((y3 + 5) * scale))
                        # cv2.rectangle(frame, pt1, pt2, (0, 255, 255), 1)

                        dead_count += 1

                        if (tracks[i].g_truth[frame_index] == -1):
                            gt_d_count += 1
                        elif tracks[i].g_truth[frame_index] == 1:# mark the error
                            cv2.putText(frame, str(track_id), (int((x3 + 5) * scale), int((y3 + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.2 * scale, (0, 255, 255), int(0.3 * scale))
                            pass
                        else:
                            # print("wrong tracks[i].g_truth[frame_index]", i, frame_index)
                            pass

                    # else:
                    #     pass

                    # if(get_cells == True):
                    #     cell_radius = 10
                    #     if((y3 - cell_radius) > 0 and (y3 + cell_radius) < (frame.shape[0] / scale) and x3 - cell_radius > 0 and (x3 + cell_radius) < (frame.shape[1] / scale)):
                    #         one_cell = frame_org[int((y3 - cell_radius) * scale):int((y3 + cell_radius) * scale), int((x3 - cell_radius) * scale):int((x3 + cell_radius) * scale)]
                    #         cell_img_path = cells_path + str(tracks[i].track_id) + "/" + str(frame_index) + ".tif"
                    #         # print(x3, y3, frame_index, one_cell.shape, frame.shape, cell_img_path)
                    #         cv2.imwrite(cell_img_path, one_cell)

                    if(get_cells == True):
                        cell_radius = 9
                        # cell_radius = 5
                        scale_cell = 1
                        # ret, frame_cell = read_frame(out_path + "images_ucf/Beacon_" + str(Beacon) + "/", frame_count = frame_index, data_type = 1, scale = scale_cell, crop_width = crop_width, crop_height = crop_height)
                        if((y3 - cell_radius) > 0 and (y3 + cell_radius) < (frame_cell.shape[0] / scale_cell) and x3 - cell_radius > 0 and (x3 + cell_radius) < (frame_cell.shape[1] / scale_cell)):
                            one_cell = frame_cell[int((y3 - cell_radius) * scale_cell):int((y3 + cell_radius) * scale_cell), int((x3 - cell_radius) * scale_cell):int((x3 + cell_radius) * scale_cell)]
                            cell_img_path = cells_path + "{0:0=4d}/".format(tracks[i].track_id) + "{0:0=4d}".format(tracks[i].track_id) + "_" + str(frame_index) + ".tif"
                            cv2.imwrite(cell_img_path, one_cell)
                            print(frame_index, -1, x3 - cell_radius, y3 - cell_radius, cell_radius * 2, cell_radius * 2, 1, -1, -1, -1, file=f_det_txt, sep=',')



        if (label):
            # frame_label = cv2.resize(frame_label, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(label_path + "label_" + "{0:0=3d}".format(frame_index) + ".tif", frame_label)

            # frame_label_2 = cv2.resize(frame_label_2, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(label_path + "version_0/label_" + "{0:0=3d}".format(frame_index) + ".tif", frame_label_2)


        cv2.putText(frame, str(frame_index), (5 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (0, 255, 255), int(0.5 * scale))
        # cv2.putText(frame, str(frame_index), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (138, 221, 48), 1)
        if(add_imageJ):
            cv2.putText(frame, "live: " + str(live_count) + "(" + str(gt_count) + "," + str(live_count - gt_count) + ")", (30 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * scale, (255, 255, 0), int(0.5 * scale))
            cv2.putText(frame, "dead: " + str(dead_count) + "(" + str(gt_d_count) + "," + str(dead_count - gt_d_count) + ")", (150 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale,
                        (0, 255, 255), int(0.5 * scale))
        else:
            cv2.putText(frame, "live: " + str(live_count),
                        (30 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * scale, (255, 255, 0), int(0.5 * scale))
            cv2.putText(frame, "dead: " + str(dead_count),
                        (150 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale,
                        (0, 255, 255), int(0.5 * scale))


        if(debug == 1):
            print(live_count, gt_count, dead_count, gt_d_count, file = self.f_state)

        if (add_imageJ and len(gt_frame) > 0 and live_count > 0):
            # accu = float(gt_count) / live_count
            diff = float(live_count - gt_count + dead_count - gt_d_count)
            total = float(live_count + dead_count)
            cv2.putText(frame, "accu: " + str(float("{0:.2f}".format(((total - diff) / total)))), (300 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * scale, (0, 0, 255), int(0.5 * scale))

        return frame, frame_red

    def process_gt_frame(self, frame, gt_frame, crop_height, crop_width, scale):

        pad_wid = 200
        if (len(gt_frame) == 0):
            return False, None

        gt_frame = gt_frame[0:crop_height, 0:crop_width]
        # gt_frame = cv2.resize(gt_frame, (gt_frame.shape[1] * scale, gt_frame.shape[0] * scale),
        #                       interpolation=cv2.INTER_CUBIC)

        # cv2.imshow("frame", frame)
        # cv2.imshow("gt_frame", gt_frame)

        d0 = frame.shape[0] >> 2
        d1 = frame.shape[1] >> 2
        template = gt_frame[d0:3 * d0, d1:3 * d1, 0]
        ret = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF)
        resu = cv2.minMaxLoc(ret)

        coord = [resu[2][1] - d0, resu[2][0] - d1]
        coord = -np.array(coord)
        # print(coord)

        gt_frame_pad = cv2.copyMakeBorder(gt_frame, pad_wid, pad_wid, pad_wid, pad_wid, cv2.BORDER_CONSTANT)
        gt_frame = gt_frame_pad[
                   pad_wid + coord[0]:pad_wid + coord[0] + gt_frame.shape[0],
                   pad_wid + coord[1]:pad_wid + coord[1] + gt_frame.shape[1]]

        # frame_0 = frame[:, :, 0]
        frame_1 = gt_frame[:, :, 1]
        frame_2 = gt_frame[:, :, 2]

        red = frame_2.astype(np.float) - frame_1.astype(np.float)
        red_uint8 = np.clip(red, 0, 255).astype(np.uint8)

        red_uint8 = cv2.resize(red_uint8, (red_uint8.shape[1] * scale, red_uint8.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

        # ret, th4 = cv2.threshold(red_uint8, 10, 255, cv2.THRESH_BINARY)

        # contours, hierarchy = cv2.findContours(th4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)

        # return contours, th4
        return True, red_uint8

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

    def mark_mitosis(self, frame, frame_index, scale, image_path, frame_count):
        print("mark_mitosis: num of tracks: ", len(self.tracks), len(self.del_tracks))

        for tracks in (self.del_tracks, self.tracks):
            for i in range(len(tracks)):
                cell = tracks[i].full_trace[frame_index]
                if (cell != None):
                    x3 = cell[0]
                    y3 = cell[1]
                    box = cell[7]
                    box = cell[6]  # contour
                    if (x3 > 0 and y3 > 0):
                        # if(tracks[i].parent != None or len(tracks[i].children) > 0):
                        if(tracks[i].mitosis_time[frame_index] == 1):
                            # print("mark_mitosis", i)
                            # cv2.drawContours(frame, [box], 0, (255, 255, 0), 2)
                            # cv2.putText(frame, str(self.tracks[i].track_id), (int((x3 + 15) * scale), int((y3 + 15) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            #             (255, 255, 0), 2)
                            cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), 5 * scale, (255, 255, 0), ((1 * scale) >> 2))
                            cv2.putText(frame, str(tracks[i].track_id), (int((x3 + 7) * scale), int((y3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 255, 0), 2)
                        else:
                            # cv2.circle(frame, (int(x3 * scale), int(y3 * scale)), 5 * scale, (0, 255, 255), ((1 * scale) >> 2))
                            # cv2.putText(frame, str(tracks[i].track_id), (int((x3 + 7) * scale), int((y3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            #             (0, 255, 255), 2)
                            pass

        cv2.putText(frame, str(frame_index), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (138, 221, 48), 2)

        return frame

    def analyse_mitosis(self, image_path, frame_amount):

        self.image_amount = frame_amount
        f_data = open(image_path + "data.txt", "w")
        f_data.write("Timepoint,Nucleus No,Split Nucleus No 1,Split Nucleus No 2,Split,No Change\n")

        for i in range(len(self.tracks)):
            if(self.tracks[i].parent == None):
                for j in range(self.image_amount):
                    # row = [None] * 5
                    # row.append(j + 1)
                    row = []
                    row.append(j + 1)
                    if (len(self.tracks[i].children) == 0):
                        row.append(self.tracks[i].track_id)
                        row.append("")
                        row.append("")
                        row.append("No")
                        row.append("Yes")
                    else:

                        c_id = self.tracks[i].children[0]
                        mitosis_time = self.tracks[c_id].trace[0][9]

                        if (j >= mitosis_time):
                            row.append("")
                            row.append(self.tracks[i].track_id)
                            row.append(self.tracks[i].children[0])

                            if (j == mitosis_time):
                                row.append("Yes")
                                row.append("No")
                            else:
                                row.append("No")
                                row.append("Yes")
                        else:
                            row.append(self.tracks[i].track_id)
                            row.append("")
                            row.append("")
                            row.append("No")
                            row.append("Yes")

                    # print(row, file=f_data)
                    print("%s,%s,%s,%s,%s,%s" % (row[0], row[1], row[2], row[3], row[4], row[5]), file=f_data)

        f_data.close()
