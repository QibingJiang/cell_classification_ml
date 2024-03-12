# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

# from application_util import preprocessing
# from application_util import visualization
# from deep_sort import nn_matching
# from deep_sort.detection import Detection
# from deep_sort.tracker import Tracker

# from deep_sort_2 import preprocessing
# from deep_sort_2 import visualization
# from deep_sort_2 import nn_matching
# from deep_sort_2.detection import Detection
# from deep_sort_2.tracker import Tracker

import preprocessing
import visualization
import nn_matching
from detection import Detection
from tracker import Tracker

import statistics
import re

array_size = 800

def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_dir": sequence_dir,
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
det_out = None
def create_detections(detection_mat, frame_idx, min_height=0, img_path=None, sequence_dir=None):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    global det_out
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    # calculate area start
    frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(frame, (frame.shape[1] * 8, frame.shape[0] * 8), interpolation=cv2.INTER_CUBIC)
    gray_cnt = np.zeros_like(gray,dtype=float)
    ret, th4 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(th4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        cv2.drawContours(gray_cnt, [contours[i]], -1, (area, area, area), -1)
        pass
    #     calculate area end


    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue

        x3 = bbox[0] + bbox[2] / 2
        y3 = bbox[1] + bbox[3] / 2
        scale = 8
        area = gray_cnt[int(y3*scale)][int(x3*scale)]

        detection_list.append(Detection(bbox, confidence, feature, area))

        cv2.circle(frame, (int(x3), int(y3)), 6, (255, 255, 0), 1)
        # scale = 3
        # cv2.putText(frame, str(frame_idx) + str(x3) + str(y3), (int(x3), int(y3)), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (255, 255, 255), int(0.3 * scale))


    if det_out is None:
        det_out = cv2.VideoWriter(os.path.join(sequence_dir, "cell_detect.mp4"), fourcc, 3.0, (frame.shape[1], frame.shape[0]), isColor=False)

    scale = 1
    cv2.putText(frame, str(frame_idx), (5 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (255, 255, 255), int(0.3 * scale))

    det_out.write(frame)

    # if(frame_idx == 67):
    # cv2.imshow("det", frame)
    # cv2.waitKey()

    return detection_list



def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, out_path = "/home/qibing/disk_16t/qibing/output_ml/"):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    tracker.seq_info = seq_info.copy()

    ret = re.sub(r'.*Beacon', '', sequence_dir)
    Beacon = re.sub(r'/.*', '', ret[1:])

    if(Beacon == ''):
        Beacon = 0
    else:
        Beacon = int(Beacon)


    for new_folder in ["images_ucf/Beacon_" + str(Beacon) + "/", "Misc/Results_ucf/", "Results/", "Misc/info_ucf/"]:
        os.makedirs(out_path + new_folder, exist_ok = True)


    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #
    # if(det_video == None):
    #     det_video = cv2.VideoWriter(sequence_dir + "classified.mp4", fourcc, 2.0, (frame.shape[1], frame.shape[0]), isColor=True)



    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height, seq_info["image_filenames"][frame_idx], seq_info["sequence_dir"])
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections, frame_idx)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            # vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks, frame_idx)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)

    f_id = open("track_id.txt", "w")
    f = open("track_motion.txt", "w")
    # calculate x and y variance
    for tracks in [tracker.tracks, tracker.del_tracks]: #tracker.del_tracks,
        for j in range(len(tracks)):
            x_s = tracks[j].tlwh_s[tracks[j].start_t:tracks[j].end_t, 0]
            y_s = tracks[j].tlwh_s[tracks[j].start_t:tracks[j].end_t, 1]

            x_s_diff = np.diff(tracks[j].tlwh_s[:, 0])
            y_s_diff = np.diff(tracks[j].tlwh_s[:, 1])
            motion_vector = np.sqrt(x_s_diff*x_s_diff + y_s_diff*y_s_diff)
            f_id.write(str(tracks[j].track_id) + " ")
            print(*motion_vector, file=f)

            tracks[j].peak_num = np.count_nonzero(motion_vector > 10)
            # motion_vector_v = motion_vector[np.where(motion_vector > 0)]

            x_s = x_s[np.where(x_s > 0)]
            y_s = y_s[np.where(y_s > 0)]
            tracks[j].valid_t = len(x_s)

            if(tracks[j].valid_t > 5):
                # tracks[j].x_variance = statistics.variance(x_s)
                # tracks[j].y_variance = statistics.variance(y_s)
                # print(tracks[j].x_variance, tracks[j].y_variance)
                tracks[j].max_x = np.max(np.abs(np.diff(x_s)))
                tracks[j].max_y = np.max(np.abs(np.diff(y_s)))
                tracks[j].x_std = np.std(x_s)
                tracks[j].y_std = np.std(y_s)

                if (tracks[j].x_std < 20 and tracks[j].y_std < 20):
                    tracks[j].type = "myeloma"
    f_id.close()
    f.close()

    tracker.analyse_classification_8(out_path, tracker.seq_info["max_frame_idx"] + 1, None, 8, Beacon, gt = False)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    scale = 3
    out_video = None
    for i in range(seq_info["max_frame_idx"] + 1):
        frame = cv2.imread(sequence_dir + "img1/" + "{0:0=6d}".format(i) + ".jpg")
        frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

        for tracks in [tracker.del_tracks, tracker.tracks]:
            for j in range(len(tracks)):
                x = tracks[j].tlwh_s[i][0] + tracks[j].tlwh_s[i][2] / 2
                y = tracks[j].tlwh_s[i][1] + tracks[j].tlwh_s[i][3] / 2

                if(x > 0 and y > 0):
                    if(tracks[j].valid_t > 5):
                        if(tracks[j].x_std < 20 and tracks[j].y_std < 20):
                            cv2.circle(frame, (int(x * scale), int(y * scale)), 5 * scale, (0, 255, 0), int(0.5 * scale))
                            if(tracks[j].live_state[i] > 1):
                                cv2.putText(frame, str(tracks[j].track_id), (int((x + 5) * scale), int((y + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (255, 255, 0), int(0.5 * scale))
                            else:
                                cv2.putText(frame, str(tracks[j].track_id), (int((x + 5) * scale), int((y + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (0, 255, 255), int(0.5 * scale))

                            # tracks[j].type = "myeloma"
                        else:
                            # if(tracks[j].peak_num < -1):
                            #     cv2.circle(frame, (int(x * scale), int(y * scale)), 5 * scale, (255, 255, 255), int(0.5 * scale))
                            #     cv2.putText(frame, str(tracks[j].track_id), (int((x + 5) * scale), int((y + 3) * scale)),
                            #                 cv2.FONT_HERSHEY_SIMPLEX,
                            #                 0.3 * scale, (255, 255, 255), int(0.5 * scale))
                            # else:
                            #     cv2.circle(frame, (int(x * scale), int(y * scale)), 5 * scale, (255, 0, 255), int(0.5 * scale))
                            #     cv2.putText(frame, str(tracks[j].track_id), (int((x + 5) * scale), int((y + 3) * scale)),
                            #                 cv2.FONT_HERSHEY_SIMPLEX,
                            #                 0.3 * scale, (255, 0, 255), int(0.5 * scale))
                            cv2.circle(frame, (int(x * scale), int(y * scale)), 5 * scale, (255, 0, 255), int(0.5 * scale))

                            if(tracks[j].live_state[i] > 1):
                                cv2.putText(frame, str(tracks[j].track_id), (int((x + 5) * scale), int((y + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (255, 255, 0), int(0.5 * scale))
                            else:
                                cv2.putText(frame, str(tracks[j].track_id), (int((x + 5) * scale), int((y + 3) * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (0, 255, 255), int(0.5 * scale))
                    else:
                        cv2.putText(frame, str(tracks[j].track_id), (int((x + 5) * scale), int((y + 3) * scale)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (255, 255, 255), int(0.5 * scale))
                        cv2.putText(frame, str(tracks[j].track_id), (int((x + 5) * scale), int((y + 3) * scale)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (255, 255, 255), int(0.5 * scale))

        cv2.putText(frame, str(i), (5 * scale, 10 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (0, 255, 255), int(0.5 * scale))

        if(out_video == None):
            out_video = cv2.VideoWriter(sequence_dir + "classified.mp4", fourcc, 2.0, (frame.shape[1], frame.shape[0]), isColor=True)

        out_video.write(frame)
    out_video.release()


    # tracker.analyse_classification_8(out_path, tracker.seq_info["max_frame_idx"] + 1, None, 8, Beacon, gt = False)
    # tracker.analyse_classification_5(out_path, tracker.seq_info["max_frame_idx"] + 1, None, 8, Beacon, gt = False)
    return tracker

def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()

def my_run(sequence_dir, detection_file, out_path):
    run(sequence_dir, detection_file, "/tmp/hypotheses.txt", 0.3, 1.0, 0, 0.2, 100, False, out_path)

if __name__ == "__main__":
    args = parse_args()
    # print(args.sequence_dir, args.detection_file, args.output_file,
    #     args.min_confidence, args.nms_max_overlap, args.min_detection_height,
    #     args.max_cosine_distance, args.nn_budget, args.display)
    # exit()

    # run(
    #     args.sequence_dir, args.detection_file, args.output_file,
    #     args.min_confidence, args.nms_max_overlap, args.min_detection_height,
    #     args.max_cosine_distance, args.nn_budget, args.display)

    my_run("/home/qibing/disk_16t/qibing/macrophage/Macrophage_VENDAR_Pt641/TimeLapseVideos/ML/Beacon_147/",
           "/home/qibing/disk_16t/qibing/macrophage/Macrophage_VENDAR_Pt641/TimeLapseVideos/ML/Beacon_147/Beacon_147.npy",
           "/home/qibing/disk_16t/qibing/macrophage/Macrophage_VENDAR_Pt641/TimeLapseVideos/ML/Beacon_147/optimize_0/")

# --sequence_dir=/home/qibing/disk_16t/qibing/macrophage/Macrophage_VENDAR_Pt641/TimeLapseVideos/ML/Beacon_147/     --detection_file=/home/qibing/disk_16t/qibing/macrophage/Macrophage_VENDAR_Pt641/TimeLapseVideos/ML/Beacon_147/Beacon_147.npy     --min_confidence=0.3     --nn_budget=100     --display=True

# --sequence_dir=/home/qibing/disk_16t/qibing/macrophage/Macrophage_VENDAR_Pt641/TimeLapseVideos/ML/Beacon_147/     --detection_file=/home/qibing/disk_16t/qibing/macrophage/Macrophage_VENDAR_Pt641/TimeLapseVideos/ML/Beacon_147/Beacon_147.npy     --min_confidence=0.3     --nn_budget=100     --display=True
# /home/qibing/disk_16t/qibing/macrophage/Macrophage_VENDAR_Pt641/TimeLapseVideos/ML/Beacon_147/
# /home/qibing/disk_16t/qibing/macrophage/Macrophage_VENDAR_Pt641/TimeLapseVideos/ML/Beacon_147/Beacon_147.npy
# /tmp/hypotheses.txt 0.3 1.0 0 0.2 100 True

# --sequence_dir=/home/qibing/disk_16t/qibing/macrophage/Macrophage_VENDAR_Pt641/TimeLapseVideos/ML/Beacon_147/     --detection_file=/home/qibing/disk_16t/qibing/macrophage/Macrophage_VENDAR_Pt641/TimeLapseVideos/ML/Beacon_147/Beacon_147.npy     --min_confidence=0.3     --nn_budget=100     --display=True