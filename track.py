# vim: expandtab:ts=4:sw=4
import cv2
import numpy as np
array_size = 800
class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None, tlwh=None, frame_idx=0, frame_org = None, area = 0, img_amount = 0, cell_d = None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

        self.tlwh_s = np.zeros((289, 4), dtype=float)
        self.tlwh_s[:] = np.nan
        self.tlwh_s[frame_idx,:] = tlwh

        self.last_tlwh = tlwh

        self.start_t = frame_idx
        self.end_t = 289

        self.area = np.zeros(289, dtype=float)
        self.area[frame_idx] = area

        self.x_variance = 0
        self.y_variance = 0
        self.max_x = 0
        self.max_y = 0
        self.valid_t = 0
        self.x_std = 0
        self.y_std = 0

        self.peak_num = 0

        self.cell_diff = np.zeros(array_size, dtype=float)
        self.cell_diff[:] = np.nan

        self.area = np.zeros(array_size, dtype=float)
        self.area[:] = np.nan

        self.live_state = np.zeros(array_size, dtype=float)
        self.live_state[:] = np.nan

        self.g_truth = np.zeros(array_size, dtype=float)
        self.g_truth[:] = np.nan
        self.g_truth_t = -1


        # self.horizontal_x = x3 = self.tlwh_s[frame_idx][1] + self.tlwh_s[frame_idx][3] / 2
        # self.vertical_y = y3 = self.tlwh_s[frame_idx][0] + self.tlwh_s[frame_idx][2] / 2

        x3 = self.tlwh_s[frame_idx][1] + self.tlwh_s[frame_idx][3] / 2
        y3 = self.tlwh_s[frame_idx][0] + self.tlwh_s[frame_idx][2] / 2


        scale = 8
        # d = 5
        cell_r = 9
        if cell_r < x3 < (frame_org.shape[1] / scale - cell_r) and cell_r < y3 < (frame_org.shape[0] / scale - cell_r):
            self.last_cell = frame_org[int((y3 - 5) * scale):int((y3 + 5) * scale), int((x3 - 5) * scale):int((x3 + 5) * scale)].copy()
        else:
            print(mean, covariance, track_id, n_init, max_age, feature, tlwh, frame_idx, frame_org, area)
            pass
        self.type = "unknown"

        self.amount = img_amount

        self.coord = np.zeros((289, 2), dtype=float)
        self.coord[:] = 0
        self.coord[frame_idx][0] = x3
        self.coord[frame_idx][1] = y3

        self.score = np.zeros(289, dtype=float)
        self.score[:] = np.nan
        self.score[frame_idx] = cell_d.score
        self.good = True#if the track has few cells, or the cells are all scored low, it becomes bad
        self.full_trace = [None] * array_size
        self.full_trace[frame_idx] = cell_d
        self.motion_std = 0
        self.fluo = False
        pass

    # def to_tlwh(self):
    #     """Get current position in bounding box format `(top left x, top left y,
    #     width, height)`.
    #
    #     Returns
    #     -------
    #     ndarray
    #         The bounding box.
    #
    #     """
    #     ret = self.mean[:4].copy()
    #     ret[2] *= ret[3]
    #     ret[:2] -= ret[2:] / 2
    #     return ret


    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection, frame_idx, frame_org):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        self.tlwh_s[frame_idx, :] = detection.tlwh
        self.last_tlwh = detection.tlwh
        self.area[frame_idx] = detection.area

        old_cell = self.last_cell
        d = 5
        scale = 8
        x3 = detection.horizontal_x
        y3 = detection.vertical_y
        one_cell = frame_org[int((y3 - 5) * scale):int((y3 + 5) * scale), int((x3 - 5) * scale):int((x3 + 5) * scale)].copy()

        if 2 * d < x3 < (frame_org.shape[1] / scale - 2 * d) and 2 * d < y3 < (frame_org.shape[0] / scale - 2 * d):
            image_a = old_cell
            image_b = frame_org[int((y3 - 2 * d) * scale):int((y3 + 2 * d) * scale), int((x3 - 2 * d) * scale):int((x3 + 2 * d) * scale)].copy()

            template = image_a
            ret = cv2.matchTemplate(image_b, template, cv2.TM_SQDIFF)
            resu = cv2.minMaxLoc(ret)
            # shift = [resu[2][1] - d * scale, resu[2][0] - d * scale]  # resu[2][0] is x, resu[2][1] is y.
            diff = resu[0]
            # count_0 += 1
        else:
            diff = ((one_cell.astype(float) - old_cell.astype(float)) ** 2).sum()
            # shift = [0, 0]
            # count_1 += 1

        self.cell_diff[frame_idx] = diff
        self.last_cell = one_cell
        self.coord[frame_idx][0] = x3
        self.coord[frame_idx][1] = y3
        self.score[frame_idx] = detection.score
        self.full_trace[frame_idx] = detection


    def mark_missed(self, frame_idx):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
            self.end_t = frame_idx - self.time_since_update + 1

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        # return self.state == TrackState.Confirmed
        return True

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
