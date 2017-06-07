import numpy as np
import util
from track import Track
from tracker_base import TrackerBase
import tracking_constants as const
from scipy.optimize import linear_sum_assignment


class MultipleObjectTracker(TrackerBase):
    def __init__(self):
        TrackerBase.__init__(self)
        self.tracks = []

    def __save_tracks_to_json(self):
        for track in self.tracks:
            if track.is_dead():
                np.save('tracks/' + str(track.uid), track.history)

    def __delete_duplicate_tracks(self):

        # check if tracks 'heads' are identical
        for i in xrange(len(self.tracks)):
            track1 = self.tracks[i]
            for j in xrange(len(self.tracks)):
                if j == i:
                    continue
                track2 = self.tracks[j]
                if util.check_tracks_equal(track1, track2):
                    # print 'duplicate found!'
                    # if so, delete shortest track
                    if track1.get_length() > track2.get_length():
                        track2.delete_me = True
                    else:
                        track1.delete_me = True

        self.tracks = [t for t in self.tracks if t.delete_me is False]

    def __assign_detections_to_tracks_munkres(self, detections, frame_id, save=False):

        # if there are no tracks yet, all detections are new tracks
        if len(self.tracks) == 0:
            for det in detections:
                t = Track()
                t.add_to_track(det)
                self.tracks.append(t)
            return True
        # find distance from all tracks to all detections and formulate dists matrix
        dists = np.zeros(shape=(len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            predicted_next_bb = track.get_predicted_next_bb()
            for j, det in enumerate(detections):
                dist = util.dist_btwn_bb_centroids(predicted_next_bb, det.bbox)
                if track.is_singular():
                    max_dist = const.MAX_PIXELS_DIST_TRACK_START
                else:
                    max_dist = const.MAX_PIXELS_DIST_TRACK
                if dist > max_dist:
                    dist = 1e6  # set to arbitrarily high number
                dists[i, j] = dist
        # set all tracks as unassigned
        for t in self.tracks:
            t.has_match = False
        # assign all detections to tracks with munkres algorithm
        assigned_rows, assigned_cols = linear_sum_assignment(dists)
        for idx, row in enumerate(assigned_rows):
            col = assigned_cols[idx]
            # if track is assigned a detection with dist=1e6, discard that assignment
            if dists[row, col] != 1e6:
                self.tracks[row].has_match = True
                detections[col].has_match = True
                self.tracks[row].add_to_track(detections[col])
                self.tracks[row].num_misses = 0

        # create new tracks from unassigned detections:
        for det in detections:
            if det.has_match is False:
                t = Track()
                t.add_to_track(det)
                self.tracks.append(t)

        # keep track of how many times a track has gone unassigned
        for t in self.tracks:
            if t.has_match is False:
                t.num_misses += 1
                # t.propagate_track(frame_id=frame_id)

        # cleanup any duplicate tracks that have formed (TODO: how do they form?)
        self.__delete_duplicate_tracks()
        # save dead tracks before deletion
        if save:
            self.__save_tracks_to_json()
        # remove dead tracks
        self.tracks = [t for t in self.tracks if (t.is_dead() is False and t.delete_me is False)]

    def __assign_detections_to_tracks(self, detections, frame_id, save=False):
        # if there are no tracks yet, all detections are new tracks
        if len(self.tracks) == 0:
            for det in detections:
                t = Track()
                t.add_to_track(det)
                self.tracks.append(t)
            return True
        # assign detections to existing tracks
        for track in self.tracks:
            track.has_match = False
            predicted_next_bb = track.get_predicted_next_bb()
            for det in detections:
                # singular tracks search radially
                if track.is_singular():
                    iou = util.bb_intersection_over_union(predicted_next_bb, det.bbox)
                    dist = util.dist_btwn_bb_centroids(predicted_next_bb, det.bbox)
                    if dist < const.MAX_PIXELS_DIST_TRACK_START and iou > const.MIN_IOU_TRACK_START:
                        track.add_to_track(det)
                        track.has_match = True
                        track.num_misses = 0
                        break
                # established tracks search in predicted location
                elif track.is_established():
                    # TODO: get distance, iou to det
                    iou = util.bb_intersection_over_union(predicted_next_bb, det.bbox)
                    dist = util.dist_btwn_bb_centroids(predicted_next_bb, det.bbox)
                    if dist < const.MAX_PIXELS_DIST_TRACK and iou > const.MIN_IOU_TRACK:
                        track.add_to_track(det)
                        track.has_match = True
                        track.num_misses = 0
                        # TODO: handle case where decision is tough (2 detections very close)
                        break
            # if no track was assigned, give penalty to track
            if not track.has_match:
                # delete singular tracks that didn't get assigned (probably false detection)
                if track.num_misses > 0:
                    if track.is_singular():
                        track.delete_me = True
                    else:
                        # continue track using predicted state
                        track.propagate_track(frame_id=frame_id)
                track.num_misses += 1
            else:
                # reset match flag
                track.has_match = False

        for i, det in enumerate(detections):
            # if det hasn't been assigned yet, create new tracks
            if det.num_matches == 0:
                # print 'new track created. len(tracks)={}, num_det={}'.format(len(tracks),len(detections))
                t = Track()
                t.add_to_track(det)
                self.tracks.append(t)
            elif det.num_matches > 1:
                # TODO: resolve detections with multiple matches
                # print 'multiple assignment!! (num_matches({})={})'.format(i, det.num_matches)
                pass

        # cleanup any duplicate tracks that have formed (TODO: how do they form?)
        self.__delete_duplicate_tracks()
        # save dead tracks before deletion
        if save:
            self.__save_tracks_to_json()
        # remove dead tracks
        self.tracks = [t for t in self.tracks if (t.is_dead() is False and t.delete_me is False)]
        # for i, track in enumerate(tracks):
        #     print '{}: {}'.format(i, track.get_latest_bb())
        return True

    def update_tracks(self, detections, frame_id, save=False):
        self.__assign_detections_to_tracks_munkres(detections, frame_id)
        # self._assign_detections_to_tracks(detections, frame_id, save=save)

    def draw_tracks(self, img):
        for track in self.tracks:
            track.draw_history(img)
