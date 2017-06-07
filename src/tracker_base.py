"""
Base class for different flavours of Tracking algorithms
"""
from abc import abstractmethod


class TrackerBase(object):
    def __init__(self):
        pass

    @abstractmethod
    def update_tracks(self, detections, frame_id, save=False):
        """
        :param detections: list of bounding boxes in [[x1,x2,y1,y2], ...] format
        :param frame_id: current frame id from camera capture
        :param save: whether to save history of tracks to file
        :return:
        """
        pass

    @abstractmethod
    def draw_tracks(self, img):
        """
        :param img: draw all current tracks to this img
        :return:
        """
        pass
