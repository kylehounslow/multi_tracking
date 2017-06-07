"""
Kalman filter helps smooth noisy detections
"""
import cv2
import numpy as np

__author__ = "Kyle Hounslow"


# kalman filter with 6 states, 4 meas params
# mostly taken from http://www.robot-home.it/blog/en/software/ball-tracker-con-filtro-di-kalman/
class KalmanFilter(object):
    def __init__(self):
        self.first_run = True
        self.dynamParams = 6
        self.measureParams = 4
        self.kalman = cv2.KalmanFilter(dynamParams=self.dynamParams,
                                       measureParams=self.measureParams)  # 6 states, 4 measurement params
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 0, 1]], np.float32)
        # Transition matrix ( eg. p(k) = p(k-1) + v(k-1)*dT ), init dT = 1
        dT = 0.05  # assume ~20fps
        self.kalman.transitionMatrix = np.array([[1, 0, dT, 0, 0, 0],
                                                 [0, 1, 0, dT, 0, 0],
                                                 [0, 0, 1, 0, 0, 0],
                                                 [0, 0, 0, 1, 0, 0],
                                                 [0, 0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 0, 1]], np.float32)
        # process noise covariance matrix (rough values)
        self.kalman.processNoiseCov = np.array([[0.01, 0, 0, 0, 0, 0],
                                                [0, 0.01, 0, 0, 0, 0],
                                                [0, 0, 2.0, 0, 0, 0],
                                                [0, 0, 0, 1.0, 1.0, 0],
                                                [0, 0, 0, 0, 1.0, 0.01],
                                                [0, 0, 0, 0, 0, 0.01]], np.float32)
        # measurement noise covariance matrix (rough values)
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1

    def reset(self):
        self.kalman.__init__(dynamParams=self.dynamParams, measureParams=self.measureParams)
        self.first_run = True

    def get_predicted_bb(self):
        pred = self.kalman.predict().T[0]
        pred_bb = np.array([pred[0], pred[1], pred[4], pred[5]])

        return pred_bb

    def get_current_velocity(self):
        return self.velocity.copy()

    def get_current_unit_velocity(self):
        print self.velocity_unit_vec
        return self.velocity_unit_vec.copy()

    def correct(self, bb):
        # measurement is numpy array [[x1,y1,x2,y2]]
        measurement = np.array([bb], dtype=np.float32).T
        if self.first_run is True:
            self.kalman.statePre = np.array([measurement[0], measurement[1], [0], [0], measurement[2], measurement[3]],
                                            dtype=np.float32)
            self.first_run = False
        corr_bb = self.kalman.correct(measurement).T[0]
        self.velocity = np.array([corr_bb[2], corr_bb[3]])

        self.velocity_unit_vec = self.velocity / np.linalg.norm(self.velocity + 1e-6)
        return np.array([corr_bb[0], corr_bb[1], corr_bb[4], corr_bb[5]])

