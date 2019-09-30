# frames processing utils
import cv2
import numpy as np


def cut_frames(frame, *areas_argv):
    return [cut_frame(frame, area[0], area[1]) for area in areas_argv]


def cut_frame(frame, p1, p2):
    return frame[p1[1]:p2[1], p1[0]:p2[0]]


def gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)
