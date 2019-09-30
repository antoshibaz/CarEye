import cv2 as cv

import detection.traffic_signs.features_detectors as fd
from utils import frameproc as frproc


# implementation finding traffic signs - return detected traffic signs
class TrafficSignsDetector:
    def __init__(self, cascadeDetectorPath, hogSvmDetectorPath=None, hog=None):
        if cascadeDetectorPath is None:
            return
        elif hogSvmDetectorPath is None or hog is None:
            self.cascadeDetector = cv.CascadeClassifier(cascadeDetectorPath)
            self.hogSvmDetector = None
            self.hog = None
        elif hogSvmDetectorPath is not None and hog is not None:
            self.cascadeDetector = cv.CascadeClassifier(cascadeDetectorPath)
            self.hogSvmDetector = cv.ml.SVM_create().load(hogSvmDetectorPath)
            self.hog = hog

    def find(self, frame, slidingWindowScale, minNeighbors,
             minSignSize, maxSignSize):
        detected_traffic_signs = fd.cascade_detector(frame, self.cascadeDetector, slidingWindowScale, minNeighbors,
                                                     minSignSize, maxSignSize)

        if self.hogSvmDetector is not None and self.hog is not None:
            filtered_detected_traffic_signs = []
            hogs_for_signs = []
            for sign in detected_traffic_signs:
                sign_candidate = frproc.cut_frame(frame, (sign[0], sign[1]),
                                                  (sign[0] + sign[2], sign[1] + sign[3]))

                predict_and_hog = fd.hogsvm_detector(sign_candidate, self.hogSvmDetector, self.hog)
                if predict_and_hog[0]:
                    filtered_detected_traffic_signs.append(sign)
                    hogs_for_signs.append(predict_and_hog[1])

            return filtered_detected_traffic_signs, hogs_for_signs

        else:
            return detected_traffic_signs
