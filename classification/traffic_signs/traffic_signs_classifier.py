import cv2 as cv

from beans.traffic_signs import traffic_signs_classes


class TrafficSignsClassifier:
    def __init__(self, hogSvmClassifierPath):
        self.hogSvmClassifier = cv.ml.SVM_create().load(hogSvmClassifierPath)

    def predict(self, hogRoi, onlyLabel=True):
        if onlyLabel:
            return self.hogSvmClassifier.predict(hogRoi)[1].ravel()[0]
        else:
            sign_class = self.hogSvmClassifier.predict(hogRoi)[1].ravel()[0]
            return traffic_signs_classes.get(sign_class)
