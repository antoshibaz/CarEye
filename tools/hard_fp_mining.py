# mining hard false positives images for training detectors
import cv2 as cv
import imutils as imu
import numpy as np

from beans.traffic_signs import traffic_sign_hog_descriptor
from utils import frameproc as frutil
from videocapture.videostream import VideoStream

# args
saveImagesPath = "D:/Downloads/traffic_signs_dataset/detection/train/negative_batch/1/hard_fp/hog/"
numberName = 12345
imgExt = "png"
typeHog = False
v = VideoStream('C:/Users/Ghost/Desktop/Work/video_tests/1.avi', 2).start()
# v = VideoStream('C:/Users/Ghost/Desktop/Work/video_tests/4K/5.mkv', 2).start()

detector = cv.CascadeClassifier("../resourses/cascades/triang_rect_signs_cascade_detector.xml")
if typeHog:
    hog = traffic_sign_hog_descriptor()
    svm = cv.ml.SVM_create()
    svm = svm.load("../resourses/svm/triang_rect_signs_hogsvm_detector_rbf_kfold10.xml")

while True:
    frame = imu.resize(v.read(), width=1280)
    fr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    detections = detector.detectMultiScale(fr, 1.2, minNeighbors=4, minSize=(24, 24),
                                           maxSize=(144, 144))

    for d in detections:
        if typeHog:
            sign_candidate = frutil.cut_frame(fr, (d[0], d[1]), (d[0] + d[2], d[1] + d[3]))
            if sign_candidate.shape[0] != 24 or sign_candidate.shape[1] != 24:
                sign_candidate = cv.resize(sign_candidate, (24, 24), interpolation=cv.INTER_CUBIC)

            descriptor = np.float32([hog.compute(sign_candidate)])
            predict = svm.predict(descriptor)[1].ravel()[0]
            if predict:
                cv.imwrite(saveImagesPath + str(numberName) + "." + imgExt,
                           frutil.cut_frame(frame, (d[0], d[1]), (d[0] + d[2], d[1] + d[3])))
                numberName += 1
                cv.rectangle(fr, (d[0], d[1]), (d[0] + d[2], d[1] + d[3]), (255, 0, 255), 2)
        else:
            cv.imwrite(saveImagesPath + str(numberName) + "." + imgExt,
                       frutil.cut_frame(frame, (d[0], d[1]), (d[0] + d[2], d[1] + d[3])))
            numberName += 1
            cv.rectangle(fr, (d[0], d[1]), (d[0] + d[2], d[1] + d[3]), (255, 0, 255), 2)

    cv.imshow("Frame", fr)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

v.stop()
cv.destroyAllWindows()
