import time

import cv2 as cv
import imutils as imu

import gui.gui as gui
from beans.traffic_signs import circle_signs_keys
from beans.traffic_signs import traffic_sign_hog_descriptor
from beans.traffic_signs import traffic_signs_classes
from beans.traffic_signs import triang_rect_signs_keys
from beans.traffic_signs import globallySignsList
from beans.traffic_signs import locallySignsList
from beans.traffic_signs import ruleLastSignsList
from classification.traffic_signs.traffic_signs_classifier import TrafficSignsClassifier
from detection.traffic_signs.traffic_signs_detector import TrafficSignsDetector
from utils import frameproc as frutil
from videocapture.videostream import VideoStream

resolution = 1280

trafficSignHog = traffic_sign_hog_descriptor()
circleTrafficSignsDetector = TrafficSignsDetector("resourses/cascades/circle_signs_cascade_detector.xml",
                                                  hogSvmDetectorPath="resourses/svm/circle_signs_hogsvm_detector_rbf_kfold10.xml",
                                                  hog=trafficSignHog)

triangRectTrafficSignsDetector = TrafficSignsDetector("resourses/cascades/triang_rect_signs_cascade_detector.xml",
                                                      hogSvmDetectorPath="resourses/svm/triang_rect_signs_hogsvm_detector_rbf_kfold10.xml",
                                                      hog=trafficSignHog)

trafficSignsClassifier = TrafficSignsClassifier("resourses/svm/traffic_signs_hogsvm_classifier_lin.xml")

# circle_signs_hitrate_histogram = {k: 0 for k in circle_signs_keys}
# triang_rect_signs_hitrate_histogram = {k: 0 for k in triang_rect_signs_keys}
# hitrate = 2

v = VideoStream('/media/datadisk/Videos/test2.mkv', 18)
# v = VideoStream('C:/Users/Ghost/Desktop/Work/video_tests/2.avi', 18).start()
# v = VideoStream().start()
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter("test.avi", fourcc, 20, (1280, 720))

flag = 0
fps = 0
i1 = 0
i2 = 0
coeff = 3
stack = []
stack_time = {k: 0 for k in zip(circle_signs_keys, triang_rect_signs_keys)}
time_limit = 5
stack_limit = 7
signal = False
signalCounter = 0

locallySignTimings = {k: 0 for k in locallySignsList}
localTime = 25

globallySignTimings = {k: 0 for k in globallySignsList}
globalTime = 60

v = v.start()
while True:
    t = time.time()
    # if fps != 0:
    #     coeff = int((fps / 2) * 0.75)

    vi = v.read()
    if vi is None:
        print("Camera is not found...")
        cv.waitKey(1000)
        v.stop()
        v = VideoStream().start()
        break

    frame = imu.resize(vi, width=resolution)
    fr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    fr_cut_main = frutil.cut_frame(fr, (0, 0), (fr.shape[1], int(0.7 * fr.shape[0])))

    if flag == 0:
        detectedSigns, calcSignsHogs = circleTrafficSignsDetector.find(fr_cut_main, 1.2, 4, (24, 24), (144, 144))
        i1 += 1
    else:
        detectedSigns, calcSignsHogs = triangRectTrafficSignsDetector.find(fr_cut_main, 1.2, 5,
                                                                           (24, 24), (144, 144))
        i2 += 1

    sign_list = []
    for sign, hog in zip(detectedSigns, calcSignsHogs):
        sign_class = trafficSignsClassifier.predict(hog)
        if traffic_signs_classes.get(sign_class) is not None:
            sign_list.append(sign_class)
            cv.rectangle(frame, (sign[0], sign[1]), (sign[0] + sign[2], sign[1] + sign[3]), (0, 255, 0), 2)

            if int(sign_class) in ruleLastSignsList or int(sign_class) == 107:
                stack = [x for x in stack if x not in ruleLastSignsList]

            if int(sign_class) == 108:
                stack.remove(40)

            if int(sign_class) == 107:
                stack = [x for x in stack if x not in circle_signs_keys]

            if int(sign_class) not in stack:
                stack.append(int(sign_class))
                stack_time[int(sign_class)] = time.time()
                signal = True
                signalCounter = 0

                if int(sign_class) in locallySignsList:
                    locallySignTimings[int(sign_class)] = time.time()

                if int(sign_class) in globallySignsList:
                    globallySignTimings[int(sign_class)] = time.time()

            else:
                # print(time.time() - stack_time[int(sign_class)])
                if time.time() - stack_time[int(sign_class)] > time_limit:
                    stack.remove(int(sign_class))
                    stack.append(int(sign_class))
                    stack_time[int(sign_class)] = time.time()
                    signal = True
                    signalCounter = 0

                    if int(sign_class) in locallySignsList:
                        locallySignTimings[int(sign_class)] = time.time()

                    if int(sign_class) in globallySignsList:
                        globallySignTimings[int(sign_class)] = time.time()

            if len(stack) > stack_limit:
                stack = stack[1::]

            # print(stack)
            # print([traffic_signs_classes[s] for s in stack])
            # print(traffic_signs_classes[int(sign_class)])

    flag = not flag
    t = time.time() - t
    if fps == 0:
        fps = round(1 / t, 2)
    else:
        fps = round((fps + round(1 / t, 2)) / 2, 2)

    for si in stack.copy():
        if int(si) in locallySignsList:
            # print(time.time() - locallySignTimings[int(si)])
            if time.time() - locallySignTimings[int(si)] >= localTime:
                locallySignTimings[int(si)] = 0
                stack.remove(int(si))

        if int(si) in globallySignsList:
            if time.time() - globallySignTimings[int(si)] >= globalTime:
                globallySignTimings[int(si)] = 0
                stack.remove(int(si))

    gui.showTrafficSignsOnLeftPanel(frame, stack.copy(), "resourses/images/traffic_signs/", isNewSign=signal)

    if signalCounter >= 5 * fps:
        signal = False
    elif signal:
        signalCounter += 1

    cv.namedWindow("CarEye", cv.WINDOW_NORMAL)
    cv.imshow("CarEye",
              cv.putText(frame, "FPS: " + str(fps), (frame.shape[1] - 292, frame.shape[0] - 16),
                         cv.FONT_HERSHEY_SIMPLEX,
                         1.5, (255, 255, 255), 3))

    # cv.createButton("Exit", )
    # cv.setWindowProperty("Frame", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    # cv.resizeWindow("Frame", 1680, 1050)
    # cv.moveWindow("Frame", 0, 0)
    # out.write(frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

v.stop()
# out.release()
cv.destroyAllWindows()
