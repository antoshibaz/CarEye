# testing hog+svm for traffic signs detection
import glob
import time

import cv2 as cv
import numpy as np

import utils.frameproc as frproc
import utils.metrics as metrics
from beans.traffic_signs import traffic_sign_hog_descriptor
from utils.print import print_one_line

# args
svm_path = "../../resourses/svm/triang_rect_signs_hogsvm_detector_rbf_kfold10.xml"
test_images_pos = "D:/Downloads/traffic_signs_dataset/detection/test/images/"
test_images_neg = "D:/Downloads/traffic_signs_dataset/detection/test/images/"
test_pos_annotations_path = "D:/Downloads/traffic_signs_dataset/detection/test/1/test2.txt"
test_neg_annotations_path = "D:/Downloads/traffic_signs_dataset/detection/test/2/test1.txt"
# mode0 - predict: object or not object, mode 1 - find objects: bounding rects for objects
mode = 0
posWithRoi = True
negWithRoi = True
show_false_negatives = False
show_all = False

svm = cv.ml.SVM_create()
svm = svm.load(svm_path)
hog = traffic_sign_hog_descriptor()

true_positive = 0
false_positive = 0
false_negative = 0
if not mode:
    print("Mode 0 : Testing...")
    if posWithRoi:
        pos_images = open(test_pos_annotations_path).readlines()
    else:
        pos_images = glob.glob1(test_images_pos, "*")

    pos_count = len(pos_images)
    c = 0
    t = 0
    for pos in pos_images:
        if posWithRoi:
            params = pos.split(" ")
            x1 = int(params[2])
            y1 = int(params[3])
            x2 = x1 + int(params[4])
            y2 = y1 + int(params[5])
            img = frproc.cut_frame(cv.imread(test_images_pos + str(params[0])), (x1, y1), (x2, y2))
        else:
            img = cv.imread(test_images_pos + pos)

        if (img.shape[1], img.shape[0]) != hog.winSize:
            img = cv.resize(img, hog.winSize, interpolation=cv.INTER_CUBIC)

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ti = time.time()
        descriptor = np.float32([hog.compute(img)])
        predict = svm.predict(descriptor)[1].ravel()[0]
        t += time.time() - ti
        if predict:
            true_positive += 1
        else:
            false_negative += 1
            if show_false_negatives:
                cv.imshow("FN", img)
                if cv.waitKey(0) & 0xFF == ord('q'):
                    break

        c += 1
        print_one_line("POS: %d of %d completed" % (c, pos_count))

    if negWithRoi:
        neg_images = open(test_neg_annotations_path).readlines()
    else:
        neg_images = glob.glob1(test_images_neg, "*")

    neg_count = len(neg_images)
    c = 0
    print("")
    for neg in neg_images:
        if negWithRoi:
            params = neg.split(" ")
            x1 = int(params[2])
            y1 = int(params[3])
            x2 = x1 + int(params[4])
            y2 = y1 + int(params[5])
            img = frproc.cut_frame(cv.imread(test_images_pos + str(params[0])), (x1, y1), (x2, y2))
        else:
            img = cv.imread(test_images_pos + neg)

        if (img.shape[1], img.shape[0]) != hog.winSize:
            img = cv.resize(img, hog.winSize, interpolation=cv.INTER_CUBIC)

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        descriptor = np.float32([hog.compute(img)])
        predict = svm.predict(descriptor)[1].ravel()[0]
        if predict:
            false_positive += 1
        c += 1
        print_one_line("NEG: %d of %d completed" % (c, neg_count))

    print("")
    precision = metrics.precision_metric(true_positive, false_positive)
    recall = metrics.recall_metric(true_positive, false_negative)
    f1score = metrics.f1score_metric(precision, recall)
    print("Real positives: " + str(pos_count))
    print("Real negatives: " + str(neg_count))
    print("True positives: " + str(true_positive))
    print("False positives: " + str(false_positive))
    print("False negatives: " + str(false_negative))
    print("Precision: " + str(round(precision * 100, 2)) + "%")
    print("Recall: " + str(round(recall * 100, 2)) + "%")
    print("f1-score: " + str(round(f1score * 100, 2)) + "%")
    print("Time prediction: " + str(t / pos_count))
else:
    print("Mode 1 : Testing...")
