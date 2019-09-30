# testing hog+svm for traffic signs classification
import glob
import os
import time

import cv2 as cv
import numpy as np

import utils.metrics as metrics
from beans.traffic_signs import traffic_sign_hog_descriptor
from utils.print import print_one_line

# args
testPath = "D:/Downloads/traffic_signs_dataset/classification/test"
svmModelPath = "traffic_signs_hogsvm_classifier_lin.xml"
show_false_negatives = False

classes = [c[0] for c in os.walk(testPath)]
classes = classes[1:]
classes = [int(c.split("\\")[1]) for c in classes]
classes.sort()
print(classes)

hog = traffic_sign_hog_descriptor()
svm_new = cv.ml.SVM_create()
svm_new = svm_new.load(svmModelPath)
true_positive = 0
false_negative = 0
pos_count = 0
t = 0
for cl in classes:
    testPathCl = testPath + "/" + str(cl)
    imgs = glob.glob1(testPathCl, "*")
    pos_count += len(imgs)
    for i in imgs:
        fr = cv.imread(testPathCl + "/" + i)
        img = cv.cvtColor(fr, cv.COLOR_BGR2GRAY)

        ti = time.time()
        descriptor = np.float32([hog.compute(img)])
        predict = svm_new.predict(descriptor)[1].ravel()[0]
        t += time.time() - ti

        if int(predict) == cl:
            true_positive += 1
        else:
            false_negative += 1
            if show_false_negatives:
                cv.imshow("Pred:" + str(predict) + " Real:" + str(cl), fr)
                if cv.waitKey(0) & 0xFF == ord('q'):
                    break

    print_one_line("Class %d is computed..." % cl)

recall = metrics.recall_metric(true_positive, false_negative)
print("")
print("Samples count: " + str(pos_count))
print("True positives: " + str(true_positive))
print("False negatives: " + str(false_negative))
print("Recall: " + str(round(recall * 100, 2)) + "%")
print("Time prediction: " + str(t / pos_count))
